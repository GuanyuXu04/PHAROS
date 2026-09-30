/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "AD7175_8.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi3;

UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_tx;

/* USER CODE BEGIN PV */
AD7175_8 *adc;
uint32_t PD_readings[NUM_PDs];
uint32_t data[NUM_PDs][NUM_LEDs];

static uint8_t led_frame[LED_FRAME_SIZE];
static volatile uint8_t led_idx = 0;

static char tx_frame[TX_FRAME_SIZE];
static uint16_t tx_len = 0;
static volatile bool tx_busy = false;

static uint8_t adc_rx[4];
static volatile uint8_t got_mask = 0;
static volatile uint8_t sample_cnt = 0;
uint8_t zeros[4] = { 0 }; /* drives DIN low */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_SPI1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_SPI3_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ---------------------------------------------------------------------------*/
/* Enable the interrupt on PB4                                                */
/* REQUIRE: PB4 already in EXTI mode.                                         */
/*----------------------------------------------------------------------------*/
static inline void rdy_exti_enable() {
	EXTI->PR1  = (1U << 4);   /* 1) clear any stale pending flag  */
	EXTI->IMR1 |= (1U << 4);  /* 2) un-mask the line              */
	NVIC_ClearPendingIRQ(EXTI4_IRQn);
	NVIC_EnableIRQ(EXTI4_IRQn);
}

/* ---------------------------------------------------------------------------*/
/* Disable the interrupt on PB4                                               */
/* REQUIRE: PB4 already in EXTI mode.                                         */
/*----------------------------------------------------------------------------*/
static inline void rdy_exti_disable() {
	EXTI->IMR1 &= ~(1U << 4); /* mask line-4; PR1 can still set   */
	NVIC_DisableIRQ(EXTI4_IRQn);
}

/*
 * Change PB4 from SPI3_MISO to EXTI
 * Falling edge triggered external interrupt
 * REQUIRE: PB4 in SPI3_MISO mode.
 */
static inline void rdy_to_exti(){
	GPIOB->MODER &= ~(0x3u << (4 * 2));   /* one AHB write   */
}

/*
 * Change PB4 from EXTI to SPI3_MISO
 * REQUIRE: PB4 in EXTI mode
 */
static inline void rdy_to_spi(){
	GPIOB->MODER = (GPIOB->MODER & ~(0x3u << (4 * 2))) | (0x2u << (4 * 2));
}

/* ========================================================================= */
/*  1.  Unsigned-32-bit → decimal ASCII                                      */
/*      – works for any 0 … 4 294 967 295                                    */
/*      – returns pointer to first digit (string is NUL-terminated)          */
/* ========================================================================= */
static inline char* u32_to_dec(uint32_t v, char *p_end) {
	char *p = p_end;
	*--p = '\0'; /* terminate                          */
	do {
		*--p = (char) ('0' + (v % 10u));
		v /= 10u;
	} while (v);
	return p;
}

/* ========================================================================= */
/*  2.  Build one CSV frame in `tx_frame` and set `tx_len`                   */
/*      Frame:  <tick_ms mod 100000>,d00,d01,…,d(P,N-1)\r\n                  */
/* ========================================================================= */
static void build_tx_frame(uint32_t tick_ms) {
	char *p = tx_frame;

	/* ---------- timestamp (5 decimal digits max) ----------------------- */
	{
		char tmp[11];
		char *s = u32_to_dec(tick_ms % 100000u, &tmp[10]);
		while (*s)
			*p++ = *s++;
		*p++ = ','; /* field separator                    */
	}

	/* ---------- sensor matrix ------------------------------------------ */
	for (uint8_t led = 0; led < NUM_LEDs; ++led) {
		for (uint8_t pd = 0; pd < NUM_PDs; ++pd) {

			/* convert 24-bit code (0…0xFFFFFF) -------------------------- */
			char tmp[12]; /* 8 digits + '\0' + spare            */
			char *s = u32_to_dec(data[pd][led] & 0xFFFFFFu, &tmp[11]);
			while (*s)
				*p++ = *s++;

			/* add comma or line terminator ------------------------------ */
			*p++ = (led == NUM_LEDs - 1 && pd == NUM_PDs - 1) ? '\r' : ',';
		}
	}
	*p++ = '\n';

	/* total length for UART/DMA transfer -------------------------------- */
	tx_len = (uint16_t) (p - tx_frame);
}

static void init_frame(void) {
	memset(led_frame, 0x00, 4);
	memset(&led_frame[4 + NUM_LEDs * 4], 0xFF, (NUM_LEDs + 15) / 16);
}

static inline void set_pixel(uint8_t idx, uint8_t br, uint8_t g, uint8_t r,
		uint8_t b) {
	uint8_t *p = &led_frame[4 + idx * 4];
	p[0] = 0xE0 | (br & 0x1F);
	p[1] = g;
	p[2] = r;
	p[3] = b;
}

/* =========================================================================
 *   EXTI-falling on PA8  (DOUT/RDY ↓)  →  start SPI3 DMA read
 * ========================================================================= */
void HAL_GPIO_EXTI_Callback(uint16_t pin) {
	if (pin != RDY_PIN)
		return;

	/* PB4 switch to SPI3_MISO */
	rdy_to_spi();
	HAL_SPI_TransmitReceive(&hspi3, zeros, adc_rx, 4, HAL_MAX_DELAY);
	sample_cnt++;

	/* Wait for NUM_PDs samples to arrive */
	if (got_mask == (1u << NUM_PDs) - 1u) {
		/* PB4 switch to EXTI, disable interrupt, wait for LED transfer */
		rdy_to_exti();
		rdy_exti_disable();
		__HAL_GPIO_EXTI_CLEAR_IT(RDY_PIN);
		return;
	}

	uint8_t ch;
	uint32_t code;
	ad7175_8_unpack(adc_rx, &ch, &code);
	if (ch < NUM_PDs) {
		PD_readings[ch] = code;
		got_mask |= (1u << ch);
	}

	/* PB4 switch to EXTI,
	 * Enable interrupt, hear next falling edge */
	EXTI->PR1  = (1U << 4);   /* 1) clear any stale pending flag  */
	rdy_to_exti();
}

/* =========================================================================
 *   UART-DMA complete → release buffer
 * ========================================================================= */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
	if (huart == &huart2)
		tx_busy = false;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_SPI1_Init();
  MX_USART2_UART_Init();
  MX_SPI3_Init();
  /* USER CODE BEGIN 2 */
	HAL_GPIO_WritePin(AD7175_CS_PORT, AD7175_CS_PIN, GPIO_PIN_RESET);
	/* PB4 in SPI3_MISO */
	if (AD7175_8_Init(&adc, &hspi3, GPIOA, GPIO_PIN_12) != 0)
		Error_Handler();
	ad7175_8_start_continuous(adc);
	init_frame();

	rdy_to_exti();

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1) {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		uint32_t now = HAL_GetTick();
		for (uint8_t led = 0; led < NUM_LEDs; led++) {

			// Update LED
			uint8_t prev = (led + NUM_LEDs - 1) % NUM_LEDs;
			set_pixel(prev, 0, 0, 0, 0);
			set_pixel(led, MAX_BRIGHTNESS, 255, 255, 255);
			HAL_SPI_Transmit(&hspi1, led_frame, LED_FRAME_SIZE, HAL_MAX_DELAY);

			// Clear flag & enable EXTI
			got_mask = 0;
			sample_cnt = 0;
			rdy_exti_enable(); // Start reading from ADC
			while (got_mask != (1u << NUM_PDs) - 1u)
				__WFI(); /* sleep until done */
			rdy_exti_disable(); // Disable interrupt while building csv
			for (uint8_t pd = 0; pd < NUM_PDs; pd++) {
				data[pd][led] = PD_readings[pd];
				//printf("data[%d][%d] = %ld\r\n", pd, led, PD_readings[pd]);
			}
		}

		build_tx_frame(now);
		while (tx_busy); /* wait previous frame */
		tx_busy = true;
		HAL_UART_Transmit_DMA(&huart2, (uint8_t*) tx_frame, tx_len);
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 15;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief SPI3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI3_Init(void)
{

  /* USER CODE BEGIN SPI3_Init 0 */

  /* USER CODE END SPI3_Init 0 */

  /* USER CODE BEGIN SPI3_Init 1 */

  /* USER CODE END SPI3_Init 1 */
  /* SPI3 parameter configuration*/
  hspi3.Instance = SPI3;
  hspi3.Init.Mode = SPI_MODE_MASTER;
  hspi3.Init.Direction = SPI_DIRECTION_2LINES;
  hspi3.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi3.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi3.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi3.Init.NSS = SPI_NSS_SOFT;
  hspi3.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
  hspi3.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi3.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi3.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi3.Init.CRCPolynomial = 7;
  hspi3.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi3.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
  if (HAL_SPI_Init(&hspi3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI3_Init 2 */

  /* USER CODE END SPI3_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 2000000;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel7_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_12, GPIO_PIN_SET);

  /*Configure GPIO pin : PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : PA12 */
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* EXTI interrupt init*/

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  /* Set PB4 as an external interrupt triggered on falling edge */
  RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;						/* Set AHB clock to enable bank B */
  RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;						/* Enable SYSCFG clock */

  SYSCFG->EXTICR[1] &= ~SYSCFG_EXTICR2_EXTI4_Msk;
  SYSCFG->EXTICR[1] |= (0x1u << SYSCFG_EXTICR2_EXTI4_Pos);	/* Map PB4 to external interrupt 4 */

  EXTI->RTSR1 &= ~EXTI_RTSR1_RT4;
  EXTI->FTSR1 |= EXTI_FTSR1_FT4;
  EXTI->PR1 = EXTI_PR1_PIF4;
  EXTI->IMR1 |= EXTI_IMR1_IM4;

  HAL_NVIC_SetPriority(EXTI4_IRQn, 0, 0);
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif /* __GNUC__ */
PUTCHAR_PROTOTYPE {
	HAL_UART_Transmit(&huart2, (uint8_t*) &ch, 1, 0xFFFF);
	return ch;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
