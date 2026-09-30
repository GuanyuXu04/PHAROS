# Optical Sensor Data Acquisition & LED Multiplexing

Firmware for **NUCLEO‑L432KC** that

1. Reads multiple **ADPD2211** photodiodes through an **AD7175‑8** ADC running in continuous‑read mode, and
2. Time‑division multiplexes illumination from **SK9822‑EC20** addressable LEDs.

Designed for high‑speed sensing with minimal MCU overhead and a 2 Mbit s⁻¹ uplink to the host PC.

<img width="362" height="361" alt="image" src="https://github.com/user-attachments/assets/faa30987-2440-4382-b758-cccd1818bfc7" />

<img width="1075" height="958" alt="image" src="https://github.com/user-attachments/assets/d9bf045e-78c8-4ed2-8cc2-aec65b2b5425" />


---

## Hardware Overview

| Part          | Qty | Notes                                  |
| ------------- | --- | -------------------------------------- |
| NUCLEO‑L432KC | 1   | STM32L4 Arm‑M4 @ 80 MHz                |
| AD7175‑8      | 1   | 24‑bit Σ‑Δ ADC, ±2.5 V, RDY on PB4     |
| ADPD2211      | *n* | Photodiode / TIA front‑end             |
| SK9822‑EC20   | *m* | SPI‑style RGB LEDs (APA102‑compatible) |

---

## MCU Peripheral Configuration (CubeMX / *.ioc*)

### SPI1 — LED Driver

| Property        | Value                                         |
| --------------- | --------------------------------------------- |
| Mode            | Full‑Duplex Master                            |
| Data Size       | 8 bit                                         |
| Clock Prescaler | 2 → **30 Mbit s⁻¹** (max SK9822: 30 Mbit s⁻¹) |
| CPOL / CPHA     | Low / 1st Edge (Mode 0)                       |
| Pins            | PA5 (SCK), PA6 (MISO), PA7 (MOSI)             |

### SPI3 — ADC Interface

| Property        | Value                                           |
| --------------- | ----------------------------------------------- |
| Mode            | Full‑Duplex Master                              |
| Data Size       | 8 bit                                           |
| Clock Prescaler | 2 → **24 Mbit s⁻¹**                             |
| CPOL / CPHA     | High / 2nd Edge (Mode 3 – required by AD7175‑8) |
| Pins            | PB3 (SCK), **PB4 (MISO)**, PB5 (MOSI)   |
| CS (GPIO)       | PA12 (active low)                               |

### USART2 — PC Uplink

| Property | Value                  |
| -------- | ---------------------- |
| Baud     | **2 000 000 Bits/s**   |
| Frame    | 8‑N‑1                  |
| DMA      | TX → DMA1‑Ch7 (Normal) |
| Pins     | PA2 (TX), PA15 (RX)    |

### NVIC

| Interrupt             | Priority | Notes            |
| --------------------- | -------- | ---------------- |
| DMA1\_Ch7 (USART2 TX) | 0        | Max throughput   |
| USART2                | 0        | Error handling   |
| EXTI4 (PB4)           | 0        | Conversion ready |

---

## PB4 Dual‑Role (SPI ⇄ EXTI)

**PB4** acts as the ADC `MISO` during SPI transfers and as the **RDY** falling‑edge interrupt line between conversions.
To toggle the alternate function of PB4 between SPI3_MISO and GPIO_EXTI4, use the following helper functions:
```c
/* Runtime pin‑mux helpers -----------------------------------------------*/
/** Switch PB4 from SPI3_MISO ➜ EXTI (falling edge) */
static inline void rdy_to_exti(void) {
    GPIOB->MODER &= ~(0x3u << (4 * 2));        /* Input mode      */
}
/** Switch PB4 from EXTI ➜ SPI3_MISO */
static inline void rdy_to_spi(void) {
    GPIOB->MODER = (GPIOB->MODER & ~(0x3u << (4 * 2))) | (0x2u << (4 * 2));
}
```
To temporarily disable or enable EXTI interrupt (e.g., when updating LED or transmitting data to PC), use the following helper functions:
```c
/* EXTI mask control ------------------------------------------------------*/
static inline void rdy_exti_enable(void) {
    EXTI->PR1 = (1U << 4);                     /* Clear pending  */
    EXTI->IMR1 |= (1U << 4);                   /* Un‑mask line‑4 */
    NVIC_ClearPendingIRQ(EXTI4_IRQn);
    NVIC_EnableIRQ(EXTI4_IRQn);
}
static inline void rdy_exti_disable(void) {
    EXTI->IMR1 &= ~(1U << 4);                  /* Mask line‑4    */
    NVIC_DisableIRQ(EXTI4_IRQn);
}
```

---

## GPIO & EXTI Setup (add to **`MX_GPIO_Init()`**) 

CubeMX generates `MX_GPIO_Init()` first; our custom code must **run after all other MX\_… initializations** or be placed in `USER CODE BEGIN 2`.

```c
/* USER CODE BEGIN MX_GPIO_Init_2 -----------------------------------------*/
/* Configure PB4 as EXTI4 (falling‑edge) */
RCC->AHB2ENR  |= RCC_AHB2ENR_GPIOBEN;          // Enable GPIOB clock
RCC->APB2ENR  |= RCC_APB2ENR_SYSCFGEN;         // Enable SYSCFG clock
SYSCFG->EXTICR[1] = (SYSCFG->EXTICR[1] & ~SYSCFG_EXTICR2_EXTI4_Msk)
                  | (0x1u << SYSCFG_EXTICR2_EXTI4_Pos);  // PB4 → EXTI4
EXTI->RTSR1 &= ~EXTI_RTSR1_RT4;                // Disable rising trigger
EXTI->FTSR1 |= EXTI_FTSR1_FT4;                 // Enable falling trigger
EXTI->PR1    = EXTI_PR1_PIF4;                  // Clear any pending flag
EXTI->IMR1  |= EXTI_IMR1_IM4;                  // Un‑mask line‑4
HAL_NVIC_SetPriority(EXTI4_IRQn, 0, 0);
/* USER CODE END MX_GPIO_Init_2 -------------------------------------------*/
```

> **Regeneration Tip**  Whenever you re‑generate code from CubeMX, **verify** that `MX_GPIO_Init()` still executes *after* SPI/USART/DMA setups, or move the snippet to `USER CODE BEGIN 2`.

---

## Runtime Flow

1. **LED‑scan loop** selects LED *i* (SPI1).
2. AD7175‑8 runs in *continuous read*; each falling RDY on PB4 triggers `EXTI4_IRQHandler()`, which reads in one channel data.
3. Handler:

   1. `rdy_to_spi()`   → switch PB4 to MISO.
   2. Read 4 bytes over SPI3 (24‑bit conversion + 8 bit status).
   3. Unpack the frame to extract channel and data. 
   4. `rdy_to_exti()`  → return PB4 to EXTI.
4. After collecting one full LED frame, disable `RDY` interrupt, update LED pattern
5. Stream data to PC via USART2 DMA. Re‑enable EXTI when the bus is free.

The waveform on a logic analyzer looks like the following:
<img width="1010" height="416" alt="image" src="https://github.com/user-attachments/assets/156682a9-f74d-4d37-ae61-a740db44b862" />
<img width="838" height="421" alt="image" src="https://github.com/user-attachments/assets/9dd20722-48b6-4acd-addf-886a02c8eb0b" />

Under the fastest mode, the ADC produces one data every 37 μs. 

---


## References

* **ADPD2211** Datasheet (Analog Devices)
* **AD7175‑8** Datasheet & Application Notes (Analog Devices)
* **SK9822‑EC20** Datasheet (Jiangsu RGB LED Co.)
* **STM32L432** Reference Manual (RM0394)
