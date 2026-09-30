/*
 * AD7175_8.h
 *
 *  Created on: Jul 23, 2025
 *      Author: Lenovo
 */

#ifndef INC_AD7175_8_H_
#define INC_AD7175_8_H_

#ifdef __cplusplus
extern "C"{
#endif

#ifndef NUM_PDs
#define NUM_PDs 4          /* fallback if main.c forgot to define it */
#endif

#include "ad717x.h"
#include "ad7175_8_regs.h"
#include <string.h>
#include <errno.h>

typedef ad717x_dev AD7175_8;



static void build_init_struct(ad717x_init_param *ip, SPI_HandleTypeDef *hspi,
		GPIO_TypeDef *cs_port, uint16_t cs_pin) {
	memset(ip, 0, sizeof(*ip));

	/* -------- SPI layer fields (match your no_os_spi_init_param) ------- */
	ip->spi_init.hspi = hspi;
	ip->spi_init.cs_port = cs_port;
	ip->spi_init.cs_pin = cs_pin;

	/* -------- Register template / meta data ---------------------------- */

	/* Enable DATA_STAT (bit 6), disable CRC_EN (bits [3:2]) */
	ad7175_8_regs[AD717X_IFMODE_REG].value |= AD717X_IFMODE_REG_DATA_STAT;
	ad7175_8_regs[AD717X_IFMODE_REG].value &= ~AD717X_IFMODE_REG_CRC_EN;

	/* Disable SING_CYC (bit 13) */
	ad7175_8_regs[AD717X_ADCMODE_REG].value &= ~AD717X_ADCMODE_SING_CYC;

	/* Disable enhanced 50Hz/60Hz rejection for Setup 0 */
	ad7175_8_regs[AD717X_FILTCON0_REG].value &= ~AD717X_FILT_CONF_REG_ENHFILTEN;
	ad7175_8_regs[AD717X_FILTCON0_REG].value |= AD717X_FILT_CONF_REG_ENHFILT(6);
	ad7175_8_regs[AD717X_FILTCON0_REG].value |= AD717X_FILT_CONF_REG_ORDER(0);
	ip->regs = ad7175_8_regs;
	ip->num_regs = sizeof(ad7175_8_regs) / sizeof(ad7175_8_regs[0]);
	ip->active_device = ID_AD7175_8;
	ip->ref_en = true;

	ip->num_setups = 1;
	ip->num_channels = NUM_PDs;

	/* -------- SETUP0: unipolar, internal 2 V5 ref, buffers on ---------- */
	ip->setups[0] = (struct ad717x_channel_setup ) {
		.bi_unipolar  = false,
	    .ref_buff     = true,
		.input_buff   = false,
		.ref_source   = INTERNAL_REF
	};

	/* -------- FILTCON0: ODR code 0  (~250 kSPS) ------------------------ */
	ip->filter_configuration[0] = (struct ad717x_filtcon ) {
		    .sinc3_map = false, /* use Sinc5+Sinc1 branch            */
			.enhfilten = false, /* enhanced-50/60 Hz rejection OFF   */
			.enhfilt   = sps27_db47_ms36p7, /* don’t-care when enhfilten == 0    */
			.oder      = sinc5_sinc1, /* normal Sinc5+Sinc1 filter         */
			.odr       = 0      /* ODR code 0  → 50000 sps / chan    */
	};

	/* -------- Channel map 0-3 single-ended ----------------------------- (unchecked)*/
	for (uint8_t ch = 0; ch < NUM_PDs; ch++) {
		ip->chan_map[ch].channel_enable = true; /* disabled at start   */
		ip->chan_map[ch].setup_sel = 0; /* SETUP0              */
		ip->chan_map[ch].analog_inputs.ainp.pos_analog_input = (enum ad717x_analog_input) ch; /* AIN0..AIN3          */
		ip->chan_map[ch].analog_inputs.ainp.neg_analog_input = AINCOM; /* Connect AIN16 to GND */
	}

	ip->mode = CONTINUOUS; /* driver sets each read */

}

/* =============================  PUBLIC API  ============================= */

/* ---------------------------------------------------------------
 * Initial register setup: (the following change are made)
 * IFMODE register:    DATA_STAT = 1, CRC_EN = 00
 * ADCMODE register:   ING_CYC = 0
 * CHANNELx register:  CH_ENx = 1, SETUP_SELx = 000 (setup 0),
 *                     ANIPOSx = AINx, AINNEGx = AIN16 (single-ended)
 * SETUPCON0 register: BI_UNIPOLAR0 = 0 (unipolar code),
 * 					   REF_SEL0 = 10 (internal 2.5V reference)
 * 					   REFBUF0+ = 1, REFBUF0- = 1, (enable reference buffer)
 * 					   AINBUF0+ = 0, AINBUF0- = 1, (disable input buffer)
 * FILTCON0 register:  ODR0 = 00000 (250000 sps),
 * 					   ENHFILTEN0 = 0 (disable enhanced filter),
 * 					   ENHFILT0 = 010 (don't care)
 * ------------------------------------------------------------ */
int AD7175_8_Init(AD7175_8 **dev, SPI_HandleTypeDef *hspi,
		GPIO_TypeDef *cs_port, uint16_t cs_pin) {
	ad717x_init_param ip;
	build_init_struct(&ip, hspi, cs_port, cs_pin);
	return AD717X_Init(dev, ip); /* call the stock driver */
}


/* ---------------------------------------------------------------
 * Enable continuous-read   (IFMODE.CONT_READ = 1)
 * (write 0x0080 to register 0x02)
 * ------------------------------------------------------------ */
int ad7175_8_start_continuous(ad717x_dev *dev) {

	ad717x_st_reg *ifm = AD717X_GetReg(dev, AD717X_IFMODE_REG);
	if (!ifm)
		return -EINVAL;

	ifm->value |= AD717X_IFMODE_REG_CONT_READ;  /* cont-read    */

	return AD717X_WriteRegister(dev, AD717X_IFMODE_REG);
}

/* ------------------------------------------------------------------ */
/*  Helper to parse 24-bit sample + status byte                       */
/* ------------------------------------------------------------------ */
static inline void ad7175_8_unpack(uint8_t *rx4, uint8_t *ch, uint32_t *code24) {
	uint32_t raw = ((uint32_t) rx4[0] << 24) |
				   ((uint32_t) rx4[1] << 16) |
			       ((uint32_t) rx4[2] << 8)  |
			       rx4[3];

	*ch = raw & 0x0F; /* STATUS[3:0]   */
	*code24 = (raw >> 8) & 0xFFFFFF;
}

#ifdef __cplusplus
}
#endif

#endif /* INC_AD7175_8_H_ */
