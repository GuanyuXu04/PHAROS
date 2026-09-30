/*============================================================================
 * no_os_spi.h — Lightweight STM32‑HAL wrapper for ADI no‑OS drivers
 *===========================================================================*/
#ifndef _NO_OS_SPI_H_
#define _NO_OS_SPI_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "stm32l4xx_hal.h"   /* <-- change to your MCU family header */
/* ------------------------------------------------------------------------- */
/* Error helpers (minimal subset)                                            */
/* ------------------------------------------------------------------------- */
#ifndef ENOSYS
# define ENOSYS        38
#endif
#ifndef EINVAL
# define EINVAL        22
#endif
#ifndef HAL_OK
# define HAL_OK        0x00U
#endif

/* ------------------------------------------------------------------------- */
/* Public structures                                                         */
/* ------------------------------------------------------------------------- */

/* One‑to‑one mapping between an SPI peripheral & chip‑select GPIO. */
struct no_os_spi_desc {
    SPI_HandleTypeDef *hspi;      /* HAL SPI handle (e.g. &hspi3)        */
    GPIO_TypeDef      *cs_port;   /* GPIO port for manual CS (or NULL)   */
    uint16_t           cs_pin;    /* GPIO pin for manual CS              */
};

struct no_os_spi_init_param {
    SPI_HandleTypeDef *hspi;      /* HAL handle that is already Init()ed */
    GPIO_TypeDef      *cs_port;   /* GPIO port of CS                     */
    uint16_t           cs_pin;    /* GPIO pin of CS                      */
};

/* ------------------------------------------------------------------------- */
/* Function prototypes                                                       */
/* ------------------------------------------------------------------------- */
int32_t no_os_spi_init(struct no_os_spi_desc **desc,
                       const struct no_os_spi_init_param *param);
int32_t no_os_spi_remove(struct no_os_spi_desc *desc);
int32_t no_os_spi_write_and_read(struct no_os_spi_desc *desc,
                                 uint8_t *data,
                                 uint16_t bytes_number);

#ifdef __cplusplus
}
#endif

#endif /* _NO_OS_SPI_H_ */
