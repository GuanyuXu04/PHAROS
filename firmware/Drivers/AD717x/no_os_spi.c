/*============================================================================
 * no_os_spi.c — Lightweight STM32‑HAL wrapper implementation
 *===========================================================================*/
#include "no_os_spi.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>


/* ------------------------------------------------------------------------- */
/* API implementations                                                       */
/* ------------------------------------------------------------------------- */
int32_t no_os_spi_init(struct no_os_spi_desc **desc,
                       const struct no_os_spi_init_param *param)
{
    if (!desc || !param || !param->hspi)
        return -EINVAL;

    struct no_os_spi_desc *d = (struct no_os_spi_desc *)malloc(sizeof(*d));
    if (!d)
        return -EINVAL;

    d->hspi     = param->hspi;
    d->cs_port  = param->cs_port;
    d->cs_pin   = param->cs_pin;

    /* Ensure CS is de-asserted */
    //cs_high(d);

    *desc = d;
    return 0;
}

int32_t no_os_spi_remove(struct no_os_spi_desc *desc)
{
    if (!desc)
        return -EINVAL;

    free(desc);
    return 0;
}

int32_t no_os_spi_write_and_read(struct no_os_spi_desc *desc,
                                 uint8_t *data,
                                 uint16_t bytes_number)
{
    if (!desc || !desc->hspi || !data || bytes_number == 0)
        return -EINVAL;

    /* Use a separate RX buffer – safer with DMA-capable HAL API */
    uint8_t rx[256];                     /* stack buffer for ≤256-byte Xfers   */
    if (bytes_number > sizeof(rx))
        return -EINVAL;                  /* or malloc if you need bigger       */

    //cs_low(desc);
    HAL_StatusTypeDef st = HAL_SPI_TransmitReceive(desc->hspi, data, rx, bytes_number, 10u);  /* 10 ms timeout           */
    //cs_high(desc);

    if (st == HAL_OK)
    {
        memcpy(data, rx, bytes_number);  /* copy result back in-place */
        return 0;
    }

    /* Map HAL status to errno-style negative codes */
    if (st == HAL_TIMEOUT)   return -ETIMEDOUT;
    if (st == HAL_BUSY)      return -EBUSY;
    return -EIO;             /* HAL_ERROR or anything else */
}
