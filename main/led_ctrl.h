#pragma once
#include "esp_err.h"

esp_err_t led_init(void);
esp_err_t led_set_intensity(uint8_t percent);