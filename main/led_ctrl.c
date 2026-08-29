#include "led_ctrl.h"
#include "led_strip.h"
#include "esp_log.h"

static led_strip_handle_t strip = NULL;

esp_err_t led_init(void) {
    ESP_LOGW("led", "-> led_init");
    led_strip_config_t config = {
        .strip_gpio_num = 48,
        .max_leds = 1,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,  // antes: .led_pixel_format = LED_PIXEL_FORMAT_GRB,
        .led_model = LED_MODEL_WS2812,
        .flags = { .invert_out = false },
    };
    led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 64,
        .flags = { .with_dma = false },
    };
    esp_err_t err = led_strip_new_rmt_device(&config, &rmt_config, &strip);
    if (err == ESP_OK) led_strip_clear(strip);
    return err;
}

esp_err_t led_set_intensity(uint8_t percent) {
    if (!strip) return ESP_ERR_INVALID_STATE;
    uint8_t val = (uint8_t)(percent * 255 / 100);
    led_strip_set_pixel(strip, 0, val, val, val);
    return led_strip_refresh(strip);
}