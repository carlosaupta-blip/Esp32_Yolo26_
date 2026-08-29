#pragma once
#include "esp_camera.h"
#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Pines de la cámara OV2640 (placa propia)
 * --------------------------------------------------------------------- */
#define CAM_PIN_PWDN   38
#define CAM_PIN_RESET  -1
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK   13
#define CAM_PIN_XCLK   15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D0     11
#define CAM_PIN_D1      9
#define CAM_PIN_D2      8
#define CAM_PIN_D3     10
#define CAM_PIN_D4     12
#define CAM_PIN_D5     18
#define CAM_PIN_D6     17
#define CAM_PIN_D7     16


esp_err_t camera_init(void);
esp_err_t camera_deinit(void);
esp_err_t camera_set_param(const char *name, int value);
sensor_t *camera_get_sensor(void);
camera_fb_t *camera_capture_frame(void);
void camera_return_frame(camera_fb_t *fb);