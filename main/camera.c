#include "camera.h"
#include "config_store.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <string.h>

static const char *TAG = "CAM";

static SemaphoreHandle_t cam_mutex = NULL;
static camera_config_t camera_config = {0};

/* ---------------------------------------------------------------------
 * Log de entrada a funciones (amarillo según pedido)
 * ------------------------------------------------------------------- */
static void log_entry(const char *func) {
    ESP_LOGW(TAG, "-> %s", func);
}

/* ---------------------------------------------------------------------
 * Inicialización de la cámara OV2640 con driver esp32-camera v2.1.7
 * ------------------------------------------------------------------- */
esp_err_t camera_init(void) {
    log_entry(__func__);

    if (cam_mutex == NULL) {
        cam_mutex = xSemaphoreCreateMutex();
        if (cam_mutex == NULL) {
            ESP_LOGE(TAG, "No se pudo crear mutex de cámara");
            return ESP_ERR_NO_MEM;
        }
    }

    // Configuración base del sensor y del bus de cámara
    camera_config = (camera_config_t){
        .pin_pwdn      = CAM_PIN_PWDN,
        .pin_reset     = CAM_PIN_RESET,
        .pin_xclk      = CAM_PIN_XCLK,
        .pin_sccb_sda  = CAM_PIN_SIOD,
        .pin_sccb_scl  = CAM_PIN_SIOC,

        .pin_d7        = CAM_PIN_D7,
        .pin_d6        = CAM_PIN_D6,
        .pin_d5        = CAM_PIN_D5,
        .pin_d4        = CAM_PIN_D4,
        .pin_d3        = CAM_PIN_D3,
        .pin_d2        = CAM_PIN_D2,
        .pin_d1        = CAM_PIN_D1,
        .pin_d0        = CAM_PIN_D0,
        .pin_vsync     = CAM_PIN_VSYNC,
        .pin_href      = CAM_PIN_HREF,
        .pin_pclk      = CAM_PIN_PCLK,

        // Reloj del sensor
        .xclk_freq_hz  = 20000000,          // 20 MHz recomendado para OV2640
        .ledc_timer    = LEDC_TIMER_0,
        .ledc_channel  = LEDC_CHANNEL_0,

        // Formato y resolución inicial
        .pixel_format  = PIXFORMAT_JPEG,    // JPEG comprimido por hardware
        .frame_size    = FRAMESIZE_QVGA,    // 320x240 por defecto (buena relación fluidez/calidad)
        .jpeg_quality  = 13,                // Calidad media-alta (0=mejor, 63=menor)

        // Buffers de frame: 2 activan modo continuo por hardware (mayor FPS)
        .fb_count      = 2,
        .fb_location   = CAMERA_FB_IN_PSRAM, // Usar PSRAM para almacenar frames
        .grab_mode     = CAMERA_GRAB_LATEST, // Siempre entregar el frame más reciente
    };

    // Inicializar el driver
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Fallo al inicializar cámara: 0x%x", err);
        return err;
    }

    ESP_LOGI(TAG, "Cámara inicializada correctamente (driver v2.1.7)");

    // Aplicar configuración persistida en NVS (si existe)
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        int32_t val;
        // Parámetros de imagen
        if (config_store_get_cam("quality", &val) == ESP_OK) s->set_quality(s, val);
        if (config_store_get_cam("brightness", &val) == ESP_OK) s->set_brightness(s, val);
        if (config_store_get_cam("contrast", &val) == ESP_OK) s->set_contrast(s, val);
        if (config_store_get_cam("saturation", &val) == ESP_OK) s->set_saturation(s, val);
        if (config_store_get_cam("special_effect", &val) == ESP_OK) s->set_special_effect(s, val);
        if (config_store_get_cam("wb_mode", &val) == ESP_OK) s->set_wb_mode(s, val);
        if (config_store_get_cam("aec", &val) == ESP_OK) s->set_aec2(s, val);
        if (config_store_get_cam("aec2", &val) == ESP_OK) s->set_aec2(s, val);
        if (config_store_get_cam("ae_level", &val) == ESP_OK) s->set_ae_level(s, val);
        if (config_store_get_cam("agc", &val) == ESP_OK) s->set_agc_gain(s, val);
        if (config_store_get_cam("agc_gain", &val) == ESP_OK) s->set_agc_gain(s, val);
        if (config_store_get_cam("gainceiling", &val) == ESP_OK) s->set_gainceiling(s, (gainceiling_t)val);
        if (config_store_get_cam("bpc", &val) == ESP_OK) s->set_bpc(s, val);
        if (config_store_get_cam("wpc", &val) == ESP_OK) s->set_wpc(s, val);
        if (config_store_get_cam("raw_gma", &val) == ESP_OK) s->set_raw_gma(s, val);
        if (config_store_get_cam("lenc", &val) == ESP_OK) s->set_lenc(s, val);
        if (config_store_get_cam("hmirror", &val) == ESP_OK) s->set_hmirror(s, val);
        if (config_store_get_cam("vflip", &val) == ESP_OK) s->set_vflip(s, val);
        if (config_store_get_cam("dcw", &val) == ESP_OK) s->set_dcw(s, val);
        if (config_store_get_cam("colorbar", &val) == ESP_OK) s->set_colorbar(s, val);
        if (config_store_get_cam("sharpness", &val) == ESP_OK) s->set_sharpness(s, val);
        if (config_store_get_cam("denoise", &val) == ESP_OK) s->set_denoise(s, val);

        // La resolución (framesize) se aplica al final, ya que puede reiniciar internamente
        if (config_store_get_cam("framesize", &val) == ESP_OK) {
            s->set_framesize(s, (framesize_t)val);
            ESP_LOGI(TAG, "Resolución aplicada: %d", val);
        }
    } else {
        ESP_LOGW(TAG, "No se pudo obtener sensor_t para aplicar ajustes");
    }

    return ESP_OK;
}

/* ---------------------------------------------------------------------
 * Desinicialización (liberar recursos)
 * ------------------------------------------------------------------- */
esp_err_t camera_deinit(void) {
    log_entry(__func__);
    if (cam_mutex) {
        vSemaphoreDelete(cam_mutex);
        cam_mutex = NULL;
    }
    return esp_camera_deinit();
}

/* ---------------------------------------------------------------------
 * Ajuste de un parámetro en caliente y guardado en NVS
 * name: nombre del parámetro (sin espacios)
 * value: valor entero (o booleano 0/1)
 * ------------------------------------------------------------------- */
esp_err_t camera_set_param(const char *name, int value) {
    log_entry(__func__);
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        ESP_LOGE(TAG, "Sensor no disponible");
        return ESP_ERR_INVALID_STATE;
    }

    if (cam_mutex) xSemaphoreTake(cam_mutex, portMAX_DELAY);

    esp_err_t ret = ESP_OK;

    // Aplicar al sensor
    if (strcmp(name, "quality") == 0) {
        s->set_quality(s, value);
    } else if (strcmp(name, "brightness") == 0) {
        s->set_brightness(s, value);
    } else if (strcmp(name, "contrast") == 0) {
        s->set_contrast(s, value);
    } else if (strcmp(name, "saturation") == 0) {
        s->set_saturation(s, value);
    } else if (strcmp(name, "special_effect") == 0) {
        s->set_special_effect(s, value);
    } else if (strcmp(name, "wb_mode") == 0) {
        s->set_wb_mode(s, value);
    } else if (strcmp(name, "aec") == 0) {
        s->set_aec2(s, value);
    } else if (strcmp(name, "aec2") == 0) {
        s->set_aec2(s, value);
    } else if (strcmp(name, "ae_level") == 0) {
        s->set_ae_level(s, value);
    } else if (strcmp(name, "agc") == 0) {
        s->set_agc_gain(s, value);
    } else if (strcmp(name, "agc_gain") == 0) {
        s->set_agc_gain(s, value);
    } else if (strcmp(name, "gainceiling") == 0) {
        s->set_gainceiling(s, (gainceiling_t)value);
    } else if (strcmp(name, "bpc") == 0) {
        s->set_bpc(s, value);
    } else if (strcmp(name, "wpc") == 0) {
        s->set_wpc(s, value);
    } else if (strcmp(name, "raw_gma") == 0) {
        s->set_raw_gma(s, value);
    } else if (strcmp(name, "lenc") == 0) {
        s->set_lenc(s, value);
    } else if (strcmp(name, "hmirror") == 0) {
        s->set_hmirror(s, value);
    } else if (strcmp(name, "vflip") == 0) {
        s->set_vflip(s, value);
    } else if (strcmp(name, "dcw") == 0) {
        s->set_dcw(s, value);
    } else if (strcmp(name, "colorbar") == 0) {
        s->set_colorbar(s, value);
    } else if (strcmp(name, "sharpness") == 0) {
        s->set_sharpness(s, value);
    } else if (strcmp(name, "denoise") == 0) {
        s->set_denoise(s, value);
    } else if (strcmp(name, "framesize") == 0) {
        // Al cambiar framesize, el driver puede reiniciar internamente
        s->set_framesize(s, (framesize_t)value);
        ESP_LOGW(TAG, "Resolución cambiada a %d", value);
    } else {
        ESP_LOGW(TAG, "Parámetro desconocido: %s", name);
        ret = ESP_ERR_INVALID_ARG;
    }

    if (cam_mutex) xSemaphoreGive(cam_mutex);

    // Guardar en NVS si la operación fue exitosa
    if (ret == ESP_OK) {
        config_store_set_cam(name, value);
    }

    return ret;
}

/* ---------------------------------------------------------------------
 * Obtener el sensor_t (por si se necesita manipulación directa)
 * ------------------------------------------------------------------- */
sensor_t *camera_get_sensor(void) {
    return esp_camera_sensor_get();
}

/* ---------------------------------------------------------------------
 * Capturar un frame (protegiendo con mutex)
 * ------------------------------------------------------------------- */
camera_fb_t *camera_capture_frame(void) {
    if (cam_mutex) xSemaphoreTake(cam_mutex, portMAX_DELAY);
    camera_fb_t *fb = esp_camera_fb_get();
    if (cam_mutex) xSemaphoreGive(cam_mutex);
    return fb;
}

/* ---------------------------------------------------------------------
 * Devolver el frame al driver
 * ------------------------------------------------------------------- */
void camera_return_frame(camera_fb_t *fb) {
    if (fb) {
        esp_camera_fb_return(fb);
    }
}