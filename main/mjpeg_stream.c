#include "mjpeg_stream.h"
#include "camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <stdio.h>
#include <string.h>

static const char *TAG = "MJPEG";

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

esp_err_t mjpeg_stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];  // Buffer suficientemente grande para la cabecera

    // Establecer el tipo de contenido multipart
    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    // Enviar encabezados HTTP iniciales
    res = httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    if (res != ESP_OK) {
        return res;
    }

    ESP_LOGI(TAG, "Cliente conectado al streaming MJPEG");

    while (true) {
        // Capturar frame (protegido por mutex en camera.c)
        fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGE(TAG, "Fallo al capturar frame");
            res = ESP_FAIL;
            break;
        }

        // Enviar boundary
        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res != ESP_OK) break;

        // Enviar cabecera de la imagen
        int hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned int)fb->len);
        if (hlen < 0 || hlen >= sizeof(part_buf)) {
            ESP_LOGE(TAG, "Cabecera de imagen truncada");
            camera_return_frame(fb);
            res = ESP_FAIL;
            break;
        }
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) {
            camera_return_frame(fb);
            break;
        }

        // Enviar datos JPEG
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        // Liberar frame inmediatamente después del envío
        camera_return_frame(fb);
        fb = NULL;

        if (res != ESP_OK) {
            break;
        }

        // Pequeño retardo para controlar la tasa (~20 fps)
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    // Si salimos por error, asegurarnos de liberar el frame si aún está retenido
    if (fb) {
        camera_return_frame(fb);
    }

    ESP_LOGW(TAG, "Streaming detenido");
    return res;
}