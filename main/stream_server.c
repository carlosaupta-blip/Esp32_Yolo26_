#include "stream_server.h"
#include "camera.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <unistd.h>

static const char *TAG = "stream_srv";

#define STREAM_PORT 81
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace;boundary=" PART_BOUNDARY "\r\n\r\n";
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static void stream_task(void *arg) {
    int listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock < 0) {
        ESP_LOGE(TAG, "No se pudo crear socket");
        vTaskDelete(NULL);
        return;
    }

    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in server_addr = {0};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(STREAM_PORT);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(listen_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) != 0) {
        ESP_LOGE(TAG, "Error en bind");
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }

    if (listen(listen_sock, 2) != 0) {
        ESP_LOGE(TAG, "Error en listen");
        close(listen_sock);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Servidor de streaming MJPEG en puerto %d", STREAM_PORT);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_sock = accept(listen_sock, (struct sockaddr *)&client_addr, &client_len);
        if (client_sock < 0) {
            ESP_LOGE(TAG, "Error en accept");
            continue;
        }

        ESP_LOGI(TAG, "Cliente conectado al streaming");

        // Enviar cabeceras HTTP
        send(client_sock, STREAM_CONTENT_TYPE, strlen(STREAM_CONTENT_TYPE), 0);

        while (1) {
            camera_fb_t *fb = camera_capture_frame();
            if (!fb) {
                ESP_LOGE(TAG, "Fallo al capturar frame");
                break;
            }

            // Enviar boundary
            if (send(client_sock, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY), 0) < 0) {
                camera_return_frame(fb);
                break;
            }

            // Enviar cabecera de la imagen
            char part_buf[128];
            int hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned int)fb->len);
            if (send(client_sock, part_buf, hlen, 0) < 0) {
                camera_return_frame(fb);
                break;
            }

            // Enviar datos JPEG
            if (send(client_sock, (const char *)fb->buf, fb->len, 0) < 0) {
                camera_return_frame(fb);
                break;
            }

            camera_return_frame(fb);

            vTaskDelay(pdMS_TO_TICKS(50)); // ~20 fps
        }

        close(client_sock);
        ESP_LOGI(TAG, "Cliente desconectado");
    }

    close(listen_sock);
    vTaskDelete(NULL);
}

esp_err_t stream_server_start(void) {
    xTaskCreatePinnedToCore(stream_task, "stream_srv", 8192, NULL, 10, NULL, 1);
    return ESP_OK;
}