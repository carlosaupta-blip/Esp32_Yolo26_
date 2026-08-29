#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "config_store.h"
#include "wifi_manager.h"
#include "camera.h"
#include "led_ctrl.h"
#include "web_server.h"
#include "stream_server.h"

#include "mjpeg_stream.h"

static const char *TAG = "main";

static void heartbeat_task(void *arg) {
    while (1) {
        ESP_LOGI(TAG, "💓 heartbeat: alive");
        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

void app_main(void) {
    ESP_LOGW(TAG, "-> app_main");
    ESP_ERROR_CHECK(config_store_init());

    // Montar SPIFFS
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "spiffs",
        .max_files = 5,
        .format_if_mount_failed = true
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&conf));

    // WiFi
    ESP_ERROR_CHECK(wifi_manager_init());
    ESP_ERROR_CHECK(wifi_manager_start());

    // Cámara
    ESP_ERROR_CHECK(camera_init());

    // LED
    ESP_ERROR_CHECK(led_init());

    // Servidor web
    ESP_ERROR_CHECK(web_server_start());

    ESP_ERROR_CHECK(stream_server_start());   // <-- añadir

    // Tarea heartbeat
    xTaskCreate(heartbeat_task, "heartbeat", 2048, NULL, 1, NULL);
    ESP_LOGI(TAG, "Sistema listo");
}