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
#include "yolo_bridge.h"          // <-- Añadido
#include "esp_camera.h"           // Para camera_fb_t y fmt2rgb888
#include "img_converters.h"       // Para fmt2rgb888
#include "yolo_frame.h"



static const char *TAG = "main";
QueueHandle_t yolo_frame_queue = NULL;
float yolo_threshold = 0.25f;   // umbral inicial
/* ---------------------------------------------------------------------
 * Tarea de inferencia YOLO (Core 1)
 * Captura frames independientes a baja frecuencia y ejecuta el modelo
 * ------------------------------------------------------------------- */
static void yolo_inference_task(void *arg) {
    ESP_LOGI(TAG, "Inicializando YOLO...");
    if (yolo_init() != ESP_OK) {
        ESP_LOGE(TAG, "Error en yolo_init");
        vTaskDelete(NULL);
        return;
    }
    ESP_LOGI(TAG, "yolo_init completado");
    yolo_inspect_io_tensors();

    // Buffer RGB en PSRAM
    uint8_t *rgb_buffer = heap_caps_malloc(320 * 240 * 3, MALLOC_CAP_SPIRAM);
    if (!rgb_buffer) {
        ESP_LOGE(TAG, "No se pudo reservar memoria para RGB buffer");
        vTaskDelete(NULL);
        return;
    }

    yolo_detection_t detections[10];
    yolo_frame_t frame;

    

    while (1) {
        // Solo procesar si hay clientes de streaming activos
        int clients = stream_server_get_active_clients();
        ESP_LOGI(TAG, "Clientes activos: %d", clients);
        if (clients == 0) {
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }
        
        // Esperar a recibir un frame JPEG desde la cola
        if (xQueueReceive(yolo_frame_queue, &frame, pdMS_TO_TICKS(1000)) == pdTRUE) {
            if (frame.buf && frame.len > 0) {
                // Convertir JPEG a RGB888
                ESP_LOGI(TAG, "Frame recibido para YOLO: %d bytes", frame.len);
                bool ok = fmt2rgb888(frame.buf, frame.len, PIXFORMAT_RGB888, rgb_buffer);
                free(frame.buf);  // liberar la copia JPEG

                if (ok) {
                    yolo_prepare_input(rgb_buffer, 320, 240);                    
                    yolo_run_inference();
                    //yolo_run_diagnostic_dump();
                    int n = yolo_get_detections(detections, 10, yolo_threshold);
                    if (n > 0) {
                        ESP_LOGI(TAG, "=== Detecciones YOLO ===");
                        for (int i = 0; i < n; i++) {
                            ESP_LOGI(TAG, "Clase %d | Score %.2f | [x:%.1f, y:%.1f, w:%.1f, h:%.1f]",
                                     detections[i].class_id, detections[i].score,
                                     detections[i].x, detections[i].y,
                                     detections[i].w, detections[i].h);
                        }
                    }
                } else {
                    ESP_LOGE(TAG, "Error convirtiendo JPEG a RGB");
                }
            } else {
                // Si el frame es inválido, liberar si es necesario
                ESP_LOGI(TAG, "Timeout esperando frame YOLO");
                if (frame.buf) free(frame.buf);
            }
        }
        yolo_frame_t tmp;
        while (xQueueReceive(yolo_frame_queue, &tmp, 0) == pdTRUE) {
            free(tmp.buf);
        }

        // Esperar 1 segundo antes de la próxima inferencia
        vTaskDelay(pdMS_TO_TICKS(1000));
    }

}

/* ---------------------------------------------------------------------
 * Heartbeat
 * ------------------------------------------------------------------- */
static void heartbeat_task(void *arg) {
    while (1) {
        ESP_LOGI(TAG, "💓 heartbeat: alive");
        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

/* ---------------------------------------------------------------------
 * app_main
 * ------------------------------------------------------------------- */
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

    // Servidor de streaming MJPEG
    ESP_ERROR_CHECK(stream_server_start());

    yolo_frame_queue = xQueueCreate(2, sizeof(yolo_frame_t));
    if (!yolo_frame_queue) {
        ESP_LOGE(TAG, "No se pudo crear la cola YOLO");
        return;
    }

    // Tarea heartbeat
    xTaskCreate(heartbeat_task, "heartbeat", 2048, NULL, 1, NULL);

    // Tarea de inferencia YOLO en Core 1
    BaseType_t ret = xTaskCreatePinnedToCore(yolo_inference_task, "yolo_task", 8192, NULL, 5, NULL, 1);
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Error al crear yolo_task");
    } else {
        ESP_LOGI(TAG, "yolo_task creada correctamente");
    }
    ESP_LOGI(TAG, "Sistema listo");
}