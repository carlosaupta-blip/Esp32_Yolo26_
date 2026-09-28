#include <stdio.h>
#include <stdlib.h>   // qsort: ordena las detecciones por score (ver compare_detections)
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_spiffs.h"
#include "config_store.h"
#include "wifi_manager.h"
#include "camera.h"
#include "led_ctrl.h"
#include "web_server.h"
#include "stream_server.h"
#include "mjpeg_stream.h"
#include "yolo_bridge.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "yolo_frame.h"
#include "esp_http_server.h"
#include "cJSON.h"

static const char *TAG = "main";
QueueHandle_t yolo_frame_queue = NULL;
float yolo_threshold = 0.1f;

/* Numero maximo de detecciones que se conservan y se publican.
 *
 * Antes esto era un unico `yolo_detection_t best_detection` mas un bool
 * `has_detection`: la tarea de inferencia recorria las N detecciones que
 * devolvia YOLO26, se quedaba con la de mayor score y tiraba el resto. El
 * post-proceso de ESP-DL es NMS-free y su top-K (kTargetK = 32 en
 * yolo_bridge.cpp) ya decide que es una deteccion real y cual no, asi que
 * descartar las demas era una limitacion del firmware y no del modelo.
 *
 * 10 es el tope practico: el puente entrega hasta max_results y cada
 * deteccion ocupa ~77 bytes en el JSON (ver detections_get_handler). */
#define MAX_DETECTIONS 10
static yolo_detection_t s_detections[MAX_DETECTIONS];
static int s_detection_count = 0;

static int last_orig_w = 640;
static int last_orig_h = 480;
static SemaphoreHandle_t detections_mutex = NULL;

/* ----------------------------------------------------------------------
 * Manejador HTTP para /api/detections
 * ------------------------------------------------------------------- */
static esp_err_t detections_get_handler(httpd_req_t *req) {
    /* Copia local bajo el mutex: formatear el JSON leyendo los globales sin
     * mutex permitiria que la tarea de YOLO los reescribiera a mitad de la
     * respuesta, y el array contendria detecciones de dos inferencias
     * distintas mezcladas. */
    yolo_detection_t dets[MAX_DETECTIONS];
    int count = 0;
    int orig_w = 640, orig_h = 480;

    if (xSemaphoreTake(detections_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        count = s_detection_count;
        if (count > MAX_DETECTIONS) count = MAX_DETECTIONS;
        memcpy(dets, s_detections, (size_t)count * sizeof(yolo_detection_t));
        orig_w = last_orig_w;
        orig_h = last_orig_h;
        xSemaphoreGive(detections_mutex);
    }

    if (orig_w <= 0) orig_w = 640;
    if (orig_h <= 0) orig_h = 480;

    // El JSON se arma a mano en un buffer de PILA, sin cJSON.
    //
    // Antes se construia un arbol cJSON (~9 mallocs por request) y se
    // serializaba con cJSON_PrintUnformatted(), que reserva un buffer de heap
    // que empieza en 256 B y se duplica en cada ensure(). Eso reventaba de
    // forma intermitente con LoadProhibited dentro de print_object al recursar
    // por el arbol (cJSON.c:1850), desde la tarea de httpd, mientras la tarea
    // de YOLO tiene ~900 KB de RGB en PSRAM, 2.79 MB de modelo residente y el
    // TX de WiFi encima. Tres requests identicos tres milisegundos antes habian
    // funcionado bien, asi que no era un dato malo: era la pila de mallocs la
    // que se quedaba corta en el peor momento.
    //
    // El array se llena deteccion a deteccion comprobando que CADA entrada
    // cabra COMPLETA antes de copiarla. Asi el JSON nunca queda partido por la
    // mitad: si no cabe la siguiente, se cierra el array limpio y se avisa con
    // un log. Recortar a lo bruto produciria JSON invalido y la UI dejaria de
    // dibujar del todo.
    char body[1400];
    size_t used;
    int written = 0;
    int n;

    n = snprintf(body, sizeof(body), "{\"detections\":[");
    if (n < 0 || (size_t)n >= sizeof(body)) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    used = (size_t)n;

    for (int i = 0; i < count; i++) {
        char entry[128];
        int len = snprintf(entry, sizeof(entry),
                           "%s{\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f,"
                           "\"score\":%.4f,\"class_id\":%d}",
                           (i == 0) ? "" : ",",
                           (double)dets[i].x, (double)dets[i].y,
                           (double)dets[i].w, (double)dets[i].h,
                           (double)dets[i].score, dets[i].class_id);
        if (len < 0) break;
        // +64 reserva el cierre '],"orig_width":N,"orig_height":N}'
        if (used + (size_t)len + 64 >= sizeof(body)) break;
        memcpy(body + used, entry, (size_t)len);
        used += (size_t)len;
        written++;
    }

    n = snprintf(body + used, sizeof(body) - used,
                 "],\"orig_width\":%d,\"orig_height\":%d}", orig_w, orig_h);
    if (n < 0 || (size_t)n >= sizeof(body) - used) {
        ESP_LOGE(TAG, "No se pudo cerrar el JSON de detecciones");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    used += (size_t)n;

    if (written < count) {
        ESP_LOGW(TAG, "Solo se publicaron %d de %d detecciones: no caben en %u bytes",
                 written, count, (unsigned)sizeof(body));
    }
    ESP_LOGD(TAG, "Publicando %d/%d detecciones", written, count);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, body, used);
    return ESP_OK;
}

/* Ordena por score descendente.
 *
 * NO se puede confiar en que yolo_get_detections() devuelva el array ordenado:
 * YOLO26::postprocess() (managed_components/espressif__yolo26/yolo26.cpp:153)
 * usa std::nth_element con greater_box, que es una PARTICION parcial: deja los
 * target_k mejores al principio pero sin Guarantee de orden entre ellos. Asi
 * que detections[0] no es necesariamente la de mayor confianza. Ordenar aqui
 * (10 elementos, coste irrelevante) deja el contrato claro para la UI, que
 * dibuja en orden y quiere ver la mas fuerte encima. */
static int compare_detections(const void *a, const void *b) {
    const yolo_detection_t *da = (const yolo_detection_t *)a;
    const yolo_detection_t *db = (const yolo_detection_t *)b;
    if (da->score > db->score) return -1;
    if (da->score < db->score) return 1;
    return 0;
}

/* ----------------------------------------------------------------------
 * Tarea de inferencia YOLO (mejorada)
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

    // Buffer estático en PSRAM para decodificación RGB (VGA)
    const int max_w = 640;
    const int max_h = 480;
    uint8_t *rgb_buffer = heap_caps_malloc(max_w * max_h * 3, MALLOC_CAP_SPIRAM);
    if (!rgb_buffer) {
        ESP_LOGE(TAG, "Error: No se pudo reservar memoria para decodificación RGB en PSRAM");
        vTaskDelete(NULL);
        return;
    }

    const int max_detections = MAX_DETECTIONS;
    yolo_detection_t detections[max_detections];
    yolo_frame_t frame;

    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(100); // 10 fps

    while (1) {
        vTaskDelayUntil(&xLastWakeTime, xFrequency);

        int clients = stream_server_get_active_clients();
        if (clients == 0) {
            continue; // Sin clientes, no procesar
        }

        if (xQueueReceive(yolo_frame_queue, &frame, 0) == pdTRUE) {
            if (frame.buf && frame.len > 0) {
                // Decodificar JPEG a RGB888
                int64_t t_mark = esp_timer_get_time();
                bool ok = fmt2rgb888(frame.buf, frame.len, PIXFORMAT_JPEG, rgb_buffer);
                free(frame.buf); // Liberar el buffer del frame
                const int64_t t_jpeg_us = esp_timer_get_time() - t_mark;

                if (ok) {
                    // Preparar entrada y ejecutar inferencia
                    t_mark = esp_timer_get_time();
                    yolo_prepare_input(rgb_buffer, frame.width, frame.height);
                    const int64_t t_pre_us = esp_timer_get_time() - t_mark;

                    t_mark = esp_timer_get_time();
                    yolo_run_inference();
                    const int64_t t_inf_us = esp_timer_get_time() - t_mark;

                    // Obtener detecciones (usando la nueva función mejorada)
                    t_mark = esp_timer_get_time();
                    int n = yolo_get_detections(detections, max_detections, yolo_threshold);
                    const int64_t t_post_us = esp_timer_get_time() - t_mark;

                    // Desglose por etapa. El benchmark oficial de ESP-DL para
                    // yolo26n_512_s8_s3 da Pre 34 ms | Inf 7822 ms | Post 23 ms,
                    // medidos con esp_timer alrededor de cada llamada y SIN el
                    // decode JPEG, asi que la comparacion es directa. "JPEG" es
                    // el extra que el ejemplo oficial no mide: decode software a
                    // RGB888 a 640x480.
                    // Ojo con la escala: son ~7.8 s de inferencia, no 7.8 ms. Si
                    // esperabas paridad con los 20 fps del stream (33 ms), el
                    // cuello de botella es model->run() y ningun ajuste de DMA ni
                    // de RTOS lo acerca; la unica palanca real es la resolucion
                    // de entrada, que escala con el cuadrado de los pixeles.
                    const int64_t t_total_us = t_jpeg_us + t_pre_us + t_inf_us + t_post_us;
                    ESP_LOGI(TAG,
                             "Tiempos | JPEG: %" PRId64 " ms | Pre: %" PRId64 " ms | "
                             "Inf: %" PRId64 " ms | Post: %" PRId64 " ms | Total: %" PRId64 " ms "
                             "-> %.3f FPS",
                             t_jpeg_us / 1000, t_pre_us / 1000, t_inf_us / 1000,
                             t_post_us / 1000, t_total_us / 1000,
                             (t_total_us > 0) ? (1000000.0 / (double)t_total_us) : 0.0);

                    int src_w, src_h;
                    yolo_get_current_resolution(&src_w, &src_h);

                    if (xSemaphoreTake(detections_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                        last_orig_w = src_w;
                        last_orig_h = src_h;

                        // Se publican TODAS las detecciones que superaron
                        // yolo_threshold, no solo la mejor. yolo_get_detections()
                        // ya devuelve unicamente las que YOLO26 deemio reales
                        // (decode_grid con su pre-filtrado entero de confianza y
                        // su top-K NMS-free), asi que el unico filtro que queda
                        // es el umbral que controla el usuario. Antes se
                        // descartaban todas menos la de mayor score.
                        int keep = (n > max_detections) ? max_detections : n;
                        if (keep > 1) {
                            qsort(detections, (size_t)keep, sizeof(yolo_detection_t),
                                  compare_detections);
                        }
                        memcpy(s_detections, detections, (size_t)keep * sizeof(yolo_detection_t));
                        s_detection_count = keep;

                        if (keep > 0) {
                            ESP_LOGI(TAG, "✅ Inferencia OK [%dx%d] -> %d deteccion(es) "
                                          "sobre umbral %.2f (mejor: %.2f%%)",
                                     src_w, src_h, keep, yolo_threshold,
                                     detections[0].score * 100.0f);
                            for (int i = 0; i < keep; i++) {
                                ESP_LOGI(TAG, "   [%d] clase=%d score=%.3f "
                                              "[x:%.1f, y:%.1f, w:%.1f, h:%.1f]",
                                              i, detections[i].class_id,
                                              detections[i].score,
                                              detections[i].x, detections[i].y,
                                              detections[i].w, detections[i].h);
                            }
                        } else {
                            ESP_LOGD(TAG, "❌ Sin detecciones sobre umbral %.2f", yolo_threshold);
                        }
                        xSemaphoreGive(detections_mutex);
                    }
                } else {
                    ESP_LOGE(TAG, "Fallo al descomprimir JPEG a RGB888");
                }
            } else {
                if (frame.buf) free(frame.buf);
            }
        }

        // Limpiar frames acumulados en la cola (evitar lag)
        yolo_frame_t tmp;
        while (xQueueReceive(yolo_frame_queue, &tmp, 0) == pdTRUE) {
            free(tmp.buf);
        }
    }
}

/* ----------------------------------------------------------------------
 * Tarea heartbeat (mantener vivo el sistema)
 * ------------------------------------------------------------------- */
static void heartbeat_task(void *arg) {
    while (1) {
        ESP_LOGI(TAG, "💓 heartbeat: alive");
        vTaskDelay(pdMS_TO_TICKS(30000));
    }
}

/* ----------------------------------------------------------------------
 * app_main - punto de entrada
 * ------------------------------------------------------------------- */
void app_main(void) {
    ESP_LOGW(TAG, "-> app_main");

    // Motivo del reinicio anterior, impreso lo primero de todo. Es la unica
    // forma de saber si un brownout ocurrio ANTES de app_main (durante el init
    // de PSRAM, en el que no hay forma de actuar) o DESPUES (en cuyo caso la
    // ultima linea del log antes del corte dice exactamente en que etapa).
    switch (esp_reset_reason()) {
        case ESP_RST_BROWNOUT:
            ESP_LOGE(TAG, "Reinicio anterior: BROWNOUT (caida de alimentacion). "
                          "En ESP32-S3 el detector ya esta en el nivel 7 = 2.44 V, "
                          "el mas permisivo de los 8; no hay ajuste de firmware. "
                          "Revisar fuente, cable USB y condensador de bulk.");
            break;
        case ESP_RST_POWERON:    ESP_LOGW(TAG, "Reinicio anterior: encendido"); break;
        case ESP_RST_TASK_WDT: ESP_LOGE(TAG, "Reinicio anterior: WDT de tarea"); break;
        case ESP_RST_INT_WDT:  ESP_LOGE(TAG, "Reinicio anterior: WDT de interrupcion"); break;
        case ESP_RST_WDT:      ESP_LOGE(TAG, "Reinicio anterior: otro WDT"); break;
        case ESP_RST_PANIC:    ESP_LOGE(TAG, "Reinicio anterior: panic"); break;
        case ESP_RST_SW:       ESP_LOGW(TAG, "Reinicio anterior: software"); break;
        default:               ESP_LOGW(TAG, "Reinicio anterior: otro (0x%x)", esp_reset_reason()); break;
    }

    ESP_ERROR_CHECK(config_store_init());

    // Montar SPIFFS para almacenamiento de configuración
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = "spiffs",
        .max_files = 5,
        .format_if_mount_failed = true
    };
    ESP_ERROR_CHECK(esp_vfs_spiffs_register(&conf));

    // Inicializar WiFi y cámara
    ESP_ERROR_CHECK(wifi_manager_init());
    ESP_ERROR_CHECK(wifi_manager_start());
    ESP_ERROR_CHECK(camera_init());
    ESP_ERROR_CHECK(led_init());

    // Iniciar servidor web (puerto 80)
    ESP_ERROR_CHECK(web_server_start());

    // Registrar endpoint /api/detections
    httpd_handle_t server = web_server_get_handle();
    if (server) {
        httpd_uri_t detections_uri = {
            .uri       = "/api/detections",
            .method    = HTTP_GET,
            .handler   = detections_get_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &detections_uri);
        ESP_LOGI(TAG, "Registrado /api/detections");
    }

    // Iniciar servidor de streaming (puerto 81)
    ESP_ERROR_CHECK(stream_server_start());

    // Crear cola para frames YOLO (tamaño 2)
    yolo_frame_queue = xQueueCreate(2, sizeof(yolo_frame_t));
    if (!yolo_frame_queue) {
        ESP_LOGE(TAG, "No se pudo crear la cola YOLO");
        return;
    }

    // Crear mutex para detecciones
    detections_mutex = xSemaphoreCreateMutex();
    if (!detections_mutex) {
        ESP_LOGE(TAG, "No se pudo crear mutex");
        return;
    }

    // Tarea heartbeat (baja prioridad)
    xTaskCreate(heartbeat_task, "heartbeat", 2048, NULL, 1, NULL);

    // Tarea de inferencia YOLO: STACK AUMENTADO A 12288 bytes para soportar std::vector y std::sort
    BaseType_t ret = xTaskCreatePinnedToCore(
        yolo_inference_task,
        "yolo_task",
        12288,          // <--- STACK MEJORADO
        NULL,
        5,              // Prioridad alta
        NULL,
        1               // Core 1 (dejar core 0 para WiFi y eventos)
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Error al crear yolo_task");
    } else {
        ESP_LOGI(TAG, "yolo_task creada correctamente con stack de 12KB");
    }

    ESP_LOGI(TAG, "Sistema listo");
}