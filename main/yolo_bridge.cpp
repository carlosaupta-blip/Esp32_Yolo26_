#include "yolo_bridge.h"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>

static const char *TAG = "YOLO_BRIDGE";

extern const uint8_t yolo26n_quantized_espdl_start[] asm("_binary_yolo26n_quantized_espdl_start");
extern const uint8_t yolo26n_quantized_espdl_end[]   asm("_binary_yolo26n_quantized_espdl_end");

static dl::Model *yolo_model = nullptr;
static const int imgsz = 224;

static int current_src_w = 640;
static int current_src_h = 480;

static bool debug_logged = false;

struct Detection {
    float x1, y1, x2, y2;   // coordenadas en espacio original (640x480)
    float score;
    int class_id;
};

esp_err_t yolo_init(void) {
    size_t model_size = yolo26n_quantized_espdl_end - yolo26n_quantized_espdl_start;
    ESP_LOGI(TAG, "Cargando modelo embebido (%d bytes)...", model_size);

    yolo_model = new dl::Model(
        (const char *)yolo26n_quantized_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (!yolo_model) {
        ESP_LOGE(TAG, "Fallo crítico al instanciar el modelo.");
        return ESP_FAIL;
    }
    yolo_model->test(); // Ejecutar una inferencia de prueba para inicializar pesos y buffers
    ESP_LOGI(TAG, "¡Modelo YOLO26n inicializado exitosamente!");
    return ESP_OK;
}

void yolo_inspect_io_tensors(void) {
    if (!yolo_model) return;

    auto inputs = yolo_model->get_inputs();
    ESP_LOGI(TAG, "=== TENSORES DE ENTRADA ===");
    for (auto const& [name, tensor] : inputs) {
        std::vector<int> shape = tensor->get_shape();
        float scale = std::pow(2.0f, tensor->exponent);
        ESP_LOGI(TAG, "Nombre: %s | Exponente: %d (Scale: %f)", name.c_str(), tensor->exponent, scale);
    }

    auto outputs = yolo_model->get_outputs();
    ESP_LOGI(TAG, "=== TENSORES DE SALIDA ===");
    for (auto const& [name, tensor] : outputs) {
        std::vector<int> shape = tensor->get_shape();
        float scale = std::pow(2.0f, tensor->exponent);
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            shape_str += std::to_string(shape[i]) + (i < shape.size() - 1 ? ", " : "");
        }
        shape_str += "]";
        ESP_LOGI(TAG, "Nombre: %s | Exponente: %d (Scale: %f) | Shape: %s", 
                 name.c_str(), tensor->exponent, scale, shape_str.c_str());
    }
}

void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h) {
    if (!yolo_model || !rgb_src) return;

    current_src_w = src_w;
    current_src_h = src_h;

    auto inputs = yolo_model->get_inputs();
    if (inputs.empty()) return;

    dl::TensorBase *input_tensor = inputs.begin()->second; 
    if (!input_tensor) return;

    int8_t *model_input = (int8_t *)input_tensor->get_element_ptr();
    if (!model_input) return;

    int channel_stride = imgsz * imgsz;

    if (!debug_logged) {
        ESP_LOGI(TAG, "=== PREPROCESAMIENTO ===");
        ESP_LOGI(TAG, "Imagen origen: %dx%d -> modelo: %dx%d", src_w, src_h, imgsz, imgsz);
        ESP_LOGI(TAG, "Escala de entrada: %f", std::pow(2.0f, input_tensor->exponent));
        debug_logged = true;
    }

    for (int y = 0; y < imgsz; y++) {
        int src_y = (y * src_h) / imgsz;
        for (int x = 0; x < imgsz; x++) {
            int src_x = (x * src_w) / imgsz;
            
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_pixel_idx = y * imgsz + x;

            uint8_t r_raw = rgb_src[src_idx];
            uint8_t g_raw = rgb_src[src_idx + 1];
            uint8_t b_raw = rgb_src[src_idx + 2];

            float norm_r = (float)r_raw / 255.0f;
            float norm_g = (float)g_raw / 255.0f;
            float norm_b = (float)b_raw / 255.0f;

            int q_r = (int)std::round(norm_r * 128.0f);
            int q_g = (int)std::round(norm_g * 128.0f);
            int q_b = (int)std::round(norm_b * 128.0f);

            model_input[dst_pixel_idx]                      = (int8_t)std::clamp(q_r, -128, 127);
            model_input[dst_pixel_idx + channel_stride]     = (int8_t)std::clamp(q_g, -128, 127);
            model_input[dst_pixel_idx + 2 * channel_stride] = (int8_t)std::clamp(q_b, -128, 127);
        }
    }
}

extern "C" void yolo_run_inference(void) {
    if (yolo_model) {
        yolo_model->run();
    }
}



// Calcula la Intersección sobre Unión (IoU) entre dos cajas
static float compute_iou(const Detection& a, const Detection& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

// Función principal de post-procesamiento
extern "C" int yolo_get_detections(yolo_detection_t *results, int max_results, float thresh) {
    if (!yolo_model || !results || max_results <= 0) return 0;

    // 1. Obtener el tensor de salida "output0"
    auto outputs = yolo_model->get_outputs();
    auto it = outputs.find("output0");
    if (it == outputs.end()) {
        ESP_LOGE(TAG, "No se encontró el tensor 'output0'");
        return 0;
    }

    dl::TensorBase *out_tensor = it->second;
    if (!out_tensor) return 0;

    int8_t *out_ptr = (int8_t *)out_tensor->get_element_ptr();
    if (!out_ptr) return 0;

    // 2. Obtener la forma y la escala de cuantización
    std::vector<int> shape = out_tensor->get_shape();
    float out_scale = YOLO_OUTPUT_SCALE;
    ESP_LOGW(TAG, "Usando escala fija: %f (exponent=%d ignorado)", out_scale, out_tensor->exponent);

    // 3. Determinar el formato del tensor (NCHW vs NHWC)
    int num_detections = 0;
    int num_attrs = 0;
    bool is_nchw = false;

    if (shape.size() == 3) {
        // Forma [batch, N, C] o [batch, C, N]
        if (shape[1] == 5 || shape[1] == 6) {
            is_nchw = true;
            num_attrs = shape[1];
            num_detections = shape[2];
        } else if (shape[2] == 5 || shape[2] == 6) {
            is_nchw = false;
            num_attrs = shape[2];
            num_detections = shape[1];
        } else {
            ESP_LOGE(TAG, "Forma de tensor no reconocida: [%d, %d, %d]", shape[0], shape[1], shape[2]);
            return 0;
        }
    } else if (shape.size() == 4) {
        // Posible [1, C, H, W] con C=5 o 6 y H*W = num_detections
        if (shape[1] == 5 || shape[1] == 6) {
            is_nchw = true;
            num_attrs = shape[1];
            num_detections = shape[2] * shape[3];
        } else if (shape[3] == 5 || shape[3] == 6) {
            is_nchw = false;
            num_attrs = shape[3];
            num_detections = shape[1] * shape[2];
        } else {
            ESP_LOGE(TAG, "Forma 4D no soportada: [%d, %d, %d, %d]", shape[0], shape[1], shape[2], shape[3]);
            return 0;
        }
    } else {
        ESP_LOGE(TAG, "Forma de tensor con dimensión %zu no soportada", shape.size());
        return 0;
    }

    if (num_attrs < 5 || num_detections <= 0) {
        ESP_LOGE(TAG, "Número de atributos (%d) o detecciones (%d) inválido", num_attrs, num_detections);
        return 0;
    }

    ESP_LOGD(TAG, "Formato: %s | Detecciones: %d | Atributos: %d | Escala: %f",
             is_nchw ? "NCHW" : "NHWC", num_detections, num_attrs, out_scale);

    // 4. Leer todas las detecciones (antes de NMS)
    std::vector<Detection> detections;
    detections.reserve(num_detections);

    const float ORIG_W = (float)current_src_w;
    const float ORIG_H = (float)current_src_h;

    for (int i = 0; i < num_detections; i++) {
        // Leer los atributos (hasta 6)
        float raw_vals[6] = {0};        
        for (int a = 0; a < num_attrs && a < 6; a++) {
            int idx;
            if (is_nchw) {
                idx = a * num_detections + i;   // NCHW: atributo mayor
            } else {
                idx = i * num_attrs + a;        // NHWC: detección mayor
            }
            raw_vals[a] = (float)out_ptr[idx] * out_scale;
        }
        if (i < 10) {
            ESP_LOGE(TAG, "Ancla %d: cx=%.3f cy=%.3f w=%.3f h=%.3f score=%.3f",i, raw_vals[0], raw_vals[1], raw_vals[2], raw_vals[3], raw_vals[4]);
        }

        // Extraer score (depende de si el modelo tiene clase o no)
        float score;
        int class_id = 0;
        if (num_attrs == 5) {
            // Formato [cx, cy, w, h, score]
            score = raw_vals[4];
        } else if (num_attrs == 6) {
            // Formato [x1, y1, x2, y2, score, class_id] o [cx, cy, w, h, score, class_id]
            score = raw_vals[4];
            class_id = (int)std::round(raw_vals[5]);
        } else {
            continue;
        }

        // Normalizar score con Sigmoid si está fuera de [0,1]
        if (score < 0.0f || score > 1.0f) {
            score = 1.0f / (1.0f + std::exp(-score));
        }

        if (score < thresh) continue;

        // Convertir a coordenadas de caja (x1,y1,x2,y2) en espacio de 224x224
        float x1_224, y1_224, x2_224, y2_224;
        if (num_attrs == 5) {
            float cx = raw_vals[0];
            float cy = raw_vals[1];
            float w = raw_vals[2];
            float h = raw_vals[3];
            x1_224 = cx - w/2.0f;
            y1_224 = cy - h/2.0f;
            x2_224 = cx + w/2.0f;
            y2_224 = cy + h/2.0f;
        } else {
            x1_224 = raw_vals[0];
            y1_224 = raw_vals[1];
            x2_224 = raw_vals[2];
            y2_224 = raw_vals[3];
        }

        // Proyectar a resolución original
        float scale_x = ORIG_W / 224.0f;
        float scale_y = ORIG_H / 224.0f;
        float x1 = std::clamp(x1_224 * scale_x, 0.0f, ORIG_W);
        float y1 = std::clamp(y1_224 * scale_y, 0.0f, ORIG_H);
        float x2 = std::clamp(x2_224 * scale_x, 0.0f, ORIG_W);
        float y2 = std::clamp(y2_224 * scale_y, 0.0f, ORIG_H);

        if (x2 <= x1 || y2 <= y1) continue;

        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.score = score;
        det.class_id = class_id;
        detections.push_back(det);
    }

    // 5. Aplicar NMS (Non-Maximum Suppression)
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> filtered;
    filtered.reserve(detections.size());

    const float iou_thresh = 0.45f; // Ajustable según tu modelo
    for (const auto& det : detections) {
        bool keep = true;
        for (const auto& selected : filtered) {
            if (compute_iou(det, selected) > iou_thresh) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered.push_back(det);
            if (filtered.size() >= (size_t)max_results) break;
        }
    }

    // 6. Copiar al arreglo de salida (formato yolo_detection_t)
    int count = 0;
    for (const auto& det : filtered) {
        results[count].x = det.x1;
        results[count].y = det.y1;
        results[count].w = det.x2 - det.x1;
        results[count].h = det.y2 - det.y1;
        results[count].score = det.score;
        results[count].class_id = det.class_id;
        count++;
        if (count >= max_results) break;
    }

    ESP_LOGD(TAG, "Detecciones encontradas: %d | Filtradas: %d", detections.size(), count);
    return count;
}

extern "C" void yolo_get_current_resolution(int *w, int *h) {
    if (w) *w = current_src_w;
    if (h) *h = current_src_h;
}