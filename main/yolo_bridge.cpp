#include "yolo_bridge.h"
#include "dl_model_base.hpp"
#include "esp_log.h"
#include <cmath>
#include <string>
#include <algorithm>   // para std::clamp
#include <vector>

static const char *TAG = "YOLO_BRIDGE";

// Símbolos del modelo embebido
extern const uint8_t yolo26n_quantized_espdl_start[] asm("_binary_yolo26n_quantized_espdl_start");
extern const uint8_t yolo26n_quantized_espdl_end[]   asm("_binary_yolo26n_quantized_espdl_end");

static dl::Model *yolo_model = nullptr;
static const int imgsz = 224;

// Parámetros de cuantización de entrada (YOLO26, potencia de 2)
// Escala = 2^(-7) = 1/128, zero_point = 0
static const float input_scale = 1.0f / 128.0f;   // 0.0078125
static const int input_zeropoint = 0;

/* ---------------------------------------------------------------------
 * Inicialización del modelo
 * ------------------------------------------------------------------- */
esp_err_t yolo_init(void) {
    size_t model_size = yolo26n_quantized_espdl_end - yolo26n_quantized_espdl_start;
    ESP_LOGI(TAG, "Cargando modelo embebido (%d bytes)...", model_size);

    // En ESP-DL v3, solo se pasa el puntero y la ubicación del modelo
    yolo_model = new dl::Model(
        (const char *)yolo26n_quantized_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (!yolo_model) {
        ESP_LOGE(TAG, "Fallo crítico al instanciar el modelo.");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "¡Modelo YOLO26n inicializado exitosamente!");
    return ESP_OK;
}

/* ---------------------------------------------------------------------
 * Inspección de tensores de entrada/salida (para debug)
 * Llama a esta función justo después de yolo_init() para verificar
 * los nombres, formas y exponentes reales.
 * ------------------------------------------------------------------- */
void yolo_inspect_io_tensors(void) {
    if (!yolo_model) {
        ESP_LOGE(TAG, "El modelo no ha sido inicializado.");
        return;
    }

    // Entradas
    auto inputs = yolo_model->get_inputs();
    ESP_LOGI(TAG, "=== TENSORES DE ENTRADA ===");
    for (auto const& [name, tensor] : inputs) {
        std::vector<int> shape = tensor->get_shape();
        ESP_LOGI(TAG, "Nombre: %s", name.c_str());
        ESP_LOGI(TAG, "  Exponente: %d (Scale: %f)", tensor->exponent, std::pow(2.0f, tensor->exponent));
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            shape_str += std::to_string(shape[i]);
            if (i < shape.size() - 1) shape_str += ", ";
        }
        shape_str += "]";
        ESP_LOGI(TAG, "  Dimensiones: %s", shape_str.c_str());
    }

    // Salidas
    auto outputs = yolo_model->get_outputs();
    ESP_LOGI(TAG, "=== TENSORES DE SALIDA ===");
    for (auto const& [name, tensor] : outputs) {
        std::vector<int> shape = tensor->get_shape();
        ESP_LOGI(TAG, "Nombre: %s", name.c_str());
        ESP_LOGI(TAG, "  Exponente: %d (Scale: %f)", tensor->exponent, std::pow(2.0f, tensor->exponent));
        std::string shape_str = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            shape_str += std::to_string(shape[i]);
            if (i < shape.size() - 1) shape_str += ", ";
        }
        shape_str += "]";
        ESP_LOGI(TAG, "  Dimensiones: %s", shape_str.c_str());
    }
}

/* ---------------------------------------------------------------------
 * Preprocesamiento de imagen RGB888 -> NCHW INT8
 * ------------------------------------------------------------------- */
void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h) {
    auto inputs = yolo_model->get_inputs();
    dl::TensorBase *input_tensor = inputs.begin()->second; 
    int8_t *model_input = (int8_t *)input_tensor->get_element_ptr(); // Buffer NCHW de ESP-DL [cite: 11]

    int channel_stride = imgsz * imgsz; // 224 * 224 = 50176 bytes de desplazamiento por canal

    for (int y = 0; y < imgsz; y++) {
        int src_y = (y * src_h) / imgsz;
        for (int x = 0; x < imgsz; x++) {
            int src_x = (x * src_w) / imgsz;
            
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_pixel_idx = y * imgsz + x;

            // CORRECCIÓN CLAVE: Castear explícitamente y leer como uint8_t (0 a 255)
            uint8_t r_raw = rgb_src[src_idx];
            uint8_t g_raw = rgb_src[src_idx + 1];
            uint8_t b_raw = rgb_src[src_idx + 2];

            // 1. Normalización estándar de YOLO a [0.0, 1.0] [cite: 11]
            float norm_r = (float)r_raw / 255.0f;
            float norm_g = (float)g_raw / 255.0f;
            float norm_b = (float)b_raw / 255.0f;

            // 2. Cuantización simétrica INT8 usando escala 0.0078125 (1/128) [cite: 838]
            int q_r = (int)std::round(norm_r * 128.0f);
            int q_g = (int)std::round(norm_g * 128.0f);
            int q_b = (int)std::round(norm_b * 128.0f);

            // 3. Escribir en disposición NCHW con clamping estricto para evitar desbordamientos [cite: 11]
            model_input[dst_pixel_idx]                      = (int8_t)std::clamp(q_r, -128, 127); // Canal R
            model_input[dst_pixel_idx + channel_stride]     = (int8_t)std::clamp(q_g, -128, 127); // Canal G
            model_input[dst_pixel_idx + 2 * channel_stride] = (int8_t)std::clamp(q_b, -128, 127); // Canal B
        }
    }
}
/* ---------------------------------------------------------------------
 * Ejecutar inferencia
 * ------------------------------------------------------------------- */
extern "C" void yolo_run_inference(void) {
    if (yolo_model) {
        yolo_model->run();
    }
}

/* ---------------------------------------------------------------------
 * Decodificación de salidas NMS-Free (YOLO26)
 * ------------------------------------------------------------------- */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

extern "C" int yolo_get_detections(yolo_detection_t *results, int max_results, float thresh) {
    if (!yolo_model || !results || max_results <= 0) return 0;

    auto outputs = yolo_model->get_outputs();
    if (outputs.empty()) return 0;

    int count = 0;

    struct ScaleInfo {
        const char* box_name;
        const char* score_name;
        int grid_size;
    };

    ScaleInfo scales[] = {
        {"output0", "606", 28},
        {"613",     "626", 14},
        {"633",     "646", 7}
    };

    // --- ESCALA FIJA PARA SCORES (1/128, igual que la entrada) ---
    const float FIXED_SCORE_SCALE = 0.0078125f;

    for (auto &scale : scales) {
        auto box_it = outputs.find(scale.box_name);
        auto score_it = outputs.find(scale.score_name);
        if (box_it == outputs.end() || score_it == outputs.end()) continue;

        dl::TensorBase *box_tensor = box_it->second;
        dl::TensorBase *score_tensor = score_it->second;

        int8_t *box_ptr = (int8_t *)box_tensor->get_element_ptr();
        int8_t *score_ptr = (int8_t *)score_tensor->get_element_ptr();

        // Usamos la escala reportada para cajas (0.03125, 0.0625, 0.0625)
        float box_scale = std::pow(2.0f, box_tensor->exponent);

        for (int cell_idx = 0; cell_idx < scale.grid_size * scale.grid_size; cell_idx++) {
            // --- SCORE con escala fija ---
            float logit = (float)score_ptr[cell_idx] * FIXED_SCORE_SCALE;
            float score = sigmoid(logit);

            if (score >= thresh) {
                // --- CAJA: decodificación como coordenadas normalizadas ---
                int box_idx = cell_idx * 4;
                float left   = (float)box_ptr[box_idx]     * box_scale;
                float top    = (float)box_ptr[box_idx + 1] * box_scale;
                float right  = (float)box_ptr[box_idx + 2] * box_scale;
                float bottom = (float)box_ptr[box_idx + 3] * box_scale;

                // Proyectar a píxeles (224x224)
                float x1 = left   * 224.0f;
                float y1 = top    * 224.0f;
                float x2 = right  * 224.0f;
                float y2 = bottom * 224.0f;

                // Clamp
                x1 = std::clamp(x1, 0.0f, 224.0f);
                y1 = std::clamp(y1, 0.0f, 224.0f);
                x2 = std::clamp(x2, 0.0f, 224.0f);
                y2 = std::clamp(y2, 0.0f, 224.0f);

                results[count].x = x1;
                results[count].y = y1;
                results[count].w = x2 - x1;
                results[count].h = y2 - y1;
                results[count].score = score;
                results[count].class_id = 0;

                count++;
                if (count >= max_results) return count;
            }
        }
    }
    return count;
}

static inline float sigmoid_func(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void yolo_run_diagnostic_dump() {
    if (!yolo_model) {
        ESP_LOGE("YOLO_DIAG", "Error: El modelo no está inicializado.");
        return;
    }

    ESP_LOGW("YOLO_DIAG", "===============================================");
    ESP_LOGW("YOLO_DIAG", "=== DIAGNÓSTICO EN TIEMPO DE EJECUCIÓN ===");
    ESP_LOGW("YOLO_DIAG", "===============================================");

    // 1. Verificar tensor de entrada
    auto inputs = yolo_model->get_inputs();
    if (!inputs.empty()) {
        dl::TensorBase *input_tensor = inputs.begin()->second;
        int8_t *in_data = (int8_t *)input_tensor->get_element_ptr();
        float in_scale = std::pow(2.0f, input_tensor->exponent);

        ESP_LOGI("YOLO_DIAG", "[ENTRADA] Tensor: '%s' | Exponente: %d | Escala: %f",
                 inputs.begin()->first.c_str(), input_tensor->exponent, in_scale);

        int channel_stride = 224 * 224;
        ESP_LOGI("YOLO_DIAG", "Muestra de píxeles iniciales (NCHW):");
        for (int i = 0; i < 5; i++) {
            int8_t r_raw = in_data[i];
            int8_t g_raw = in_data[i + channel_stride];
            int8_t b_raw = in_data[i + 2 * channel_stride];

            ESP_LOGI("YOLO_DIAG", "  Pixel [%d] -> R: %d (%.3f) | G: %d (%.3f) | B: %d (%.3f)",
                     i, r_raw, (float)r_raw * in_scale,
                     g_raw, (float)g_raw * in_scale,
                     b_raw, (float)b_raw * in_scale);
        }
    }

    // 2. Verificar scores de salida (606, 626, 646)
    auto outputs = yolo_model->get_outputs();
    const char* score_names[] = {"606", "626", "646"};
    int grid_sizes[] = {28, 14, 7};

    for (int t = 0; t < 3; t++) {
        const char* name = score_names[t];
        auto iter = outputs.find(name);
        if (iter == outputs.end()) {
            ESP_LOGE("YOLO_DIAG", "No se encontró tensor '%s'", name);
            continue;
        }

        dl::TensorBase *score_tensor = iter->second;
        int8_t *scores = (int8_t *)score_tensor->get_element_ptr();
        float out_scale = std::pow(2.0f, score_tensor->exponent);
        int grid_sz = grid_sizes[t];
        int total = grid_sz * grid_sz;

        ESP_LOGI("YOLO_DIAG", "[SCORES] Tensor: '%s' (Grid %dx%d) | Escala: %f", name, grid_sz, grid_sz, out_scale);

        // Encontrar valor máximo crudo
        int8_t max_raw = -128;
        int max_idx = 0;
        for (int i = 0; i < total; i++) {
            if (scores[i] > max_raw) {
                max_raw = scores[i];
                max_idx = i;
            }
        }
        float max_logit = (float)max_raw * out_scale;
        float max_prob = sigmoid_func(max_logit);
        ESP_LOGW("YOLO_DIAG", "  -> Máximo: crudo=%d, logit=%.4f, prob=%.4f (idx=%d)", max_raw, max_logit, max_prob, max_idx);

        // Primeros 5 scores crudos
        ESP_LOGI("YOLO_DIAG", "  -> Primeros 5 scores:");
        for (int i = 0; i < 5; i++) {
            float logit = (float)scores[i] * out_scale;
            float prob = sigmoid_func(logit);
            ESP_LOGI("YOLO_DIAG", "    [%d] crudo=%d, logit=%.4f, prob=%.4f", i, scores[i], logit, prob);
        }
    }

    // 3. Verificar cajas de salida (output0, 613, 633)
    const char* box_names[] = {"output0", "613", "633"};
    for (int t = 0; t < 3; t++) {
        const char* name = box_names[t];
        auto iter = outputs.find(name);
        if (iter == outputs.end()) {
            ESP_LOGE("YOLO_DIAG", "No se encontró tensor '%s'", name);
            continue;
        }

        dl::TensorBase *box_tensor = iter->second;
        int8_t *boxes = (int8_t *)box_tensor->get_element_ptr();
        float out_scale = std::pow(2.0f, box_tensor->exponent);
        int grid_sz = grid_sizes[t];

        ESP_LOGI("YOLO_DIAG", "[CAJAS] Tensor: '%s' (Grid %dx%d) | Escala: %f", name, grid_sz, grid_sz, out_scale);

        // Primeras 3 cajas LTRB
        ESP_LOGI("YOLO_DIAG", "  -> Primeras 3 cajas (crudo -> dequantizado):");
        for (int i = 0; i < 3; i++) {
            int idx = i * 4;
            int8_t l = boxes[idx];
            int8_t t = boxes[idx+1];
            int8_t r = boxes[idx+2];
            int8_t b = boxes[idx+3];
            ESP_LOGI("YOLO_DIAG", "    [%d] crudo=[%d,%d,%d,%d] -> [%.3f,%.3f,%.3f,%.3f]",
                     i, l, t, r, b,
                     (float)l*out_scale, (float)t*out_scale, (float)r*out_scale, (float)b*out_scale);
        }
    }
    ESP_LOGW("YOLO_DIAG", "===============================================");
}