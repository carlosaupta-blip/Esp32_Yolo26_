/* ---------------------------------------------------------------------------
 * Puente de inferencia YOLO26n para ESP32-S3.
 *
 * Sustituye al pre/post-procesado manual del modelo anterior (224x224, salida
 * unica "output0") por el procesador oficial de ESP-DL, que espera el contrato
 * que produce el tutorial de cuantizacion de YOLO26:
 *
 *   entrada   %images            INT8, 1x320x320x3 (NHWC)
 *   salidas   one2one_p3_box     INT16, 1x40x40x4
 *            one2one_p3_cls     INT16, 1x40x40x1
 *            one2one_p4_box     INT16, 1x20x20x4
 *            one2one_p4_cls     INT16, 1x20x20x1
 *            one2one_p5_box     INT16, 1x10x10x4
 *            one2one_p5_cls     INT16, 1x10x10x1
 *
 * YOLO26 se encarga de:
 *   - preprocess(): letterbox con ImagePreprocessor (pad 114) + cuantizacion
 *     INT8 por LUT de hardware, escribiendo directo en la RAM del modelo.
 *   - postprocess(): decode_grid() con sigmoid, decodificacion de rejilla y
 *     seleccion top-K. YOLO26 es NMS-free, asi que no hay NMS.
 *
 * Lo que aporta este archivo es el MAPEO INVERSO DE LETTERBOX: postprocess()
 * devuelve las cajas en el espacio de la imagen letterboxed (320x320), pero la
 * UI web las dibuja sobre el video de 640x480, asi que hay que deshacer la
 * escala y el relleno. El calculo replica exactamente el de
 * ImagePreprocessor::preprocess() con letterbox activado.
 * ---------------------------------------------------------------------------
 */

#include "yolo_bridge.h"

#include "yolo26.hpp"
#include "dl_image_define.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"

#include <cstdint>
#include <string>
#include <vector>

extern const uint8_t yolo26n_face_320_s8_s3_espdl_start[]
    asm("_binary_yolo26n_face_320_s8_s3_espdl_start");
extern const uint8_t yolo26n_face_320_s8_s3_espdl_end[]
    asm("_binary_yolo26n_face_320_s8_s3_espdl_end");

static const char *TAG = "YOLO_BRIDGE";

// El modelo se entreno con nc = 1 (ver quantization_report.json: "nc": 1).
static const char *kFaceClasses[] = {"face"};

// El procesador se construye con un umbral bajo a proposito: el umbral real
// lo aplica yolo_get_detections() usando yolo_threshold, que la web puede
// cambiar en caliente (web_server.c). Si el umbral estuviera quemado aqui,
// bajar el Threshold desde la UI no tendria efecto.
//
// 0.01 sigue descartando la gran mayoria de celdas por el pre-filtrado entero
// de decode_grid(), y en espacio INT16 implica cls_thresh = -4706, dentro del
// rango [-32768, 32767]. No bajar de 0.001 sin revisar ese limite.
static const float kProcessorThresh = 0.01f;
static const int kTargetK = 32;

static dl::Model *s_model = nullptr;
static YOLO26 *s_processor = nullptr;

static int s_src_w = 640;
static int s_src_h = 480;
static int s_model_in = 0;    // lado de la entrada del modelo (320 en este .espdl)
static float s_lb_scale = 1.0f;
static int s_lb_pad_left = 0;
static int s_lb_pad_top = 0;

static const char *kExpectedOutputs[6] = {
    "one2one_p3_box", "one2one_p3_cls",
    "one2one_p4_box", "one2one_p4_cls",
    "one2one_p5_box", "one2one_p5_cls",
};

static std::string shape_to_string(const std::vector<int> &shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            out += ", ";
        }
    }
    return out + "]";
}

extern "C" esp_err_t yolo_init(void) {
    if (s_model != nullptr) {
        return ESP_OK;
    }

    const size_t model_size = (size_t)(yolo26n_face_320_s8_s3_espdl_end - yolo26n_face_320_s8_s3_espdl_start);
    ESP_LOGI(TAG, "Modelo embebido: %u bytes (%.2f MiB)",
             (unsigned)model_size, model_size / (1024.0 * 1024.0));

    // El fbs_loader de ESP-DL deserializa el flatbuffer con aritmética alineada.
    // target_add_aligned_binary_data() en main/CMakeLists.txt emite .balign 16;
    // esta comprobación detecta un cambio accidental a EMBED_FILES.
    if (((uintptr_t)yolo26n_face_320_s8_s3_espdl_start & 0xF) != 0) {
        ESP_LOGE(TAG, "El .espdl no está alineado a 16 bytes");
        return ESP_FAIL;
    }

    s_model = new dl::Model(
        (const char *)yolo26n_face_320_s8_s3_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,                        // max_internal_size: 0 = todo en PSRAM
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,                  // key
        true                      // param_copy: copia los pesos de flash a PSRAM
    );

    if (s_model == nullptr) {
        ESP_LOGE(TAG, "dl::Model devolvió nullptr");
        return ESP_FAIL;
    }

    // Validar el contrato de entrada. El .info del modelo reporta
    // "%images[INT8, 1x320x320x3]": NHWC, por eso el lado es shape[2] (ancho).
    auto inputs = s_model->get_inputs();
    if (inputs.empty()) {
        ESP_LOGE(TAG, "El modelo no declara entradas");
        return ESP_FAIL;
    }
    dl::TensorBase *input_tensor = inputs.begin()->second;
    std::vector<int> in_shape = input_tensor->get_shape();
    if (in_shape.size() != 4 || in_shape[3] != 3) {
        ESP_LOGE(TAG, "Entrada inesperada %s (se esperaba [1,H,W,3])",
                 shape_to_string(in_shape).c_str());
        return ESP_FAIL;
    }
    if (in_shape[1] != in_shape[2]) {
        ESP_LOGE(TAG, "La entrada %s no es cuadrada; el mapeo de letterbox asume H == W",
                 shape_to_string(in_shape).c_str());
        return ESP_FAIL;
    }
    s_model_in = in_shape[2];

    // YOLO26::postprocess() hace outputs.at(...) sobre estos seis nombres: si
    // falta cualquiera, revienta con std::out_of_range. Se comprueba antes.
    auto outputs = s_model->get_outputs();
    for (int i = 0; i < 6; i++) {
        if (outputs.find(kExpectedOutputs[i]) == outputs.end()) {
            ESP_LOGE(TAG, "Falta la salida '%s' que espera YOLO26::postprocess()",
                     kExpectedOutputs[i]);
            return ESP_FAIL;
        }
    }

    s_processor = new YOLO26(s_model, kTargetK, kProcessorThresh, kFaceClasses);
    if (s_processor == nullptr) {
        ESP_LOGE(TAG, "No se pudo crear el procesador YOLO26");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "YOLO26 listo: entrada %dx%dx3, nc=%d, k=%d, umbral=%.2f",
             s_model_in, s_model_in, 1, kTargetK, kProcessorThresh);
    s_model->profile_memory();
    return ESP_OK;
}

extern "C" void yolo_inspect_io_tensors(void) {
    if (s_model == nullptr) {
        return;
    }

    ESP_LOGI(TAG, "=== TENSORES DE ENTRADA ===");
    for (auto const &entry : s_model->get_inputs()) {
        dl::TensorBase *t = entry.second;
        ESP_LOGI(TAG, "  %s | tipo=%s | shape=%s | exponent=%d",
                 entry.first.c_str(), t->get_dtype_string(),
                 shape_to_string(t->get_shape()).c_str(), (int)t->exponent);
    }

    ESP_LOGI(TAG, "=== TENSORES DE SALIDA ===");
    for (auto const &entry : s_model->get_outputs()) {
        dl::TensorBase *t = entry.second;
        ESP_LOGI(TAG, "  %s | tipo=%s | shape=%s | exponent=%d",
                 entry.first.c_str(), t->get_dtype_string(),
                 shape_to_string(t->get_shape()).c_str(), (int)t->exponent);
    }
}

extern "C" void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h) {
    if (s_processor == nullptr || rgb_src == nullptr || src_w <= 0 || src_h <= 0) {
        return;
    }

    s_src_w = src_w;
    s_src_h = src_h;

    // Replica exacta de ImagePreprocessor::preprocess() con letterbox
    // (managed_components/espressif__esp-dl/vision/image/dl_image_preprocessor.cpp):
    //   scale = min(dst_w/src_w, dst_h/src_h)
    //   si scale_x < scale_y: pad vertical repartido, border_top = pad_h/2
    //   en caso contrario:            pad horizontal,     border_left = pad_w/2
    const float scale_x = (float)s_model_in / (float)src_w;
    const float scale_y = (float)s_model_in / (float)src_h;
    s_lb_scale = (scale_x < scale_y) ? scale_x : scale_y;
    s_lb_pad_left = 0;
    s_lb_pad_top = 0;
    if (scale_x < scale_y) {
        const int pad_h = s_model_in - (int)(s_lb_scale * src_h);
        s_lb_pad_top = pad_h / 2;
    } else {
        const int pad_w = s_model_in - (int)(s_lb_scale * src_w);
        s_lb_pad_left = pad_w / 2;
    }

    dl::image::img_t img;
    img.data = const_cast<uint8_t *>(rgb_src);
    img.width = (uint16_t)src_w;
    img.height = (uint16_t)src_h;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB888;

    s_processor->preprocess(img);
}

extern "C" void yolo_run_inference(void) {
    if (s_model != nullptr) {
        s_model->run();
    }
}

extern "C" int yolo_get_detections(yolo_detection_t *results, int max_results, float thresh) {
    if (s_model == nullptr || s_processor == nullptr || results == nullptr || max_results <= 0) {
        return 0;
    }

    std::vector<dl::detect::result_t> found = s_processor->postprocess(s_model->get_outputs());

    int count = 0;
    for (const auto &r : found) {
        if (count >= max_results) {
            break;
        }
        // El umbral real lo aplica la UI, no el procesador.
        if (r.score < thresh) {
            continue;
        }
        if (r.box.size() < 4) {
            continue;
        }

        // Deshacer el letterbox: espacio del modelo (320x320) -> imagen original.
        float x1 = ((float)r.box[0] - (float)s_lb_pad_left) / s_lb_scale;
        float y1 = ((float)r.box[1] - (float)s_lb_pad_top) / s_lb_scale;
        float x2 = ((float)r.box[2] - (float)s_lb_pad_left) / s_lb_scale;
        float y2 = ((float)r.box[3] - (float)s_lb_pad_top) / s_lb_scale;

        if (x1 < 0.0f) x1 = 0.0f;
        if (y1 < 0.0f) y1 = 0.0f;
        if (x2 > (float)s_src_w) x2 = (float)s_src_w;
        if (y2 > (float)s_src_h) y2 = (float)s_src_h;
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        results[count].x = x1;
        results[count].y = y1;
        results[count].w = x2 - x1;
        results[count].h = y2 - y1;
        results[count].score = r.score;
        results[count].class_id = r.category;
        count++;
    }

    if (count == 0) {
        ESP_LOGD(TAG, "Sin detecciones (candidatas=%u, umbral=%.2f)",
                 (unsigned)found.size(), thresh);
    } else {
        ESP_LOGI(TAG, "Letterbox escala=%.3f pad=(%d,%d) | detecciones=%d de %u candidatas",
                 s_lb_scale, s_lb_pad_left, s_lb_pad_top, count, (unsigned)found.size());
    }

    return count;
}

extern "C" void yolo_get_current_resolution(int *w, int *h) {
    if (w) *w = s_src_w;
    if (h) *h = s_src_h;
}
