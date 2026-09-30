/* ---------------------------------------------------------------------------
 * Puente de inferencia YOLO26n-POSE para ESP32-S3.
 * ---------------------------------------------------------------------------
 * Sustituye al pre/post-procesado manual del modelo anterior (224x224, salida
 * unica "output0") por el preprocesador de ESP-DL, que espera el contrato que
 * produce el cuaderno de cuantizacion de YOLO26 con USE_POSE=True:
 *
 *   entrada   %images            INT8, 1x320x320x3 (NHWC)
 *   salidas   one2one_p3_box     INT16, 1x40x40x4
 *            one2one_p3_cls     INT16, 1x40x40x3     <- nc=3 en este modelo
 *            one2one_p3_kpt     INT8,  1x40x40x18    <- NUEVO (6 kpt x 3 dims)
 *            one2one_p4_box     INT16, 1x20x20x4
 *            one2one_p4_cls     INT16, 1x20x20x3
 *            one2one_p4_kpt     INT8,  1x20x20x18
 *            one2one_p5_box     INT16, 1x10x10x4
 *            one2one_p5_cls     INT16, 1x10x10x3
 *            one2one_p5_kpt     INT8,  1x10x10x18
 *
 * Los anchos no son uniformes a proposito: la rama box y la cls se promovieron a
 * INT16 por USE_MIXED_INT16 (son las sensibles, el error de un canal de box se ve
 * como una caja corrida), mientras que la rama kpt se queda en INT8. Es lo que
 * salio del .espdl: NO se puede subir sin reexportar.
 *
 * YOLO26Pose se encarga de:
 *   - preprocess(): letterbox con ImagePreprocessor (pad 114) + cuantizacion
 *     INT8 por LUT de hardware, escribiendo directo en la RAM del modelo.
 *   - postprocess(): decode_grid() con ltrb (reg_max=1), decodificacion de
 *     keypoints (kpt + anchor) * stride, sigmoid de visibilidad, top-K NMS-free
 *     y mapeo inverso de letterbox.
 *
 * YOLO26 es NMS-free: la cabeza one2one ya es end-to-end y su top-K por score
 * es la seleccion definitiva, asi que no hay NMS.
 * ---------------------------------------------------------------------------
 */

#include "yolo_bridge.h"

#include "yolo26_pose.hpp"
#include "dl_image_define.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern const uint8_t yolo26n_pose_320_s8_s3_espdl_start[]
    asm("_binary_yolo26n_pose_320_s8_s3_espdl_start");
extern const uint8_t yolo26n_pose_320_s8_s3_espdl_end[]
    asm("_binary_yolo26n_pose_320_s8_s3_espdl_end");

static const char *TAG = "YOLO_BRIDGE";

/* Las 3 clases del dataset, en el ORDEN de best.pt.
 *
 * Este orden viene del quantization_report.json del propio paquete, que lo
 * copia de best.pt:  "class_names": ["Detecting", "check", "ok"].
 * Un desajuste aqui NO da error: da la etiqueta equivocada y en silencio. Por
 * eso yolo_init() comprueba ademas que la forma de one2one_p3_cls coincida con
 * este arreglo, y aborta con el numero real si no. */
static const char *kPoseClasses[] = {"Detecting", "check", "ok"};
#define POSE_NUM_CLASSES ((int)(sizeof(kPoseClasses) / sizeof(kPoseClasses[0])))

/* El procesador se construye con un umbral bajo a proposito: el umbral real lo
 * aplica yolo_get_detections() con yolo_threshold, que la web puede cambiar en
 * caliente. Si el umbral estuviera quemado aqui, bajar el Threshold desde la UI
 * no tendria efecto.
 *
 * 0.01 sigue descartando la gran mayoria de celdas por el pre-filtrado entero
 * de decode_grid(), y en espacio INT16 implica cls_thresh dentro del rango
 * [-32768, 32767]. No bajar de 0.001 sin revisar ese limite. */
static const float kProcessorThresh = 0.01f;
static const int kTargetK = 8;
static const float kKptVisThresh = 0.5f; /*!< visibilidad minima para dibujar un keypoint */

static dl::Model *s_model = nullptr;
static YOLO26Pose *s_processor = nullptr;
static int s_num_classes = 0;   /*!< deducido de la forma de one2one_pN_cls */

static int s_src_w = 640;
static int s_src_h = 480;
static int s_model_in = 0;    /*!< lado de la entrada del modelo (320 en este .espdl) */
static float s_lb_scale = 1.0f;
static int s_lb_pad_left = 0;
static int s_lb_pad_top = 0;

/* Los nueve nombres que consume YOLO26Pose::postprocess(). Se comprueban en
 * yolo_init() porque outputs.at() lanzaria std::out_of_range si faltara alguno. */
static const char *kExpectedOutputs[9] = {
    "one2one_p3_box", "one2one_p3_cls", "one2one_p3_kpt",
    "one2one_p4_box", "one2one_p4_cls", "one2one_p4_kpt",
    "one2one_p5_box", "one2one_p5_cls", "one2one_p5_kpt",
};

static std::string shape_to_string(const std::vector<int> &shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); i++) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            out += ", ";
        }
    }
    return out + "]";
}

int yolo_get_num_keypoints(void)
{
    return YOLO26_POSE_NKPT;
}

/* 6 keypoints de mano: muneca y punta de cada dedo. El ORDEN tiene que
 * coincidir con el .txt del dataset; si no, el esqueleto se dibuja cruzado.
 * Coincide con KPT_NAMES de la celda 2 del cuaderno y con KPT_NAMES en
 * main/www/script.js. */
const char *yolo_get_keypoint_name(int index)
{
    static const char *kNames[YOLO26_POSE_NKPT] = {
        "muneca", "punta_pulgar", "punta_indice", "punta_medio",
        "punta_anular", "punta_menique",
    };
    if (index < 0 || index >= YOLO26_POSE_NKPT) {
        return "?";
    }
    return kNames[index];
}

/* Numero de clases que declara el tensor de salida. Se deduce de la forma, no
 * de kPoseClasses, para que un cambio de dataset no tenga que tocar dos sitios
 * y se pueda validar que las listas de nombres siguen en sincronia. */
int yolo_get_num_classes(void)
{
    return s_num_classes;
}

const char *yolo_get_class_name(int index)
{
    if (index < 0 || index >= POSE_NUM_CLASSES) {
        return "?";
    }
    return kPoseClasses[index];
}

/* "Detecting, check, ok" para el log de arranque. */
const char *yolo_get_class_names_csv(void)
{
    static char buf[192];
    buf[0] = '\0';
    for (int i = 0; i < POSE_NUM_CLASSES; i++) {
        if (i > 0) {
            strncat(buf, ", ", sizeof(buf) - strlen(buf) - 1);
        }
        strncat(buf, kPoseClasses[i], sizeof(buf) - strlen(buf) - 1);
    }
    return buf;
}

extern "C" esp_err_t yolo_init(void)
{
    if (s_model != nullptr) {
        return ESP_OK;
    }

    const size_t model_size = (size_t)(yolo26n_pose_320_s8_s3_espdl_end - yolo26n_pose_320_s8_s3_espdl_start);
    ESP_LOGI(TAG, "Modelo embebido: %u bytes (%.2f MiB)",
             (unsigned)model_size, model_size / (1024.0 * 1024.0));

    // El fbs_loader de ESP-DL deserializa el flatbuffer con aritmetica alineada.
    // target_add_aligned_binary_data() en main/CMakeLists.txt emite .balign 16;
    // esta comprobacion detecta un cambio accidental a EMBED_FILES.
    if (((uintptr_t)yolo26n_pose_320_s8_s3_espdl_start & 0xF) != 0) {
        ESP_LOGE(TAG, "El .espdl no está alineado a 16 bytes");
        return ESP_FAIL;
    }

    s_model = new dl::Model(
        (const char *)yolo26n_pose_320_s8_s3_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,                        /*!< max_internal_size: 0 = todo en PSRAM */
        dl::MEMORY_MANAGER_GREEDY, /*!< mm_type */
        nullptr,                  /*!< key */
        true                      /*!< param_copy: copia los pesos de flash a PSRAM */
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

    // Comprobar las nueve salidas ANTES de construir el procesador: despues
    // outputs.at() reventaria con std::out_of_range, que es un crash sin
    // mensaje util.
    auto outputs = s_model->get_outputs();
    for (int i = 0; i < 9; i++) {
        if (outputs.find(kExpectedOutputs[i]) == outputs.end()) {
            ESP_LOGE(TAG, "Falta la salida '%s' que exige el post-proceso de pose",
                     kExpectedOutputs[i]);
            ESP_LOGE(TAG, "El .espdl tiene %zu salidas. Si son 6 y no 9, es un modelo "
                          "de DETECCION, no de pose: hay que exportarlo con USE_POSE=True.",
                     outputs.size());
            return ESP_FAIL;
        }
    }

    // nc sale de la forma de la rama de clase. Es la unica comprobacion que
    // ata este firmware al dataset concreto: si best.pt se reentrena con otras
    // clases, salta aqui en vez de rotular mal en la UI.
    dl::TensorBase *cls_p3 = outputs.at("one2one_p3_cls");
    s_num_classes = cls_p3->get_shape()[3];
    if (s_num_classes != POSE_NUM_CLASSES) {
        ESP_LOGE(TAG, "El .espdl declara nc=%d (one2one_p3_cls) pero kPoseClasses tiene %d",
                 s_num_classes, POSE_NUM_CLASSES);
        ESP_LOGE(TAG, "Las clases del best.pt actual son: %s", yolo_get_class_names_csv());
        ESP_LOGE(TAG, "Actualiza kPoseClasses en yolo_bridge.cpp para que coincidan.");
        return ESP_FAIL;
    }

    // Y verificar que la rama de keypoints trae los canales justos. Este .espdl
    // es de 6 keypoints x 3 dims = 18 canales; uno de 21 (MediaPipe) aborta
    // aqui en vez de que el post-proceso lea de mas.
    dl::TensorBase *kpt = outputs.at("one2one_p3_kpt");
    std::vector<int> kpt_shape = kpt->get_shape();
    if (kpt_shape.size() != 4 || kpt_shape[3] != YOLO26_POSE_NKPT * YOLO26_POSE_KPT_DIMS) {
        ESP_LOGE(TAG, "one2one_p3_kpt es %s; se esperaba [1,H,W,%d] (%d keypoints x %d dims)",
                 shape_to_string(kpt_shape).c_str(),
                 YOLO26_POSE_NKPT * YOLO26_POSE_KPT_DIMS, YOLO26_POSE_NKPT, YOLO26_POSE_KPT_DIMS);
        return ESP_FAIL;
    }

    s_processor = new YOLO26Pose(s_model, kTargetK, kProcessorThresh, kKptVisThresh, kPoseClasses);
    if (s_processor == nullptr) {
        ESP_LOGE(TAG, "No se pudo crear el procesador YOLO26Pose");
        return ESP_FAIL;
    }

    // El log de arranque es la primera comprobacion de que el firmware y la UI
    // hablan del mismo dataset. nc viene de la forma del tensor, no de una
    // constante: con 1 quemado aqui el log mintiria.
    ESP_LOGI(TAG, "YOLO26-Pose listo: entrada %dx%dx3, nc=%d (%s), k=%d, umbral=%.2f, %d keypoints",
             s_model_in, s_model_in, s_num_classes, yolo_get_class_names_csv(), kTargetK,
             kProcessorThresh, YOLO26_POSE_NKPT);
    s_model->profile_memory();
    return ESP_OK;
}

extern "C" void yolo_inspect_io_tensors(void)
{
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

    ESP_LOGI(TAG, "=== TENSORES DE SALIDA (9: 3 niveles x box+cls+kpt) ===");
    for (auto const &entry : s_model->get_outputs()) {
        dl::TensorBase *t = entry.second;
        ESP_LOGI(TAG, "  %s | tipo=%s | shape=%s | exponent=%d",
                 entry.first.c_str(), t->get_dtype_string(),
                 shape_to_string(t->get_shape()).c_str(), (int)t->exponent);
    }
}

extern "C" void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h)
{
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

extern "C" void yolo_run_inference(void)
{
    if (s_model != nullptr) {
        s_model->run();
    }
}

extern "C" int yolo_get_detections(yolo_pose_t *results, int max_results, float thresh)
{
    if (s_model == nullptr || s_processor == nullptr || results == nullptr || max_results <= 0) {
        return 0;
    }

    std::vector<YOLO26PoseResult> found = s_processor->postprocess(s_model->get_outputs());

    int count = 0;
    for (const auto &r : found) {
        if (count >= max_results) {
            break;
        }
        // El umbral real lo aplica la UI, no el procesador.
        if (r.score < thresh) {
            continue;
        }
        if (r.box[2] <= r.box[0] || r.box[3] <= r.box[1]) {
            continue;
        }

        /* LA CAJA YA VIENE MAPEADA. No la transforms otra vez.
         *
         * YOLO26Pose::postprocess() termina con map_to_original() sobre TODAS las
         * detecciones, caja incluida, usando el estado del ImagePreprocessor
         * (get_resize_scale_x / get_border_left), que es la fuente fiable.
         *
         * Este bloque la mapeaba por segunda vez con una replica a mano de la
         * misma formula. Con escala 0.5 y pad_top 40 la caja se iba de 132x125
         * a 576x480, es decir a tapar el cuadro entero, y el log lo confirmaba
         * con la aritmetica exacta:
         *
         *   modelo   [16, 59, 309, 265]
         *   1a vez   [32, 38, 618, 450]     <- map_to_original, correcto
         *   2a vez   x:64.0 y:0.0 w:576.0 h:480.0   <- lo que se publicaba
         *
         * Los keypoints NO sufrían eso porque aquí solo se copiaban. Por eso el
         * esqueleto salía en su sitio y la caja no: el síntoma era "la caja
         *engloba todo" y no "las manos estan mal indicadas".
         *
         * Aquí solo queda el recorte a la imagen, que sí hace falta. */
        float x1 = r.box[0];
        float y1 = r.box[1];
        float x2 = r.box[2];
        float y2 = r.box[3];

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
        results[count].num_kpts = r.num_kpts;
        for (int k = 0; k < r.num_kpts && k < YOLO26_POSE_NKPT; k++) {
            float kx = r.kpt_x[k];
            float ky = r.kpt_y[k];
            if (kx < 0.0f) kx = 0.0f;
            if (ky < 0.0f) ky = 0.0f;
            if (kx > (float)s_src_w) kx = (float)s_src_w;
            if (ky > (float)s_src_h) ky = (float)s_src_h;
            results[count].kpt_x[k] = kx;
            results[count].kpt_y[k] = ky;
            results[count].kpt_v[k] = r.kpt_v[k];
        }
        count++;
    }

    // INFO y no DEBUG a proposito. Con DEBUG este mensaje no llega nunca (el
    // nivel por defecto es INFO), y entonces "no aparece nada en la pagina" no
    // distingue entre: el modelo no detecto, el umbral de la UI esta alto, o el
    // post-proceso devuelve cero. Este numero es el que separa los tres casos,
    // asi que tiene que verse siempre, termasuko cuando no hay detecciones.
    if (count == 0) {
        ESP_LOGI(TAG, "Sin detecciones: %u candidatas del post-proceso, umbral=%.2f",
                 (unsigned)found.size(), thresh);
    } else {
        ESP_LOGI(TAG, "Letterbox escala=%.3f pad=(%d,%d) | detecciones=%d de %u candidatas",
                 s_lb_scale, s_lb_pad_left, s_lb_pad_top, count, (unsigned)found.size());
    }

    return count;
}

extern "C" void yolo_get_current_resolution(int *w, int *h)
{
    if (w) *w = s_src_w;
    if (h) *h = s_src_h;
}
