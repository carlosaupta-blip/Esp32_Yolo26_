#pragma once
#include "dl_image_preprocessor.hpp"
#include "dl_image_process.hpp"
#include "dl_tensor_base.hpp"
#include <map>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------
 * Post-procesador de POSE para YOLO26 (Pose26), sobre ESP-DL.
 * ---------------------------------------------------------------------------
 *
 * POR QUE NO SE USA EL COMPONENTE espressif/yolo26:
 *
 * Ese componente solo hace deteccion. Su decode_grid() lee 4 canales de box y
 * num_classes de cls, y nunca toca el tensor de keypoints. El campo
 * `keypoint` de dl::detect::result_t existe en ESP-DL, pero YOLO26 no lo llena.
 *
 * ESP-DL trae `dl_pose_yolo11_postprocessor`, pero es de otra arquitectura y
 * NO es compatible:
 *
 *   |                     | yolo11posePostProcessor | YOLO26 (este)      |
 *   |---------------------|------------------------|--------------------|
 *   | reg_max             | 16, con DFL            | 1, ltrb directo   |
 *   | decodificacion box  | anchor + dfl_integral  | cx = w + 0.5      |
 *   | keypoints           | 17 fijos (COCO)        | nk del modelo     |
 *   | NMS                  | si (nms())             | NO, es NMS-free   |
 *   | nombres de salida   | box0/score0/kpt0       | one2one_pN_*      |
 *   | dtypes por rama     | una sola combinacion   | MIXTOS, ver abajo |
 *
 * LAS TRES RAMAS NO COMPARTEN TIPO. Con USE_MIXED_INT16, el cuaderno promueve a
 * INT16 las salidas que ESP-DL considera sensibles y deja el resto en INT8:
 *
 *   one2one_pN_box   INT16      protege la geometria de la caja
 *   one2one_pN_cls   INT16      protege el score, que es el umbral de todo
 *   one2one_pN_kpt   INT8       la cabeza de keypoints no se promovio
 *
 * Por eso decode_grid() esta plantilla sobre TRES tipos, uno por rama, y
 * postprocess() instancia las ocho combinaciones. Con un unico tipo, este
 * modelo no caia en ninguna y el post-proceso devolvia cero detecciones:
 *
 *   [YOLO26Pose] Error: tipos de tensor no soportados en el nivel 0.
 *
 * Consecuencia para el diagnostico: si la cabeza de keypoints saliera
 * deformada con la caja bien, el primer sospechoso es su exponent, no el
 * firmware. Es la rama con menos proteccion cuantitativa.
 *
 * Ademas, yolo11pose decodifica con `kpt*2.0 + (anchor-0.5)`, mientras que
 * Pose26 usa `(kpt + anchor) * stride`. Copiar el codigo de YOLO11 daria
 * cada keypoint desplazado medio stride (4 px a 320), es decir dedos torcidos.
 *
 * DECODIFICACION (verificada contra ultralytics/nn/modules/head.py, Pose26):
 *
 *   anchor   = (w + 0.5, h + 0.5)          centro de la celda de la rejilla
 *   box      x1 = (anchor_x - d_l) * stride     [NHWC, reg_max = 1]
 *   kpt      kx = (raw_kx + anchor_x) * stride   [NHWC]
 *   kpt vis  v  = sigmoid(raw_v)                 en [0, 1]
 *
 * El mapeo inverso de letterbox lo hace el ImagePreprocessor, igual que en
 * dl_pose_yolo11_postprocessor: get_resize_scale_x(true) y get_border_*().
 */

/* Las constantes del contrato (NKPT, KPT_DIMS) viven en yolo26_pose_config.h,
 * que es C puro. main.c lo incluye a traves de yolo_bridge.h y solo necesita
 * esas dos macros para dimensionar yolo_pose_t: si este .hpp se alcanzara desde
 * un .c, arrastraria la cadena de cabeceras C++ de ESP-DL hasta <cstdint>, que
 * el driver de C no encuentra. Ver el comentario de yolo26_pose_config.h. */
#include "yolo26_pose_config.h"

struct YOLO26PoseResult {
    int category;                              /*!< indice de clase */
    float score;                               /*!< confianza de la deteccion */
    int box[4];                                /*!< [x1,y1,x2,y2] en la imagen ORIGINAL */
    int num_kpts;                              /*!< keypoints validos (= YOLO26_POSE_NKPT) */
    float kpt_x[YOLO26_POSE_NKPT];             /*!< x en la imagen ORIGINAL */
    float kpt_y[YOLO26_POSE_NKPT];             /*!< y en la imagen ORIGINAL */
    float kpt_v[YOLO26_POSE_NKPT];             /*!< visibilidad en [0,1] */
};

class YOLO26Pose {
  private:
    std::vector<int> grid_sizes; /*!< lado de la rejilla por nivel, calculado en preprocess */
    int num_classes;             /*!< deducido de la forma de cls */
    int target_k;
    float conf_thresh;
    float kpt_vis_thresh; /*!< por debajo de esto el keypoint se marca invisible */

    dl::image::ImagePreprocessor *m_image_preprocessor;
    dl::TensorBase *m_input_tensor; /*!< tensor de entrada, para diagnosticar que la imagen llego */

    const int strides[3] = {8, 16, 32};
    int num_levels = 3;

    /* Template con UN TIPO POR RAMA: TB para box, TC para cls, TK para kpt.
     *
     * No es un refinamiento. USE_MIXED_INT16 promueve a INT16 las salidas que
     * considera sensibles (box y cls) y deja el resto en INT8, de modo que este
     * .espdl mezcla:
     *
     *     one2one_pN_box  INT16    one2one_pN_cls  INT16    one2one_pN_kpt  INT8
     *
     * Con un unico tipo T, como estava antes, la condicion de despacho exigia
     * que las tres coincidieran y este modelo caia siempre en el error, sin
     * deteccion alguna. postprocess() instancia las ocho combinaciones.
     *
     * El umbral se precalcula en el tipo de la rama de clase, que es la que se
     * compara contra conf_thresh, y cada rama se desescala con SU exponent. Es
     * la misma optimizacion que hace el decode_grid del componente oficial:
     * las celdas que no pueden pasar se descartan sin tocar coma flotante. */
    template <typename TB, typename TC, typename TK>
    void decode_grid(dl::TensorBase *p_box,
                     dl::TensorBase *p_cls,
                     dl::TensorBase *p_kpt,
                     int stride,
                     int grid,
                     std::vector<YOLO26PoseResult> &out);

    void map_to_original(YOLO26PoseResult &r) const;

  public:
    const char **class_names;

    YOLO26Pose(dl::Model *model,
               int k,
               float thresh,
               float kpt_vis_th,
               const char **classes);
    ~YOLO26Pose();

    /* Letterbox + cuantizacion SIMD LUT, escribiendo directo en la RAM del
     * modelo. Mismo preprocesador que usa el componente oficial. */
    void preprocess(const dl::image::img_t &img);

    /* Top-K NMS-free sobre las nueve salidas del grafo. Los keypoints se
     * devuelven YA mapeados a la imagen original. */
    std::vector<YOLO26PoseResult> postprocess(const std::map<std::string, dl::TensorBase *> &outputs);
};
