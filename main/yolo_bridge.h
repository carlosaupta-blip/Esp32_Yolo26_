#ifndef YOLO_BRIDGE_H
#define YOLO_BRIDGE_H

#include <stdint.h>
#include "esp_err.h"
/* C puro a proposito. Este header lo incluyen main.c y web_server.c, que se
 * compilan con el driver de C. Si aqui se metiera yolo26_pose.hpp, su cadena de
 * cabeceras C++ de ESP-DL llegaria hasta <cstdint> y el build fallaria con un
 * error que no señala este proyecto. Ver yolo26_pose_config.h. */
#include "yolo26_pose_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Una deteccion de pose: caja + esqueleto.
 * kpt_x/kpt_y en pixeles de la imagen ORIGINAL (ya deshice el letterbox) y
 * kpt_v en [0,1]. num_kpts dice cuantos son validos; si el modelo tuviera
 * menos canales, el resto queda en cero. */
typedef struct {
    float x;      /*!< esquina superior izquierda */
    float y;
    float w;
    float h;
    float score;
    int class_id;
    int num_kpts;
    float kpt_x[YOLO26_POSE_NKPT];
    float kpt_y[YOLO26_POSE_NKPT];
    float kpt_v[YOLO26_POSE_NKPT];
} yolo_pose_t;

/* Numero de keypoints del modelo. Lo fija head.kpt_shape del best.pt y lo
 * valida yolo_init() al arrancar: si el .espdl no trae la rama de keypoints
 * con esos canales, aborta con un mensaje explicito en vez de leer basura. */
int yolo_get_num_keypoints(void);
const char *yolo_get_keypoint_name(int index);

/* Numero de clases del modelo. Se deduce de la forma de one2one_pN_cls al
 * arrancar, y yolo_init() aborta si no coincide con la lista de nombres. */
int yolo_get_num_classes(void);
const char *yolo_get_class_name(int index);
const char *yolo_get_class_names_csv(void);

/* Entrada: RGB888 (HWC) de la resolucion de captura, tipicamente 640x480.
 * La funcion aplica el letterbox que exige el modelo y deja el tensor de
 * entrada listo para run(). No allocate memoria. El lado (320) se lee del
 * .espdl con get_inputs(), no esta cableado aqui. */
void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h);

/* Ejecuta el grafo. En ESP32-S3 el tiempo escala con el cuadrado de los pixeles.
 * El benchmark oficial de ESP-DL da 7822 ms para yolo26n_512_s8_s3, y medido en
 * este proyecto: 2978 ms a 320. O sea 3.0 s, NO 3.0 ms. main.c mide el tiempo
 * real de cada etapa con esp_timer en cada inferencia. */
void yolo_run_inference(void);

/* Post-procesado NMS-free (top-K) con mapeo inverso de letterbox: devuelve las
 * cajas y los keypoints ya en el espacio de la imagen original. Rellena hasta
 * max_results detecciones y devuelve cuantas escribio. */
int yolo_get_detections(yolo_pose_t *results, int max_results, float thresh);

esp_err_t yolo_init(void);
void yolo_get_current_resolution(int *w, int *h);
void yolo_inspect_io_tensors(void);

#ifdef __cplusplus
}
#endif

#endif // YOLO_BRIDGE_H
