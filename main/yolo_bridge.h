#ifndef YOLO_BRIDGE_H
#define YOLO_BRIDGE_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} yolo_detection_t;

/* Entrada: RGB888 (HWC) de la resolucion de captura, tipicamente 640x480.
 * La funcion aplica el letterbox que exige el modelo 320x320 y deja el tensor
 * de entrada del modelo listo para run(). No allocate memoria. El lado 320 se
 * lee del .espdl con get_inputs(), no esta cableado aqui. */
void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h);

/* Ejecuta el grafo. En ESP32-S3 el tiempo escala con el cuadrado de los pixeles:
 * el benchmark oficial de ESP-DL da 7822 ms para yolo26n_512_s8_s3 y se estima
 * ~3060 ms para este yolo26n_320_s8_s3, es decir 3.0 s, NO 3 ms. main.c mide el
 * tiempo real de cada etapa con esp_timer en cada inferencia. */
void yolo_run_inference(void);

/* Post-procesado NMS-free (top-K) con mapeo inverso de letterbox: devuelve las
 * cajas ya en el espacio de la imagen original (src_w x src_h). */
int yolo_get_detections(yolo_detection_t *results, int max_results, float thresh);
esp_err_t yolo_init(void);
void yolo_get_current_resolution(int *w, int *h);
void yolo_inspect_io_tensors(void);

#ifdef __cplusplus
}
#endif

#endif // YOLO_BRIDGE_H
