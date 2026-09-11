#ifndef YOLO_BRIDGE_H
#define YOLO_BRIDGE_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLO_OUTPUT_SCALE  (1.0f)   // o 1.0f/255.0f / 128.0f

typedef struct {
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} yolo_detection_t;

esp_err_t yolo_init(void);
void yolo_prepare_input(const uint8_t *rgb_src, int src_w, int src_h);
void yolo_run_inference(void);
int yolo_get_detections(yolo_detection_t *results, int max_results, float thresh);
void yolo_run_diagnostic_dump();
void yolo_get_current_resolution(int *w, int *h);

// Nueva función de inspección
void yolo_inspect_io_tensors(void);

#ifdef __cplusplus
}
#endif

#endif // YOLO_BRIDGE_H