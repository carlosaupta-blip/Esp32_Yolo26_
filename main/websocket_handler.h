#ifndef WEBSOCKET_HANDLER_H
#define WEBSOCKET_HANDLER_H

#include "esp_err.h"
#include "yolo_bridge.h"

esp_err_t websocket_init(void);
void websocket_broadcast_detections(yolo_detection_t *dets, int count);

#endif