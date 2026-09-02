// Cola para enviar frames JPEG a la tarea de inferencia YOLO
extern QueueHandle_t yolo_frame_queue;

typedef struct {
    uint8_t *buf;
    size_t len;
} yolo_frame_t;