// Cola para enviar frames JPEG a la tarea de inferencia YOLO
extern QueueHandle_t yolo_frame_queue;

extern float yolo_threshold;

typedef struct {
    uint8_t *buf;
    size_t len;
    int width;   // Ancho original de captura
    int height;  // Alto original de captura
} yolo_frame_t;

