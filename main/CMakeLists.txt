idf_component_register(
    SRCS
        "main.c"
        "camera.c"
        "config_store.c"
        "wifi_manager.c"
        "web_server.c"  
        "mjpeg_stream.c"     
        "stream_server.c"
        "led_ctrl.c"
        "metrics.c"
        "yolo_bridge.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        esp_wifi
        esp_http_server
        nvs_flash
        esp_netif
        esp_driver_gpio  # due to include of driver/gpio.h in esp_camera.h
        esp_driver_ledc
        driver
        esp32-camera        
        spiffs
        esp_timer
        freertos
        esp_event
        log
        heap
        cjson
        esp-dl
        yolo26
)

# ---------------------------------------------------------------------------
# Modelo YOLO26n cuantizado (INT8 + LUT INT16), 320x320, salida esp32s3.
#
# Bajado de 512x512 a 320x320 porque el benchmark oficial de ESP-DL da 7822 ms
# de model->run() para yolo26n_512_s8_s3: 7.8 SEGUNDOS, no 7 ms. El costo escala
# con el cuadrado de los pixeles, asi que 320/512 = 0.39x => ~3060 ms.
#
# El fbs_loader de ESP-DL exige que el blob .espdl este alineado a 16 bytes
# (ver fbs_loader/cmake/data_file_embed_asm_aligned.cmake, que emite .balign 16).
# EMBED_FILES no garantiza ese alineado, por eso se usa la funcion del
# propio ESP-DL, igual que hace examples/tutorial/.../yolo26_detect.
#
# El simbolo generado es _binary_<nombre_sanitizado>_start, es decir
# _binary_yolo26n_face_320_s8_s3_espdl_start (ver yolo_bridge.cpp).
# El nombre NO puede coincidir con el del modelo del "zoo" de ESP-DL
# (esp-dl/models/yolo26/models/s3/yolo26n_320_s8_s3.espdl) porque ese es un
# modelo COCO de 80 clases distinto a este, entrenado con 1 clase.
#
# La forma de entrada (1x320x320x3) va horneada dentro del .espdl, no se lee de
# ningun sitio en tiempo de ejecucion. yolo_bridge.cpp la lee de alli via
# get_inputs() y por eso no hay ningun 512 cableado en el codigo del firmware.
# ---------------------------------------------------------------------------
idf_build_get_property(__component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST __component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
else()
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/models/yolo26n_face_320_s8_s3.espdl"
    BINARY
)

# Generar imagen SPIFFS a partir de www/
spiffs_create_partition_image(spiffs www FLASH_IN_PROJECT)
