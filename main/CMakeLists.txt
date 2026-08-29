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
)

# Generar imagen SPIFFS a partir de www/
spiffs_create_partition_image(spiffs www FLASH_IN_PROJECT)