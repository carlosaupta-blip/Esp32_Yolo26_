#include "metrics.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <sys/time.h>

void metrics_get_json(cJSON *root) {
    int64_t uptime = esp_timer_get_time() / 1000000;
    wifi_mode_t mode;
    esp_wifi_get_mode(&mode);
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    char ip[16] = "0.0.0.0";
    if (netif) {
        esp_netif_ip_info_t ip_info;
        if (esp_netif_get_ip_info(netif, &ip_info) == ESP_OK) {
            snprintf(ip, sizeof(ip), IPSTR, IP2STR(&ip_info.ip));
        }
    }
    cJSON_AddNumberToObject(root, "free_heap_kb", heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);
    cJSON_AddNumberToObject(root, "free_psram_kb", heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024);
    cJSON_AddNumberToObject(root, "uptime_s", (double)uptime);
    cJSON_AddStringToObject(root, "wifi_mode", mode == WIFI_MODE_STA ? "STA" : "AP");
    cJSON_AddStringToObject(root, "ip_addr", ip);
    cJSON_AddNumberToObject(root, "ws_clients", 0); // completado en web_server
}