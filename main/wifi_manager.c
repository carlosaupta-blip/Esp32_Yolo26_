#include "wifi_manager.h"
#include "config_store.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_netif.h"
#include <string.h>

static const char *TAG = "wifi";
static EventGroupHandle_t wifi_events;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static void wifi_event_handler(void *arg, esp_event_base_t base, int32_t id, void *event_data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupSetBits(wifi_events, WIFI_FAIL_BIT);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_events, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_manager_init(void) {
    ESP_LOGW(TAG, "-> wifi_manager_init");
    wifi_events = xEventGroupCreate();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL);
    esp_wifi_set_mode(WIFI_MODE_NULL);
    return ESP_OK;
}

esp_err_t wifi_manager_start(void) {
    ESP_LOGW(TAG, "-> wifi_manager_start");
    char ssid[32] = {0}, pass[64] = {0};
    bool dhcp = true;
    uint32_t ip = 0, gw = 0, nm = 0;
    if (config_store_get_wifi(ssid, sizeof(ssid), pass, sizeof(pass), &dhcp, &ip, &gw, &nm) == ESP_OK && strlen(ssid) > 0) {
        wifi_config_t sta_cfg = {0};
        strlcpy((char*)sta_cfg.sta.ssid, ssid, sizeof(sta_cfg.sta.ssid));
        strlcpy((char*)sta_cfg.sta.password, pass, sizeof(sta_cfg.sta.password));
        if (!dhcp) {
            esp_netif_dhcpc_stop(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"));
            esp_netif_ip_info_t info = { .ip = { .addr = ip }, .gw = { .addr = gw }, .netmask = { .addr = nm } };
            esp_netif_set_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &info);
        }
        esp_wifi_set_mode(WIFI_MODE_STA);
        esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
        esp_wifi_start();
        // Intentar conectar 2 min (4 intentos cada 30s)
        for (int i = 0; i < 4; i++) {
            EventBits_t bits = xEventGroupWaitBits(wifi_events, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdTRUE, pdFALSE, pdMS_TO_TICKS(30000));
            if (bits & WIFI_CONNECTED_BIT) return ESP_OK;
            ESP_LOGW(TAG, "Intento %d fallido", i+1);
        }
    }
    // Fallback AP
    wifi_config_t ap_cfg = {
        .ap = {
            .ssid = "ESP32_CAM",
            .password = "123456789",
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA_WPA2_PSK
        },
    };
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);
    esp_wifi_start();
    ESP_LOGW(TAG, "Modo AP iniciado: ESP32_CAM");
    return ESP_OK;
}