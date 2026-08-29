// config_store.c
#include "config_store.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nvs.h"
#include <string.h>

#define NVS_WIFI_NS "wifi"
#define NVS_CAM_NS  "camera"

esp_err_t config_store_init(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    return err;
}

esp_err_t config_store_set_wifi(const char *ssid, const char *pass, bool dhcp, uint32_t ip, uint32_t gw, uint32_t nm) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_WIFI_NS, NVS_READWRITE, &handle);
    if (err != ESP_OK) return err;
    nvs_set_str(handle, "ssid", ssid);
    nvs_set_str(handle, "pass", pass);
    nvs_set_u8(handle, "dhcp", dhcp ? 1 : 0);
    nvs_set_u32(handle, "ip", ip);
    nvs_set_u32(handle, "gw", gw);
    nvs_set_u32(handle, "nm", nm);
    nvs_commit(handle);
    nvs_close(handle);
    return ESP_OK;
}

esp_err_t config_store_get_wifi(char *ssid, size_t ssid_len, char *pass, size_t pass_len, bool *dhcp, uint32_t *ip, uint32_t *gw, uint32_t *nm) {
    nvs_handle_t handle;
    if (nvs_open(NVS_WIFI_NS, NVS_READONLY, &handle) != ESP_OK) return ESP_ERR_NVS_NOT_FOUND;
    size_t len = ssid_len; nvs_get_str(handle, "ssid", ssid, &len);
    len = pass_len; nvs_get_str(handle, "pass", pass, &len);
    uint8_t dhcp_val = 1; nvs_get_u8(handle, "dhcp", &dhcp_val); *dhcp = dhcp_val != 0;
    nvs_get_u32(handle, "ip", ip);
    nvs_get_u32(handle, "gw", gw);
    nvs_get_u32(handle, "nm", nm);
    nvs_close(handle);
    return ESP_OK;
}

esp_err_t config_store_set_cam(const char *key, int value) {
    nvs_handle_t handle;
    if (nvs_open(NVS_CAM_NS, NVS_READWRITE, &handle) != ESP_OK) return ESP_FAIL;
    nvs_set_i32(handle, key, value);
    nvs_commit(handle);
    nvs_close(handle);
    return ESP_OK;
}

esp_err_t config_store_get_cam(const char *key, int32_t *value) {
    nvs_handle_t handle;
    if (nvs_open(NVS_CAM_NS, NVS_READONLY, &handle) != ESP_OK) return ESP_ERR_NVS_NOT_FOUND;
    esp_err_t err = nvs_get_i32(handle, key, value);
    nvs_close(handle);
    return err;
}