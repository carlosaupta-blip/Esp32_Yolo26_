// config_store.h
#pragma once
#include "esp_err.h"

esp_err_t config_store_init(void);
esp_err_t config_store_set_wifi(const char *ssid, const char *pass, bool dhcp, uint32_t ip, uint32_t gw, uint32_t nm);
esp_err_t config_store_get_wifi(char *ssid, size_t ssid_len, char *pass, size_t pass_len, bool *dhcp, uint32_t *ip, uint32_t *gw, uint32_t *nm);
esp_err_t config_store_set_cam(const char *key, int value);
esp_err_t config_store_get_cam(const char *key, int32_t *value);