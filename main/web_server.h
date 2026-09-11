#pragma once
#include "esp_http_server.h"

esp_err_t web_server_start(void);
httpd_handle_t web_server_get_handle(void);