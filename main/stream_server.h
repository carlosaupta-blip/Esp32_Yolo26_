#pragma once
#include "esp_err.h"

esp_err_t stream_server_start(void);
extern int stream_server_get_active_clients(void);