#include "web_server.h"
#include "mjpeg_stream.h"
#include "camera.h"
#include "led_ctrl.h"
#include "metrics.h"
#include "config_store.h"
#include "cJSON.h"
#include "esp_log.h"
#include "esp_spiffs.h"
#include "esp_netif.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>

static const char *TAG = "httpd";
static httpd_handle_t server = NULL;

/* ---------------------------------------------------------------------
 * Función auxiliar para enviar archivos desde SPIFFS
 * ------------------------------------------------------------------- */
static esp_err_t send_file(httpd_req_t *req, const char *path, const char *mime_type) {
    char filepath[64];
    snprintf(filepath, sizeof(filepath), "/spiffs%s", path);

    struct stat st;
    if (stat(filepath, &st) != 0) {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    FILE *f = fopen(filepath, "r");
    if (!f) return ESP_FAIL;

    char *buf = malloc(st.st_size);
    if (!buf) {
        fclose(f);
        return ESP_ERR_NO_MEM;
    }

    fread(buf, 1, st.st_size, f);
    fclose(f);

    httpd_resp_set_type(req, mime_type);
    esp_err_t res = httpd_resp_send(req, buf, st.st_size);
    free(buf);
    return res;
}

static esp_err_t threshold_post_handler(httpd_req_t *req) {
    char content[64];
    int received = httpd_req_recv(req, content, sizeof(content)-1);
    if (received <= 0) return ESP_FAIL;
    content[received] = 0;

    cJSON *json = cJSON_Parse(content);
    if (!json) return ESP_FAIL;

    cJSON *item = cJSON_GetObjectItem(json, "threshold");
    if (item && cJSON_IsNumber(item)) {
        extern float yolo_threshold;   // declarar acceso
        yolo_threshold = (float)item->valuedouble;
    }

    cJSON_Delete(json);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

static esp_err_t favicon_get_handler(httpd_req_t *req) {
    return send_file(req, "/favicon.ico", "image/x-icon");
}

/* ---------------------------------------------------------------------
 * GET /
 * ------------------------------------------------------------------- */
static esp_err_t index_get_handler(httpd_req_t *req) {
    return send_file(req, "/index.html", "text/html");
}

/* ---------------------------------------------------------------------
 * GET /config
 * ------------------------------------------------------------------- */
static esp_err_t config_get_handler(httpd_req_t *req) {
    return send_file(req, "/config.html", "text/html");
}

/* ---------------------------------------------------------------------
 * GET /style.css
 * ------------------------------------------------------------------- */
static esp_err_t style_get_handler(httpd_req_t *req) {
    return send_file(req, "/style.css", "text/css");
}

/* ---------------------------------------------------------------------
 * GET /script.js
 * ------------------------------------------------------------------- */
static esp_err_t script_get_handler(httpd_req_t *req) {
    return send_file(req, "/script.js", "application/javascript");
}

/* ---------------------------------------------------------------------
 * POST /api/camera
 * ------------------------------------------------------------------- */
static esp_err_t camera_api_post_handler(httpd_req_t *req) {
    char content[256] = {0};
    int received = httpd_req_recv(req, content, sizeof(content)-1);
    if (received <= 0) return ESP_FAIL;
    content[received] = 0;

    cJSON *json = cJSON_Parse(content);
    if (!json) return ESP_FAIL;

    cJSON *item = json->child;
    while (item) {
        if (cJSON_IsNumber(item)) {
            if (strcmp(item->string, "flash_intensity") == 0) {
                led_set_intensity((uint8_t)item->valueint);
            } else {
                camera_set_param(item->string, item->valueint);
            }
        }
        item = item->next;
    }

    cJSON_Delete(json);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

/* ---------------------------------------------------------------------
 * GET /api/capture
 * ------------------------------------------------------------------- */
static esp_err_t capture_get_handler(httpd_req_t *req) {
    camera_fb_t *fb = camera_capture_frame();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "attachment; filename=photo.jpg");
    esp_err_t res = httpd_resp_send(req, (char*)fb->buf, fb->len);
    camera_return_frame(fb);
    return res;
}

/* ---------------------------------------------------------------------
 * GET /api/metrics
 * ------------------------------------------------------------------- */
static esp_err_t metrics_get_handler(httpd_req_t *req) {
    cJSON *root = cJSON_CreateObject();
    metrics_get_json(root);
    // Ya no usamos WebSocket, por lo que no agregamos ws_clients
    char *str = cJSON_PrintUnformatted(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, str);
    cJSON_free(str);
    cJSON_Delete(root);
    return ESP_OK;
}

/* ---------------------------------------------------------------------
 * POST /api/wifi
 * ------------------------------------------------------------------- */
static esp_err_t wifi_api_post_handler(httpd_req_t *req) {
    char content[256];
    int received = httpd_req_recv(req, content, sizeof(content)-1);
    if (received <= 0) return ESP_FAIL;
    content[received] = 0;

    cJSON *json = cJSON_Parse(content);
    if (!json) return ESP_FAIL;

    const char *ssid = cJSON_GetObjectItem(json, "ssid")->valuestring;
    const char *pass = cJSON_GetObjectItem(json, "password")->valuestring;
    bool dhcp = cJSON_IsTrue(cJSON_GetObjectItem(json, "use_dhcp"));

    uint32_t ip = 0, gw = 0, nm = 0;
    if (!dhcp) {
        cJSON *ip_arr = cJSON_GetObjectItem(json, "static_ip");
        if (cJSON_IsArray(ip_arr) && cJSON_GetArraySize(ip_arr) == 4) {
            ip = (uint32_t)cJSON_GetArrayItem(ip_arr,0)->valueint << 24 |
                 (uint32_t)cJSON_GetArrayItem(ip_arr,1)->valueint << 16 |
                 (uint32_t)cJSON_GetArrayItem(ip_arr,2)->valueint << 8  |
                 (uint32_t)cJSON_GetArrayItem(ip_arr,3)->valueint;
        }
        // Similar para gw y nm
        cJSON *gw_arr = cJSON_GetObjectItem(json, "static_gw");
        if (cJSON_IsArray(gw_arr) && cJSON_GetArraySize(gw_arr) == 4) {
            gw = (uint32_t)cJSON_GetArrayItem(gw_arr,0)->valueint << 24 |
                 (uint32_t)cJSON_GetArrayItem(gw_arr,1)->valueint << 16 |
                 (uint32_t)cJSON_GetArrayItem(gw_arr,2)->valueint << 8  |
                 (uint32_t)cJSON_GetArrayItem(gw_arr,3)->valueint;
        }
        cJSON *nm_arr = cJSON_GetObjectItem(json, "static_nm");
        if (cJSON_IsArray(nm_arr) && cJSON_GetArraySize(nm_arr) == 4) {
            nm = (uint32_t)cJSON_GetArrayItem(nm_arr,0)->valueint << 24 |
                 (uint32_t)cJSON_GetArrayItem(nm_arr,1)->valueint << 16 |
                 (uint32_t)cJSON_GetArrayItem(nm_arr,2)->valueint << 8  |
                 (uint32_t)cJSON_GetArrayItem(nm_arr,3)->valueint;
        }
    }

    config_store_set_wifi(ssid, pass, dhcp, ip, gw, nm);
    cJSON_Delete(json);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

/* ---------------------------------------------------------------------
 * Inicio del servidor
 * ------------------------------------------------------------------- */
esp_err_t web_server_start(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 16;
    config.stack_size = 8192;
    config.ctrl_port = 32768;

    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Error al iniciar el servidor HTTP");
        return ESP_FAIL;
    }

    httpd_uri_t uri = {0};

    // Rutas de archivos estáticos
    uri.uri = "/";            uri.method = HTTP_GET;  uri.handler = index_get_handler;      httpd_register_uri_handler(server, &uri);
    uri.uri = "/config";      uri.method = HTTP_GET;  uri.handler = config_get_handler;     httpd_register_uri_handler(server, &uri);
    uri.uri = "/style.css";   uri.method = HTTP_GET;  uri.handler = style_get_handler;      httpd_register_uri_handler(server, &uri);
    uri.uri = "/script.js";   uri.method = HTTP_GET;  uri.handler = script_get_handler;     httpd_register_uri_handler(server, &uri);

    // API
    uri.uri = "/api/camera";  uri.method = HTTP_POST; uri.handler = camera_api_post_handler; httpd_register_uri_handler(server, &uri);
    uri.uri = "/api/capture"; uri.method = HTTP_GET;  uri.handler = capture_get_handler;     httpd_register_uri_handler(server, &uri);

    // Streaming MJPEG (reemplaza al WebSocket)
    uri.uri = "/api/stream";  uri.method = HTTP_GET;  uri.handler = mjpeg_stream_handler;    uri.user_ctx = NULL; httpd_register_uri_handler(server, &uri);
    uri.uri = "/api/threshold"; uri.method = HTTP_POST; uri.handler = threshold_post_handler; httpd_register_uri_handler(server, &uri);
    uri.uri = "/api/metrics"; uri.method = HTTP_GET;  uri.handler = metrics_get_handler;     httpd_register_uri_handler(server, &uri);
    uri.uri = "/api/wifi";    uri.method = HTTP_POST; uri.handler = wifi_api_post_handler;   httpd_register_uri_handler(server, &uri);
    uri.uri = "/favicon.ico"; uri.method = HTTP_GET; uri.handler = favicon_get_handler; httpd_register_uri_handler(server, &uri);
    ESP_LOGW(TAG, "Servidor HTTP iniciado en puerto 80");
    return ESP_OK;
}