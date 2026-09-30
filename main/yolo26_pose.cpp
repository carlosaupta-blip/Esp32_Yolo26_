/* ---------------------------------------------------------------------------
 * Post-procesador de POSE para YOLO26 (Pose26). Ver yolo26_pose.hpp para la
 * justificacion de por que no se reutiliza el componente espressif/yolo26 ni
 * el dl_pose_yolo11_postprocessor de ESP-DL.
 * ---------------------------------------------------------------------------
 * El layout de todas las salidas es NHWC: [1, H, W, C]. El indice de celda es
 * pixel_idx = (h * grid_w) + w, y el offset de cada rama dentro de la celda es
 * pixel_idx * <canales_de_la_rama>.
 *
 * LAS TRES RAMAS NO COMPARTEN TIPO. USE_MIXED_INT16 promueve a INT16 las salidas
 * sensibles (box y cls) y deja el resto en INT8, de modo que este .espdl trae
 *
 *     one2one_pN_box  INT16      one2one_pN_cls  INT16      one2one_pN_kpt  INT8
 *
 * Por eso decode_grid() esta plantilla sobre TRES tipos, uno por rama, y
 * postprocess() despacha las ocho combinaciones. La version anterior usaba un
 * unico tipo T y exigia que box, cls y kpt fueran del mismo, con lo cual este
 * modelo caia siempre en la rama de error:
 *
 *     [YOLO26Pose] Error: tipos de tensor no soportados en el nivel 0.
 *
 * que no dice nada util. Ahora el mensaje nombra los tres tipos reales.
 * ---------------------------------------------------------------------------*/

#include "yolo26_pose.hpp"
#include "dl_math.hpp"
#include "dl_tool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

/* Codigo compacto del tipo de un tensor: 0 = INT8, 1 = INT16, -1 = no soportado.
 * Se plantilla para no tener que nombrar el tipo de TensorBase::dtype, que es un
 * detalle de la version de ESP-DL. */
template <typename DT>
static inline int dtype_code(DT dt)
{
    if (dt == dl::DATA_TYPE_INT8) {
        return 0;
    }
    if (dt == dl::DATA_TYPE_INT16) {
        return 1;
    }
    return -1;
}

static const char *dtype_name(int code)
{
    if (code == 0) {
        return "INT8";
    }
    if (code == 1) {
        return "INT16";
    }
    return "sin soporte";
}

YOLO26Pose::YOLO26Pose(dl::Model *model,
                       int k,
                       float thresh,
                       float kpt_vis_th,
                       const char **classes) :
    target_k(k), conf_thresh(thresh), kpt_vis_thresh(kpt_vis_th), class_names(classes)
{
    // Mean 0, std 255: el mismo preprocesado que el componente oficial, para
    // que el peso cuantizado sea interpretable igual.
    m_image_preprocessor = new dl::image::ImagePreprocessor(model, {0, 0, 0}, {255, 255, 255});
    m_image_preprocessor->enable_letterbox({114, 114, 114});

    // El lado de la rejilla se deriva de la entrada del MODELO, no de un
    // argumento: asi el firmware no tiene ningun 320 ni 512 cableado.
    grid_sizes.resize(num_levels);
    m_input_tensor = nullptr;
    auto inputs = model->get_inputs();
    if (!inputs.empty()) {
        m_input_tensor = inputs.begin()->second;
        int input_w = m_input_tensor->shape[2]; // NHWC: [1,H,W,C]
        for (int i = 0; i < num_levels; i++) {
            grid_sizes[i] = input_w / strides[i];
        }
    }
}

YOLO26Pose::~YOLO26Pose()
{
    delete m_image_preprocessor;
}

void YOLO26Pose::preprocess(const dl::image::img_t &img)
{
    m_image_preprocessor->preprocess(img);

    // COMPROBACION DE LA ENTRADA, UNA SOLA VEZ.
    //
    // Es el eslabon que faltaba en el diagnostico. Si el tensor de entrada llega
    // todo a cero, el modelo no esta fallando: nadie le ha dado nada, y el
    // fallo esta aguas arriba (frame vacio, JPEG corrupto, o el preprocesador
    // escribiendo en otro sitio). El rango del tensor de entrada dice eso de un
    // vistazo y sin interpretar ningun score de salida.
    static bool s_input_checked = false;
    if (!s_input_checked && m_input_tensor != nullptr) {
        s_input_checked = true;
        const int dc = dtype_code(m_input_tensor->dtype);
        printf("[YOLO26Pose] entrada: %s exp=%d shape=[1,%d,%d,%d]",
               dtype_name(dc), (int)m_input_tensor->exponent,
               m_input_tensor->shape[1], m_input_tensor->shape[2], m_input_tensor->shape[3]);
        if (dc == 0) {
            const int8_t *px = (const int8_t *)m_input_tensor->data;
            const int n = m_input_tensor->shape[1] * m_input_tensor->shape[2] * m_input_tensor->shape[3];
            int32_t mn = 127, mx = -128, nonzero = 0;
            int64_t suma = 0;
            for (int k = 0; k < n; k++) {
                const int32_t v = px[k];
                if (v != 0) {
                    nonzero++;
                }
                suma += v;
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
            }
            const float sc = DL_SCALE(m_input_tensor->exponent);
            printf(" pixeles=[%d..%d] media=%.1f no_cero=%ld/%d  -> rgb=[%.3f..%.3f] media=%.3f\n",
                   (int)mn, (int)mx, (double)suma / (double)(n > 0 ? n : 1), nonzero, n,
                   mn * sc, mx * sc, ((double)suma / (double)(n > 0 ? n : 1)) * sc);
        } else {
            printf(" (barrido de pixeles solo para INT8)\n");
        }
    }
}

void YOLO26Pose::map_to_original(YOLO26PoseResult &r) const
{
    // Mapeo inverso de letterbox. El ImagePreprocessor ya lleva la cuenta de
    // la escala y de los bordes, igual que en dl_pose_yolo11_postprocessor.
    const float inv_x = m_image_preprocessor->get_resize_scale_x(true);
    const float inv_y = m_image_preprocessor->get_resize_scale_y(true);
    const int border_left = m_image_preprocessor->get_border_left();
    const int border_top = m_image_preprocessor->get_border_top();

    r.box[0] = (int)(((float)r.box[0] - (float)border_left) * inv_x);
    r.box[1] = (int)(((float)r.box[1] - (float)border_top) * inv_y);
    r.box[2] = (int)(((float)r.box[2] - (float)border_left) * inv_x);
    r.box[3] = (int)(((float)r.box[3] - (float)border_top) * inv_y);

    for (int k = 0; k < r.num_kpts; k++) {
        r.kpt_x[k] = (r.kpt_x[k] - (float)border_left) * inv_x;
        r.kpt_y[k] = (r.kpt_y[k] - (float)border_top) * inv_y;
    }
}

/* Un tipo por rama: TB para box, TC para cls, TK para kpt. El pre-filtrado
 * entero se hace en el tipo de la rama de clase, que es la que se compara contra
 * el umbral, y cada rama se desescala con SU exponent. */
template <typename TB, typename TC, typename TK>
void YOLO26Pose::decode_grid(dl::TensorBase *p_box,
                             dl::TensorBase *p_cls,
                             dl::TensorBase *p_kpt,
                             int stride,
                             int grid,
                             std::vector<YOLO26PoseResult> &out)
{
    const float box_scale = DL_SCALE(p_box->exponent);
    const float cls_scale = DL_SCALE(p_cls->exponent);
    const float kpt_scale = DL_SCALE(p_kpt->exponent);
    const int nk = YOLO26_POSE_NKPT * YOLO26_POSE_KPT_DIMS;

    const TB *raw_box = (const TB *)p_box->data;
    const TC *raw_cls = (const TC *)p_cls->data;
    const TK *raw_kpt = (const TK *)p_kpt->data;

    // Umbral precalculado en el tipo nativo de la rama de clase: si el valor
    // crudo no llega, la celda se descarta sin pasar por sigmoid ni por coma
    // flotante.
    //
    // Se recorta al rango del tipo a proposito. inverse_sigmoid(0.01) vale -4.6 y
    // con un exponent pequeno el cociente se va de [-128,127]: con conf_thresh
    // 0.01 y escala 1/128 el umbral crudo es -589, y un cast directo a int8_t
    // seria comportamiento indefinido. Recortar mantiene el pre-filtrado
    // correcto: si el umbral cae por debajo del minimo del tipo, TODAS las
    // celdas pasan el filtro entero, que es justo lo que tiene que pasar para
    // que la comprobacion en coma flotante de mas abajo haga su trabajo.
    const float thr_f = std::floor(dl::math::inverse_sigmoid(conf_thresh) / cls_scale);
    const float lo = (float)std::numeric_limits<TC>::min();
    const float hi = (float)std::numeric_limits<TC>::max();
    const TC cls_thresh = (TC)std::min(std::max(thr_f, lo), hi);

    for (int h = 0; h < grid; h++) {
        for (int w = 0; w < grid; w++) {
            const int pixel_idx = (h * grid) + w; // NHWC
            const int cls_offset = pixel_idx * num_classes;

            float max_score = -1.0f;
            int best_cls_id = -1;
            for (int c = 0; c < num_classes; c++) {
                const TC raw_val_T = raw_cls[cls_offset + c];
                if (raw_val_T <= cls_thresh) {
                    continue;
                }
                float score = dl::math::sigmoid(dl::dequantize(raw_val_T, cls_scale));
                if (score > max_score) {
                    max_score = score;
                    best_cls_id = c;
                }
            }
            if (max_score < conf_thresh) {
                continue;
            }

            // --- Box: ltrb relativo al centro de la celda (reg_max = 1) ---
            const int box_offset = pixel_idx * 4;
            const float d_l = dl::dequantize(raw_box[box_offset + 0], box_scale);
            const float d_t = dl::dequantize(raw_box[box_offset + 1], box_scale);
            const float d_r = dl::dequantize(raw_box[box_offset + 2], box_scale);
            const float d_b = dl::dequantize(raw_box[box_offset + 3], box_scale);

            // El centro de la celda es el anchor. Para la caja se RESTA la
            // distancia; para los keypoints se SUMA el desplazamiento.
            const float anchor_x = (float)w + 0.5f;
            const float anchor_y = (float)h + 0.5f;

            YOLO26PoseResult res;
            res.category = best_cls_id;
            res.score = max_score;
            res.box[0] = (int)((anchor_x - d_l) * stride);
            res.box[1] = (int)((anchor_y - d_t) * stride);
            res.box[2] = (int)((anchor_x + d_r) * stride);
            res.box[3] = (int)((anchor_y + d_b) * stride);
            res.num_kpts = YOLO26_POSE_NKPT;

            // --- Keypoints: (kpt + anchor) * stride, vis = sigmoid ---
            const int kpt_offset = pixel_idx * nk;
            for (int k = 0; k < YOLO26_POSE_NKPT; k++) {
                const int base = kpt_offset + k * YOLO26_POSE_KPT_DIMS;
                const float kx = dl::dequantize(raw_kpt[base + 0], kpt_scale);
                const float ky = dl::dequantize(raw_kpt[base + 1], kpt_scale);
                // La visibilidad se queda en espacio de probabilidad; el
                // umbral lo aplica el consumidor (el JS no dibuja lo que
                // esta por debajo).
                const float kv = dl::dequantize(raw_kpt[base + 2], kpt_scale);

                res.kpt_x[k] = (kx + anchor_x) * (float)stride;
                res.kpt_y[k] = (ky + anchor_y) * (float)stride;
                res.kpt_v[k] = dl::math::sigmoid(kv);
            }

            out.push_back(res);
        }
    }
}

std::vector<YOLO26PoseResult> YOLO26Pose::postprocess(const std::map<std::string, dl::TensorBase *> &outputs)
{
    std::vector<YOLO26PoseResult> out;
    if (grid_sizes.empty() || grid_sizes[0] == 0) {
        printf("[YOLO26Pose] Error: rejillas sin inicializar. Llama a preprocess() primero.\n");
        return out;
    }

    dl::TensorBase *p_box[3];
    dl::TensorBase *p_cls[3];
    dl::TensorBase *p_kpt[3];
    static const char *level[3] = {"p3", "p4", "p5"};
    for (int i = 0; i < num_levels; i++) {
        char name[64];
        snprintf(name, sizeof(name), "one2one_%s_box", level[i]);
        p_box[i] = outputs.at(name);
        snprintf(name, sizeof(name), "one2one_%s_cls", level[i]);
        p_cls[i] = outputs.at(name);
        snprintf(name, sizeof(name), "one2one_%s_kpt", level[i]);
        p_kpt[i] = outputs.at(name);
    }

    // num_classes se deduce de la forma, igual que hace el componente oficial,
    // para que cambiar el dataset no obligue a recompilar.
    num_classes = p_cls[0]->shape[3];

    // Verificacion de cordura del contrato: si el modelo no trae la rama de
    // keypoints, el error tiene que ser explicito y no un crash mas tarde.
    const int kpt_channels = p_kpt[0]->shape[3];
    if (kpt_channels != YOLO26_POSE_NKPT * YOLO26_POSE_KPT_DIMS) {
        printf("[YOLO26Pose] Error: one2one_p3_kpt tiene %d canales y se esperaban %d "
               "(%d keypoints x %d dims). El .espdl no es de pose, o NKPT no coincide.\n",
               kpt_channels,
               YOLO26_POSE_NKPT * YOLO26_POSE_KPT_DIMS,
               YOLO26_POSE_NKPT,
               YOLO26_POSE_KPT_DIMS);
        return out;
    }

    out.reserve((size_t)target_k * 2);
    for (int i = 0; i < num_levels; i++) {
        const int g = grid_sizes[i];
        const int bc = dtype_code(p_box[i]->dtype);
        const int cc = dtype_code(p_cls[i]->dtype);
        const int kc = dtype_code(p_kpt[i]->dtype);

        if (bc < 0 || cc < 0 || kc < 0) {
            printf("[YOLO26Pose] Error: dtypes no soportados en one2one_%s: "
                   "box=%s cls=%s kpt=%s (solo se sabe leer INT8 e INT16)\n",
                   level[i], dtype_name(bc), dtype_name(cc), dtype_name(kc));
            return {};
        }

        // Las ocho combinaciones. Con USE_MIXED_INT16 este .espdl cae en
        // (INT16, INT16, INT8), que antes no existia y por eso el post-proceso
        // devolvia cero detecciones siempre.
        switch ((bc << 2) | (cc << 1) | kc) {
            case 0: decode_grid<int8_t, int8_t, int8_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 1: decode_grid<int8_t, int8_t, int16_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 2: decode_grid<int8_t, int16_t, int8_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 3: decode_grid<int8_t, int16_t, int16_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 4: decode_grid<int16_t, int8_t, int8_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 5: decode_grid<int16_t, int8_t, int16_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            case 6: decode_grid<int16_t, int16_t, int8_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
            default: decode_grid<int16_t, int16_t, int16_t>(p_box[i], p_cls[i], p_kpt[i], strides[i], g, out); break;
        }
    }

    // CONTRATO DE SALIDAS, impreso UNA sola vez. El exponent fija la escala de
    // desescala y es el parametro que mas facil se rompe sin fallar de forma
    // visible: si la escala es erronea los scores salen todos en 0.00, no hay
    // ninguna deteccion, y no sale ningun error.
    static bool s_printed_contract = false;
    if (!s_printed_contract) {
        s_printed_contract = true;
        printf("[YOLO26Pose] === contrato de las 9 salidas ===\n");
        for (int i = 0; i < num_levels; i++) {
            const dl::TensorBase *t[3] = {p_box[i], p_cls[i], p_kpt[i]};
            const char *n[3] = {"box", "cls", "kpt"};
            printf("[YOLO26Pose]  one2one_%s: ", level[i]);
            for (int b = 0; b < 3; b++) {
                printf("%s=%s exp=%d shape=[1,%d,%d,%d]  ", n[b],
                       dtype_name(dtype_code(t[b]->dtype)), (int)t[b]->exponent,
                       t[b]->shape[1], t[b]->shape[2], t[b]->shape[3]);
            }
            printf("\n");
        }
    }

    // BARRIDO CRUDO DEL TENSOR DE CLASE, UNA SOLA VEZ.
    //
    // "0 celdas pasaron el pre-filtrado" tiene dos causas que desde fuera se ven
    // igual, y en el fondo son opuestas:
    //
    //   a) el tensor viene vacio o lleno de valores no representables. Suele
    //      ser un problema de ENTRADA: la imagen no llega al modelo como espera.
    //   b) el tensor tiene valores variados pero todos muy negativos. El modelo
    //      corrio, leyo la imagen y no vio ninguna mano.
    //
    // El rango crudo y la cuenta de valores no cero los separan de inmediato.
    static bool s_scanned_raw = false;
    if (!s_scanned_raw) {
        s_scanned_raw = true;
        for (int i = 0; i < num_levels; i++) {
            if (dtype_code(p_cls[i]->dtype) != 1) {
                continue;   // el barrido asume INT16, que es lo que trae el modelo
            }
            const int n = grid_sizes[i] * grid_sizes[i] * num_classes;
            const int16_t *raw = (const int16_t *)p_cls[i]->data;
            int nonzero = 0;
            int32_t mn = 32767, mx = -32768;
            for (int k = 0; k < n; k++) {
                const int16_t v = raw[k];
                if (v != 0) {
                    nonzero++;
                }
                if ((int32_t)v < mn) {
                    mn = v;
                }
                if ((int32_t)v > mx) {
                    mx = v;
                }
            }
            const float scale = DL_SCALE(p_cls[i]->exponent);
            printf("[YOLO26Pose]  p%d cls: %d de %d valores no cero, crudo=[%d, %d] "
                   "escala=%.9f logit=[%.2f, %.2f] score=[%.4f, %.4f]\n",
                   i + 3, nonzero, n, (int)mn, (int)mx, scale,
                   mn * scale, mx * scale,
                   dl::math::sigmoid(mn * scale), dl::math::sigmoid(mx * scale));
        }
    }

    // DIAGNOSTICO POR FRAME, AHORRADO.
    //
    // NO se imprime en cada inferencia. A 115200 baud, ~500 caracteres son casi
    // 50 ms de escritura bloqueante: se ve en el propio log como "Post: 59 ms"
    // cuando el post-proceso real tarda microsegundos. Con eso el WDT de
    // interrupcion salta y el ESP32 se reinicia, que es exactamente lo que
    // pasaba. Un diagnostico que tumba el equipo no sirve de diagnostico.
    //
    // Solo se escribe cuando el numero de candidatas CAMBIA, y como mucho una
    // vez cada 60 inferencias (~3.5 minutos). Un cambio de valor es la senal
    // utile: dice que algo en la escena se movio.
    static int s_last_count = -1;
    static int s_frames = 0;
    s_frames++;
    const int n_out = (int)out.size();
    const bool cambio = (n_out != s_last_count);
    if (cambio || s_frames >= 60) {
        if (out.empty()) {
            s_last_count = 0;
            s_frames = 0;
            printf("[YOLO26Pose] 0 celdas sobre el pre-filtrado (grids %d/%d/%d, nc=%d, "
                   "umbral %.3f)\n",
                   grid_sizes[0], grid_sizes[1], grid_sizes[2], num_classes, conf_thresh);
        } else {
            // Los 5 mejores scores sin aplicar el umbral de la UI, con la
            // geometria del primero. Un score alto con la caja en un sitio raro
            // dice que el decodificador anda mal; con la caja donde esta la
            // mano dice que la caja va bien y hay que mirar los keypoints.
            const int top_n = n_out < 5 ? n_out : 5;
            std::partial_sort(out.begin(), out.begin() + top_n, out.end(),
                              [](const YOLO26PoseResult &a, const YOLO26PoseResult &b) {
                                  return a.score > b.score;
                              });
            s_last_count = n_out;
            s_frames = 0;
            printf("[YOLO26Pose] %d celdas. Mejores:", n_out);
            for (int i = 0; i < top_n; i++) {
                // Solo el indice, no el nombre. class_names no lleva su
                // longitud y esto corre en la ruta caliente: una lectura fuera de
                // rango seria peor que no tener el dato. El mapeo
                // 0=Detecting 1=check 2=ok ya lo loguea main.c como "clase=N".
                printf(" cls%d=%.3f", out[i].category, out[i].score);
            }
            printf("\n");
            const YOLO26PoseResult &t = out[0];
            /* OJO: estas coordenadas estan AUN en el espacio del modelo
             * (320x320 con letterbox). El mapeo a la imagen original ocurre
             * despues, en map_to_original(). Para ver las coordenadas finales
             * estan las lineas de main.c, con x/y/w/h ya en 640x480. */
            printf("[YOLO26Pose]  mejor (espacio del modelo 320): "
                   "box=[%d,%d,%d,%d] k0=(%.0f,%.0f,v=%.2f)\n",
                   t.box[0], t.box[1], t.box[2], t.box[3], t.kpt_x[0], t.kpt_y[0], t.kpt_v[0]);
        }
    }

    // Top-K NMS-free. YOLO26 no necesita NMS: la cabeza one2one ya es
    // end-to-end y su top-K por score es la seleccion definitiva.
    if ((int)out.size() > target_k) {
        std::partial_sort(out.begin(), out.begin() + target_k, out.end(),
                          [](const YOLO26PoseResult &a, const YOLO26PoseResult &b) {
                              return a.score > b.score;
                          });
        out.resize((size_t)target_k);
    }

    // El mapeo inverso de letterbox se hace al final, sobre el subconjunto
    // que sobrevive: convertir miles de keypoints descartados seria trabajo
    // tirado.
    for (auto &r : out) {
        map_to_original(r);
    }

    return out;
}
