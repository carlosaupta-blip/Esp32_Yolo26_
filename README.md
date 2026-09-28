# Detección de rostros en tiempo "casi real" con YOLO26n + ESP-DL sobre ESP32-S3

Tutorial paso a paso para replicar este proyecto de punta a punta: obtener un dataset,
entrenar un YOLO26n, cuantizarlo para el runtime de ESP-DL, y servirlo en un ESP32-S3
con cámara, streaming MJPEG y panel web de configuración.

**El proyecto no entrena dentro del repositorio de forma automática**: son tres etapas
separadas, y la etapa 1 (dataset) y la etapa 2 (entrenamiento) ocurren en Colab. La
etapa 3 (cuantización) y la etapa 4 (firmware) son las que están en este repo.

---

## Índice

1. [Qué hace y qué no hace](#1-qué-hace-y-qué-no-hace)
2. [Requisitos previos](#2-requisitos-previos)
3. [Etapa 0 — Dataset desde Roboflow](#3-etapa-0--dataset-desde-roboflow)
4. [Etapa 1 — Entrenamiento en Colab](#4-etapa-1--entrenamiento-en-colab)
5. [Etapa 2 — Cuantización a `.espdl` (Colab)](#5-etapa-2--cuantización-a-espdl-colab)
6. [Etapa 3 — Adaptar el firmware al modelo](#6-etapa-3--adaptar-el-firmware-al-modelo)
7. [Etapa 4 — Compilar, flashear y operar](#7-etapa-4--compilar-flashear-y-operar)
8. [Verificar que funciona](#8-verificar-que-funciona)
9. [Números de referencia medidos](#9-números-de-referencia-medidos)
10. [Problemas frecuentes](#10-problemas-frecuentes)
11. [Estructura del repositorio](#11-estructura-del-repositorio)
12. [Ideas futuras (no implementadas)](#12-ideas-futuras-no-implementadas)
13. [Referencias](#13-referencias)

---

## 1. Qué hace y qué no hace

### Lo que hace

- Sirve un **stream MJPEG** de la cámara en el puerto 81, con overlay de cajas sobre el
  navegador.
- Corre **YOLO26n cuantizado INT8 + LUT INT16** en el ESP32-S3 para detectar rostros.
- Expone una **interfaz web** con streaming, captura de foto, sliders de control de la
  cámara (brillo, contraste, WB, etc.) y un **umbral de confianza dinámico**.
- Mide y loguea el desglose de tiempos por etapa de cada inferencia.

### Lo que NO hace (y es importante saberlo)

- **No es "tiempo real" en el sentido de video fluido.** La inferencia tarda
  **~3 segundos por frame** en el ESP32-S3. El video va a 20 fps, pero las detecciones
  se actualizan una vez cada ~3 s. Es un detector de rostros *lento*, no un tracker.
- **El brownout detector, la alimentación y el cable USB importan.** Ver
  [Problemas frecuentes](#10-problemas-frecuentes).
- **No entrena el modelo.** Necesitas un `best.pt` previo.

> ### La advertencia más importante de este README
>
> El número **7822 ms** que verás en la documentación de ESP-DL para `yolo26n_512_s8_s3`
> son **7.8 segundos**, no 7.8 milisegundos. Ese error de escala (un decimal) es la causa
> número uno de confusión al empezar. La tabla oficial completa está en
> [Números de referencia](#9-números-de-referencia-medidos).

---

## 2. Requisitos previos

| Componente | Versión | Nota |
|---|---|---|
| **ESP-IDF** | v6.0.2 | La que se verificó en este proyecto. Windows: `C:\esp\v6.0.2\esp-idf`. |
| **Placa** | ESP32-S3 | Octal PSRAM **obligatorio** (16 MB o más). CPU 240 MHz. |
| **Cámara** | OV2640 | Configurada en `PIXFORMAT_JPEG`. |
| **Módulo** | 16 MB de flash | Por `partitions.csv`. |
| **Colab** | Runtime Python 3.12 | Solo para etapas 0-2. |
| **Cuenta Roboflow** | — | Ver [Etapa 0](#3-etapa-0--dataset-desde-roboflow). |
| **Alimentación** | USB directo, ≥1 A | Ver [Brownout](#10-problemas-frecuentes). |

Verifica que tu placa tiene PSRAM octal. En el log de arranque deberías ver la
inicialización de PSRAM. Sin ella el modelo de 2.79 MB no cabe.

---

## 3. Etapa 0 — Dataset desde [Roboflow](https://roboflow.com/)

El dataset se obtuvo de **Roboflow** (proyecto `face-recognition-ismus-fomib`,
workspace `carlos-aponte-upta`, versión 1, formato `yolo26`). Roboflow es la fuente
del dataset y de las etiquetas en formato YOLO normalizado.

### 3.1 Crear el proyecto en Roboflow

1. Ve a [roboflow.com](https://roboflow.com/) y crea una cuenta.
2. **Create Project** → tipo de proyecto **Object Detection**.
3. Sube tus imágenes y anota las cajas de los rostros. Roboflow maneja el
   *train/valid/test split* automáticamente.
4. **Generate → Version** para crear la versión 1 del dataset.
5. En **Settings → API Key** (o tu perfil → API Key) copia tu clave. **No la subas a
   git.**

### 3.2 Descargarlo en Colab

El cuaderno de cuantización (`yolo26n_esp32s3_espdl_autocuantizacion_colab.ipynb`)
puede descargar el dataset él mismo. En la **celda 2**:

```python
USE_ROBOFLOW = True
ENV_PATH = Path('/content/drive/MyDrive/Colab Notebooks/.env')
ROBOFLOW_WORKSPACE = 'carlos-aponte-upta'   # o déjalo vacío y ponlo en el .env
ROBOFLOW_PROJECT   = 'face-recognition-ismus-fomib'
ROBOFLOW_VERSION   = '1'
ROBOFLOW_FORMAT    = 'yolo26'
```

Crea el archivo `.env` en Google Drive con:

```
ROBOFLOW_API_KEY=TU_CLAVE_AQUI
```

Variables aceptadas: `ROBOFLOW_API_KEY` (o `ROBOFLOW_KEY`), `ROBOFLOW_WORKSPACE`,
`ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION`.

El cuaderno monta Drive, descarga el dataset con la API de Roboflow, y **nunca imprime
la clave**.

### 3.3 Alternativa sin Roboflow en el cuaderno

```python
USE_ROBOFLOW = False
CALIB_IMAGE_DIR = Path('/content/mis_imagenes')   # carpeta con imágenes
```

Las imágenes de calibración **no necesitan etiquetas**, solo deben ser RGB
representativas de lo que verá la cámara. El cuaderno las fuerza a RGB
(por un bug de ESP-DL, ver [Problemas frecuentes](#10-problemas-frecuentes)).

### 3.4 Estructura del dataset

Roboflow en formato YOLO exporta carpetas hermanas:

```
face-recognition-ismus-fomib-1/
├── data.yaml
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
├── valid/
│   ├── images/*.jpg
│   └── labels/*.txt
└── test/ ...
```

> **Importante:** las etiquetas están en `labels/`, **no** junto a las imágenes en
> `images/`. El cuaderno de evaluación lo tiene en cuenta; si escribes tu propio
> script de validación, no asumas `imagen.jpg → imagen.txt`.

---

## 4. Etapa 1 — Entrenamiento en Colab

> **Nota:** el cuaderno de entrenamiento de este proyecto específico
> (`EntrenamientoYOLO262`) **no está incluido en el repositorio**. El flujo de abajo
> replica el tutorial oficial de Espressif (`quantize_yolo26_roboflow.ipynb`, incluido
> en `esp-dl/examples/tutorial/.../quantize_yolo26/`) adaptado a tu dataset. Puedes
> copiar ese cuaderno como base.

### 4.1 Instalar dependencias

```python
%pip install -q ultralytics roboflow
```

### 4.2 Descargar el dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="TU_API_KEY")
project = rf.workspace("TU_WORKSPACE").project("TU_PROYECTO")
dataset = project.version(1).download("yolo26", location="dataset")
```

### 4.3 Arreglar las rutas de `data.yaml`

Roboflow a veces escribe rutas absolutas que rompen el entrenamiento local:

```python
import yaml

with open('dataset/data.yaml') as f:
    data = yaml.safe_load(f)

data['path'] = '.'
data['train'] = 'train/images'
data['val']   = 'valid/images'
data['test']  = 'test/images'

with open('dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)
```

### 4.4 Entrenar

```python
import os
from ultralytics import YOLO

os.environ["COMET_MODE"] = "disabled"
os.environ["WANDB_MODE"] = "disabled"

model = YOLO('yolo26n.pt')       # pesos base preentrenados

results = model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=512,                   # resolución de entrenamiento
    batch=24,                    # baja si te quedas sin VRAM
    device=0,
    plots=True,
    optimizer='MuSGD',
    lr0=0.005,
    lrf=0.01,
)
```

### 4.5 Extraer los pesos

```python
import shutil
from glob import glob

weights = glob("runs/detect/train*/weights/best.pt")
latest  = max(weights, key=os.path.getctime)
shutil.copy(latest, "best.pt")
```

### 4.6 Reglas que importan

| Regla | Por qué |
|---|---|
| Debe ser **YOLO26n con `reg_max = 1`**. | El cuaderno de cuantización aborta si la arquitectura no coincide. |
| El modelo debe tener **`nc` = número de clases de tu dataset**. | Ej. 1 para solo rostros. |
| La resolución de **entrenamiento** y la de **cuantización son independientes.** | Ver [Sección 5.1](#51-la-resolución-no-es-negociable). |
| Sube `best.pt` a Google Drive. | El cuaderno de cuantización lo lee de ahí. |

---

## 5. Etapa 2 — Cuantización a `.espdl` (Colab)

Este es el cuaderno principal de este repo:
**`yolo26n_esp32s3_espdl_autocuantizacion_colab.ipynb`**.

### 5.1 La resolución no es negociable

Hay **tres** resoluciones distintas en juego y se confunden entre sí:

| Concepto | Dónde | Valor en este proyecto |
|---|---|---|
| Resolución de **entrenamiento** | Etapa 1, `model.train(imgsz=...)` | 512 |
| Resolución del **modelo exportado** | Etapa 2, `IMG_SIZE` en celda 2 | **320** |
| Resolución de **captura** de la cámara | Firmware, `FRAMESIZE_*` | VGA (640×480) |

La que importa para el chip es **`IMG_SIZE`**. Está **horneada dentro del `.espdl`**:
no se puede cambiar en el firmware. Un modelo de 512 **no puede** correr a 320.

**Por qué 320 y no 512:**

| `IMG_SIZE` | Inferencia (S3) | FPS |
|---|---|---|
| 512 | 7822 ms *(oficial)* | 0.13 |
| **320** | **~2978 ms** *(medido)* | **0.32** |
| 256 | ~1960 ms *(estimado)* | 0.51 |
| 224 | ~1500 ms *(estimado)* | 0.67 |

El costo escala con el **cuadrado** de los píxeles: `(320/512)² = 0.39`. Baja a 320 es
la mayor mejora sin perder demasiada precisión en rostros.

### 5.2 Configurar la celda 2

```python
WEIGHTS_PATH = Path('/content/drive/MyDrive/.../best.pt')
RUN_MODE     = 'full'          # la ruta validada; 'autoquant' es experimental
IMG_SIZE     = 320             # múltiplo de 32
```

Deja el resto con los valores por defecto. Los que importan:

| Parámetro | Default | Significado |
|---|---|---|
| `USE_MIXED_INT16` | `True` | Backbone INT8 + cabezas INT16 + LUT. Es la ruta validada. |
| `USE_TQT_LUT` | `True` | Fusión LUT INT16. |
| `BATCH_SIZE` / `CALIB_STEPS` | `4` / `32` | 128 imágenes de calibración en 32 pasos. |
| `TQT_COLLECTING_DEVICE` | `'cuda'` si hay GPU | Mantiene pesos en GPU. Baja a `'cpu'` si da `OutOfMemoryError`. |
| `MAX_CALIB_IMAGES` | `128` | Baja si te quedas sin VRAM. |

### 5.3 Ejecutar

**Runtime → Reiniciar sesión**, luego **Run all**. No reejecutes celdas intermedias.

> **Por qué el reinicio de sesión es obligatorio:** la celda 1 reinstala NumPy
> (`numpy==1.26.4`) en un directorio aislado para evitar el conflicto ABI NumPy 1.x/2.x.
> Si ves `numpy.dtype size changed`, reinicia la sesión y vuelve a empezar desde la
> celda 1.

> **Por qué no repetir celdas:** `load_onnx_graph()` no tiene caché. Si se vuelve a
> llamar, reemplaza el grafo cuantizado por uno nuevo sin cuantizar y el resultado
> mide un modelo que en realidad es FP32
> ([esp-dl#325](https://github.com/espressif/esp-dl/issues/325)).

### 5.4 Qué hace cada celda

| Celda | Qué hace |
|---|---|
| 1 | Dependencias: `esp-ppq==1.3.11`, `onnx==1.17.0`, `ultralytics==8.4.7`, `roboflow`, etc. |
| 2 | Config, descarga de Roboflow, localiza `best.pt` e imágenes. |
| 3 | Descarga el código auxiliar de ESP-DL a un **commit fijo** (reproducibilidad). |
| 4 | DataLoader de calibración (fuerza RGB, corrige el bug de canales). |
| 5 | Exporta `best.pt` → ONNX estático con el parche de la cabeza YOLO26. |
| 6A | (Opcional) AutoQuant alternativo. |
| 6B | Pipeline oficial: **PTQ → TQT → fusión LUT**. |
| 7 | Cirugía del grafo: separa box/cls, poda, exporta el `.espdl`. |
| 8 | **Evaluación**: inferencia emulada + mAP50-95 del grafo cuantizado. |
| 9 | Reporte, SHA256 y descarga del ZIP. |

### 5.5 La celda 8 es tu judge de calidad

Antes de flashear, **mira el mAP50-95** que reporta. Compara con el `YOLO(best.pt).val()`
en FP32. Si el mAP se desploma respecto a la resolución que elegiste, baja la resolución
del entrenamiento o sube `IMG_SIZE`.

Imprime además el **desglose por etapa** (`espppq_dtype_string`) de las seis salidas,
que debe coincidir con el contrato que espera el firmware:

```
one2one_p3_box   shape=(1, 40, 40, 4) dtype=INT16 per-tensor [-32768..32767] ACTIVATED
one2one_p3_cls   shape=(1, 40, 40, 1) dtype=INT16 per-tensor [-32768..32767] ACTIVATED
one2one_p4_box   shape=(1, 20, 20, 4) ...
one2one_p5_cls   shape=(1, 10, 10, 1) ...
```

(Las dimensiones son para `IMG_SIZE=320`: 320/8=40, 320/16=20, 320/32=10.)

### 5.6 Qué descarga

Un ZIP con cuatro archivos:

```
yolo26n_320_s8_s3.espdl            # el modelo (2.79 MB)
yolo26n_320_s8_s3.info             # volcado legible del grafo
yolo26n_320_s8_s3.json             # configuración de cuantización por tensor
quantization_report.json           # metadatos, SHA256, versiones, tamaño
```

**Verifica el SHA256** del `.espdl` contra el `model_sha256` del reporte. Es la única
forma de confirmar que el archivo no se corrompió al descargarlo.

---

## 6. Etapa 3 — Adaptar el firmware al modelo

> **La etapa 3 y la 4 se invierten según tu caso.** En este proyecto ya están hechas.
> Si partes de cero, hazlas después de tener el `.espdl`, porque necesitas saber el
> nombre y la resolución exactos del modelo.

### 6.1 Copiar el modelo

```powershell
Copy-Item "yolo26n_320_s8_s3_package\yolo26n_320_s8_s3.espdl" `
          "main\models\yolo26n_face_320_s8_s3.espdl"
```

El prefijo `face_` es una convención del proyecto para diferenciarlo del modelo oficial
de 80 clases del "zoo" de ESP-DL.

### 6.2 Actualizar la tabla de particiones

`partitions.csv` define el layout de flash:

```
nvs,      data, nvs,     0x9000,   0x6000,
otadata,  data, ota,     0xF000,   0x2000,
app0,     app,  ota_0,   0x20000,  0xC80000,      # 12.5 MiB
spiffs,   data, spiffs,  0xCA0000, 0x360000,      # 3.375 MiB
```

`app0` debe ser **mayor que el `.espdl`**. Con el modelo de 2.79 MB, 5 MiB bastan
(`0x500000`), pero 12.5 MiB da margen y no estorba. El reporte de cuantización incluye
el campo `fits_current_app_partition` para comprobarlo.

> **Si cambias `partitions.csv`, borra el flash completo antes de flashear**, porque
> `CONFIG_PARTITION_TABLE_MD5=y` y el MD5 de la tabla cambia:
> ```powershell
> idf.py fullclean
> idf.py erase-flash
> idf.py flash monitor
> ```

### 6.3 CMakeLists.txt

Apunta al nombre nuevo. El símbolo generado será
`_binary_yolo26n_face_320_s8_s3_espdl_start` — fíjate en el sufijo `_espdl`, que
**forma parte del símbolo** (se deriva del nombre de archivo *con* extensión
saneada):

```cmake
target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/models/yolo26n_face_320_s8_s3.espdl"
    BINARY
)
```

### 6.4 yolo_bridge.cpp — los 4 símbolos

```cpp
extern const uint8_t yolo26n_face_320_s8_s3_espdl_start[]
    asm("_binary_yolo26n_face_320_s8_s3_espdl_start");
extern const uint8_t yolo26n_face_320_s8_s3_espdl_end[]
    asm("_binary_yolo26n_face_320_s8_s3_espdl_end");
```

Y actualiza las **4 referencias** a esos símbolos en el archivo: las líneas del
`extern` (2), la del cálculo de `model_size`, la del chequeo de alineación a 16 bytes,
y la del `new dl::Model(...)`.

**El nombre del archivo y el del símbolo deben coincidir exactamente.** El símbolo se
deriva del filename con la extensión saneada, así que
`models/yolo26n_face_320_s8_s3.espdl` produce `_binary_yolo26n_face_320_s8_s3_espdl_start`
(el punto se convierte en `_espdl`, no se elimina). Si olvidas ese sufijo, el enlazador
falla con `undefined reference`.

Las líneas exactas a actualizar en `yolo_bridge.cpp` en este proyecto son **42, 43, 44,
45, 95, 102 y 108**.

### 6.5 yolo_bridge.cpp — el contrato de clases

```cpp
static const char *kFaceClasses[] = {"face"};   // debe coincidir con `nc`
```

`nc = 1` en este proyecto. Si tu dataset tiene más clases, ponlas todas aquí, en el
mismo orden que en el entrenamiento.

**El resto de la resolución es automático:** `s_model_in` se lee del `.espdl` con
`get_inputs()`, así que no hay ningún `320` ni `512` cableado en la lógica del firmware.

### 6.6 Fusión SIMD LUT

El `.espdl` usa la ruta INT8 + LUT INT16, que **requiere ESP-DL >= 3.3.11**:

```yaml
# main/idf_component.yml
espressif/esp-dl: ^3.3.12
```

---

## 7. Etapa 4 — Compilar, flashear y operar

### 7.1 Preparar el entorno

```powershell
call C:\esp\v6.0.2\esp-idf\export.bat
```

### 7.2 Compilar y flashear

```powershell
idf.py build
idf.py -p COM5 flash monitor
```

Sustituye `COM5` por tu puerto. La placa debe enumerarse; si no, instala el driver
CP210x/CH340 según tu cable.

### 7.3 Conectarse

| Interfaz | URL |
|---|---|
| Panel principal | `http://<IP-DE-TU-PLACA>/` |
| Configuración | `http://<IP-DE-TU-PLACA>/config` |
| Stream MJPEG | `http://<IP-DE-TU-PLACA>:81/stream` |
| Detecciones (JSON) | `http://<IP-DE-TU-PLACA>/api/detections` |
| Métricas | `http://<IP-DE-TU-PLACA>/api/metrics` |

Si no hay credenciales WiFi guardadas, la placa arranca un **punto de acceso**:
SSID `ESP32_CAM`, contraseña `123456789`. Conéctate a él y entra a `192.168.4.1`.

### 7.4 Configuración por NVS

La página `/config` guarda la configuración en NVS (no volátil). Cambios relevantes:

- Credenciales WiFi (SSID, contraseña, DHCP o IP estática).
- Parámetros de cámara que se guardan: brillo, contraste, saturación, WB, nitidez,
  denoise, etc.

---

## 8. Verificar que funciona

### 8.1 En el log serie

Al arrancar deberías ver, en este orden:

```
I (xxxx) main: -> app_main
I (xxxx) main: Reinicio anterior: encendido
W (xxxx) wifi: -> wifi_manager_init
W (xxxx) wifi: -> wifi_manager_start
I (xxxx) CAM: Cámara inicializada correctamente (driver v2.1.7)
W (xxxx) httpd: Servidor HTTP iniciado en puerto 80
I (xxxx) stream_srv: Servidor de streaming MJPEG en puerto 81
I (xxxx) YOLO_BRIDGE: Modelo embebido: 2920048 bytes (2.79 MiB)
I (xxxx) YOLO_BRIDGE: YOLO26 listo: entrada 320x320x3, nc=1, k=32, umbral=0.01
```

### 8.2 El log de tiempos (la métrica clave)

Cada inferencia imprime:

```
I (977146) main: Tiempos | JPEG: 155 ms | Pre: 17 ms | Inf: 2978 ms | Post: 2 ms | Total: 3154 ms -> 0.317 FPS
I (977156) main: ✅ Inferencia OK [480x320] -> 2 deteccion(es) sobre umbral 0.10 (mejor: 79.37%)
I (977160) main:    [0] clase=0 score=0.794 [x:145.5, y:70.5, w:106.5, h:135.0]
I (977161) main:    [1] clase=0 score=0.323 [x:162.0, y:69.0, w:96.0, h:132.0]
```

Qué mirar:

| Campo | Significado | Referencia |
|---|---|---|
| `Inf` | `model->run()`. **El número que importa.** | ~2978 ms a 320 |
| `JPEG` | Decodificación software del JPEG | ~155 ms (VGA) |
| `Pre` | Letterbox + cuantización LUT | ~17 ms |
| `Post` | `decode_grid` + top-K | ~2 ms |
| `FPS` | `1 / Total` | ~0.32 |

Si `Inf` te sale cerca de 7822 ms con un modelo de 320, algo está mal: el modelo no
es el que crees, o el reloj de CPU está mal configurado.

### 8.3 En el navegador

1. Abre el panel y pulsa **Iniciar** para el stream.
2. Debes ver el video con recuadros verdes sobre cada rostro detectado.
3. Mueve el **slider de umbral** y observa cómo aparecen y desaparecen cajas. Este
   control es el que decide cuántas detecciones se consideran reales.
4. Los sliders de cámara (brillo, contraste, WB…) aplican en vivo.

---

## 9. Números de referencia medidos

### 9.1 Tabla oficial de ESP-DL (para YOLO26n INT8)

De [`models/yolo26/README.md`](https://github.com/espressif/esp-dl/blob/master/models/yolo26/README.md):

| Modelo | Entrada | Preprocess | **model(ms)** | Postprocess | mAP50-95 |
|---|---|---|---|---|---|
| `yolo26n_512_s8_p4` | 512×512 | 12.0 | **2067.0** | 13.0 | 0.365 |
| `yolo26n_640_s8_p4` | 640×640 | 17.0 | **3474.0** | 21.0 | 0.387 |
| **`yolo26n_512_s8_s3`** | 512×512 | 34.0 | **7822.0** | 23.0 | 0.363 |
| `yolo26n_640_s8_s3` | 640×640 | 51.0 | **13107.0** | 36.0 | 0.384 |

> **Reintento de la confusión de los "7 ms":** no existe ningún número de 6-7 ms para
> YOLO26n en ningún chip de Espressif. Los números de dos dígitos son del
> **preprocess** y **postprocess**, no de la inferencia. El ejemplo oficial
> (`examples/yolo26_detect`) muestra logs del **ESP32-P4**, no del S3, y ahí la
> inferencia son 2.07 s. Si lees "512 × 12 ms", esos 12 ms son el preprocesado.

### 9.2 Mediciones de este proyecto (ESP32-S3, 240 MHz, PSRAM octal 80 MHz)

| Métrica | 512 (oficial) | 320 (medido) |
|---|---|---|
| `model->run()` | 7822 ms | **2978 ms** |
| `fmt2rgb888` (JPEG→RGB888) | — | 155 ms |
| Preprocess | 34 ms | 17 ms |
| Postprocess | 23 ms | 2 ms |
| **Total por frame** | ~7879 ms | **3154 ms** |
| FPS efectivo | 0.127 | **0.317** |
| Tamaño del `.espdl` | 9.31 MB | **2.79 MB** |

El ratio de inferencia (2.63×) coincide con el teórico de solo píxeles (2.56×). Esto
**confirma** que el hardware y la configuración están correctos.

### 9.3 Memoria

| Recurso | Uso |
|---|---|
| `.espdl` en flash | 2.79 MB (`app0` de 12.5 MiB) |
| Modelo residente en PSRAM | ~2.79 MB (`param_copy: true`) |
| Buffer RGB en PSRAM | 900 KB (640×480×3) |
| PSRAM total del chip | 16 MB |

---

## 10. Problemas frecuentes

### 🔴 `E BOD: Brownout detector was triggered`

**Causa:** la alimentación se queda corta. **No es un bug de código.**

El ESP32-S3 ya está en el nivel 7 de brownout (2.44 V), que es **el más permisivo de
los 8** que existen. No hay ningún ajuste de firmware que lo baje más; bajar de nivel lo
haría saltar *antes*.

**Solución, en orden:**

1. **Conecta directo a un puerto USB de la PC**, no a un hub.
2. **Cambia el cable USB.** Los cables de carga-only o finos son la segunda causa más
   común. Usa uno de datos gruesos y cortos.
3. **Añade un condensador de bulk:** 470 µF electrolytico + 100 nF cerámico entre 5V y
   GND, y entre 3V3 y GND. Sostiene los picos de PSRAM + WiFi. Cuesta céntimos.
4. **Fuente de 2 A** en vez del USB de la laptop.

Al arrancar, el log ahora dice `Reinicio anterior: BROWNOUT` si fue eso. Si el corte
ocurre *después* de `-> app_main`, la última línea del log antes del corte te dice en
qué etapa.

### 🟠 `Guru Meditation: LoadProhibited` en `cJSON.c:1850`

**Causa:** construir el JSON con un árbol cJSON (≈9 `malloc` por request) y serializarlo
con `cJSON_PrintUnformatted()`, cuyo buffer de heap empieza en 256 B y se duplica en
cada `ensure()`. Bajo presión de heap (con el modelo de 2.79 MB y el buffer RGB
residentes), la asignación falla y se produce un acceso inválido. **Era intermitente**:
requests idénticos funcionaban un instante antes.

**Solución (ya aplicada en este proyecto):** armar el JSON con `snprintf` en un buffer
de **pila**. Cero `malloc`, imposible que falle por falta de memoria.

> **Regla general:** en el firmware del ESP32, no construyas JSON con cJSON en un
> endpoint que se llame en bucle. `snprintf` a un buffer de pila es más rápido, no
> necesita heap, y no puede reventar.

### 🟠 `FileNotFoundError: 'e'` al evaluar

**Causa:** `data['val']` de Ultralytics puede devolver un **string**, no una lista. Hacer
`[str(p) for p in data['val']]` recorre sus **caracteres**, y `Image.open('e')` falla.

**Solución (ya aplicada):** la función `resolve_eval_paths()` normaliza los cinco
formatos posibles (lista, string, `Path`, lista de un elemento, `.txt`) y resuelve rutas
relativas contra la carpeta del `data.yaml`.

### 🟠 El slider de umbral no hace nada

**Causa:** el handler tenía `char buf[5]` con una guarda `remaining >= sizeof(buf)`, así
que rechazaba **toda** petición con HTTP 400 (el navegador manda `{"threshold":0.25}`,
16 bytes). El endpoint estaba registrado pero inalcanzable.

**Solución (ya aplicada):** buffer de 64 B y validación de rango `[0.01, 1.0]`.

### 🟡 Solo aparece 1 detección

**Causa:** el firmware guardaba solo la de mayor score y descartaba el resto. El
post-proceso NMS-free de YOLO26 y su top-K (`kTargetK = 32`) ya deciden qué es real.

**Solución (ya aplicada):** se publican todas las detecciones que superen el umbral, en
un array JSON, ordenadas por score descendente. La UI dibuja todas.

> **También verifica el orden.** `yolo26.cpp` usa `std::nth_element`, que es una
> *partición parcial*: deja los mejores al principio pero **sin guarantee de orden**.
> El firmware aplica un `qsort` explícito.

### 🟡 `expected input[1, 1, H, W] to have 3 channels`

**Causa:** bug de ESP-DL ([#329](https://github.com/espressif/esp-dl/issues/329)). Su
`CaliDataset` solo convierte el modo `"L"`, así que los PNG con canal alfa (`RGBA`) o de
paleta (`P`) llegan al grafo con 1 canal y la primera conv aborta.

**Solución (ya aplicada en la celda 4 del cuaderno):** forzar la conversión incondicional
a RGB y validar que el tensor tenga 3 canales, con un error que apunte al archivo culpable.

### 🟡 El modelo no cabe en la partición

**Síntoma:** `ADVERTENCIA: el modelo no cabe en la partición app0 actual de 5 MiB`.

**Solución:** agranda `app0` en `partitions.csv` y haz `erase-flash`. Con el modelo de
2.79 MB, 5 MiB bastan; si baja a 256 o 224, también.

### 🟡 `numpy.dtype size changed`

**Causa:** conflicto ABI entre NumPy 1.x y 2.x en Colab.

**Solución:** **Runtime → Reiniciar sesión** y reejecutar desde la celda 1. La celda 1
instala NumPy 1.26.4 en un directorio aislado precisamente para evitar esto.

### 🟡 El resultado del `.espdl` mide un modelo FP32

**Causa:** reejecutar `load_onnx_graph()` ([esp-dl#325](https://github.com/espressif/esp-dl/issues/325)).
No tiene caché, así que reemplaza el grafo cuantizado por uno nuevo sin cuantizar.

**Solución:** **Run all** una sola vez, sin reejecutar celdas intermedias.

---

## 11. Estructura del repositorio

```
.
├── main/                                  # Firmware ESP32-S3
│   ├── main.c                             # app_main, tareas, endpoints /api/detections
│   ├── yolo_bridge.cpp                    # Puente C++: dl::Model + procesador YOLO26
│   ├── yolo_bridge.h                      # Contrato y documentación de la API
│   ├── camera.c / .h                      # Configuración OV2640 (pines, XCLK, JPEG)
│   ├── web_server.c                       # Panel, /api/camera, /api/threshold, etc.
│   ├── stream_server.c                    # Stream MJPEG en puerto 81
│   ├── mjpeg_stream.c                     # Handler MJPEG alternativo
│   ├── wifi_manager.c                     # Conexión STA + fallback AP
│   ├── config_store.c / .h                # Configuración persistente en NVS
│   ├── led_ctrl.c / .h                    # LED de flash
│   ├── metrics.c / .h                     # JSON de métricas del sistema
│   ├── yolo_frame.h                       # Estructura del frame encolado
│   ├── CMakeLists.txt                     # Registro de componentes + .espdl
│   ├── idf_component.yml                  # Dependencias del componente manager
│   ├── models/                            # .espdl embebido
│   └── www/                               # Panel web (index, config, script.js, css)
│
├── esp-dl/                                # Repo oficial de ESP-DL (tutoriales)
│   └── examples/tutorial/.../quantize_yolo26/
│       ├── quantize_yolo26_roboflow.ipynb   # Tutorial de referencia (LEGO)
│       ├── quantize_yolo26_coco.ipynb       # Variante COCO
│       ├── requirements.txt
│       └── README.md
│
├── yolo26n_esp32s3_espdl_autocuantizacion_colab.ipynb   # ETAPA 2 (este repo)
├── yolo26n_320_s8_s3_package/              # Salida de la cuantización
├── partitions.csv                          # Layout de flash
├── sdkconfig.defaults                      # Config de compilación por defecto
└── main/components/                        # Dependencias descargadas
```

### Mapa de responsabilidades

| Archivo | Responsabilidad |
|---|---|
| `yolo_bridge.cpp` | Cargar el `.espdl`, correr el pre/post-procesado oficial de ESP-DL, mapear el letterbox inverso. |
| `main.c` | Orquestación: cola de frames, temporización, estado de detecciones, JSON. |
| `camera.c` | Sensor OV2640: pines, XCLK, formato JPEG, parámetros en vivo. |
| `web_server.c` | Panel web y API de control. |
| `stream_server.c` | Servidor MJPEG dedicado (puerto 81) para no competir con el panel. |

---

## 12. Ideas futuras (no implementadas)

Estas ideas se evaluaron y **se decidió NO implementarlas todavía**. Se documentan
porque el análisis sigue siendo válido.

### Enviar el frame a un servidor externo cuando hay detección

**Idea:** cuando YOLO26 detecte un rostro, enviar ese frame a un servidor remoto para que
otro modelo lo clasifique y devuelva contexto (identidad, expresion, edad, etc.).

**Estado:** viable, complejidad media-baja. Requiere definir servidor, coste y protocolo.

**Lo que ya existe y se reutilizaría:**

- `esp_http_client` está disponible en ESP-IDF.
- mbedTLS con el CA bundle completo ya está compilado
  (`CONFIG_MBEDTLS_CERTIFICATE_BUNDLE_DEFAULT_FULL=y`), así que **HTTPS no requiere
  configuración extra**.
- El JPEG **ya está en RAM** cuando hace falta: `stream_server.c` ya hace `malloc` +
  `memcpy` del JPEG para la cola de YOLO. No hay que recomprimir.

**Presupuesto de banda:** un frame VGA JPEG a calidad 8 son ~30-50 KB. A 0.32
detecciones/segundo:

| | Bytes/seg | vs. stream actual |
|---|---|---|
| Upload @ 0.32/s | **~13 KB/s** | el stream ya manda ~600 KB/s |

Es un **~2%** del tráfico actual. La idea de "solo cuando hay detección" es
prácticamente gratis.

**Las piezas faltantes (3, ~150 líneas):**

1. Una cola `detect_event_queue` de profundidad 1-2 con el JPEG + metadatos. Se encola
   *después* de la inferencia, solo si hubo detección.
2. Una tarea de upload con `esp_http_client`, en **prioridad baja**, que drena la cola.
   Nunca debe bloquear la inferencia.
3. Configuración (URL, token, timeout) en `config_store`.

**Riesgos y mitigaciones:**

| Riesgo | Mitigación |
|---|---|
| `esp_http_client` está pensado para arranque lento, no para flujo rápido. Problema conocido con HTTPS por la subida de contexto mbedTLS (~5-10 KB RAM, ~1 s de CPU). | POST en HTTP plano si el servidor es propio; HTTPS solo si se cruza internet. **Subir solo el ROI recortado**, no el frame completo: un rostro de 96×132 son ~3 KB en vez de 40 KB. |
| Contención de WiFi con el stream a 20 fps. | Tarea de upload a prioridad baja hace que el stream la preempt. A 2% del tráfico es medible pero no grave. |
| Latencia: el frame que se sube es el que *causó* la detección, ~3 s después de capturarse. | Semánticamente es lo correcto. Si se necesita el frame *actual* en vez del que disparó, hay que capturarlo aparte, lo que duplica el tráfico. |
| Token de auth en el firmware. | Va en `config_store` (NVS), nunca hardcodeado. ⚠️ Aun así es extraíble por JTAG. |

**Si el modelo complementario clasifica la escena completa** (no solo el rostro), el
frame entero pesa ~40 KB y HTTPS + `esp_http_client` se vuelve el camino lento. En ese
caso la alternativa mejor es **WebSocket**, y la dependencia
`espressif/esp_websocket_client` **ya está en `idf_component.yml` sin usar**. WebSocket
da un canal persistente con mucha menos sobrecarga por mensaje.

**Decisión pendiente:** ¿el modelo complementario clasifica el **rostro (ROI)** o la
**escena completa (frame entero)**? ¿el servidor es **local o remoto**? Eso decide
HTTP vs WebSocket y si hace falta TLS.

### Otras ideas evaluadas y descartadas

| Idea | Por qué no |
|---|---|
| **DMA para acelerar la inferencia** | ESP-DL **no usa DMA**. El único accelerate de imagen (`resize_ppa`) está bajo `#if CONFIG_SOC_PPA_SUPPORTED`, que es **solo ESP32-P4**. El S3 no tiene PPA ni ISP. Además, el propio ESP-IDF advierte que el DMA a PSRAM tiene ancho de banda limitado y compite con la CPU por el mismo bus MSPI. El recuadro "DMA Pipeline" del diagrama de ESP-DL está marcado como **"Developing"**, no lanzado. |
| **PSRAM de 80 MHz a 120 MHz** | Experimental. Requiere `CONFIG_IDF_EXPERIMENTAL_FEATURES` y tiene un modo de fallo documentado: accesos a PSRAM/flash caen **aleatoriamente** si la temperatura varía ±20 °C. No existe ninguna medición oficial de ESP-DL que cuantifique la ganancia. |
| **Cuantización "streaming"** | Existe (`esp_ppq/parser/espdl/espdl_streaming.py`), pero es para inferencia **temporal**: inserta ventanas deslizantes de frames previos y agranda tensores. Exige que las convoluciones no tengan padding inferior, cosa que YOLO26 sí tiene. Para un detector de un solo frame es **más** trabajo, no menos. |
| **Mover la inferencia al núcleo 0** | Los dos núcleos son idénticos. ESP-DL ya usa ambos automáticamente (`RUNTIME_MODE_AUTO` parte las convoluciones grandes). El WiFi y la cámara están en el núcleo 0 a propósito: moverlos al 1 competiría por cache con la inferencia. |
| **A 320, cuántas capas usan doble núcleo?** | Solo las 2 primeras (320×160 y 160×80 superan el umbral `input_height >= 100 && input_width >= 50`). A 512 usaban las cuatro. Bajar a 320 quitó parte del paralelismo dual-core, y por eso el ratio real (2.63×) fue algo peor que el teórico (2.56×). |

---

## 13. Referencias

### Oficiales de Espressif

- **Tutorial de cuantización YOLO26** (el que sigue este proyecto):
  https://github.com/espressif/esp-dl/tree/master/examples/tutorial/how_to_quantize_model/quantize_yolo26
- **Cuantización en ESP-DL** (docs):
  https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html
- **Model zoo YOLO26** (tablas de benchmark oficiales):
  https://github.com/espressif/esp-dl/tree/master/models/yolo26
- **Procesador YOLO26** (letterbox + `decode_grid` NMS-free):
  https://github.com/espressif/esp-dl/tree/master/models/yolo26
- **Ejemplo `yolo26_detect`**:
  https://github.com/espressif/esp-dl/tree/master/examples/yolo26_detect
- **AutoQuant**:
  https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/auto_quantization/how_to_use_AutoQuant.html
- **Componente ESP-DL** (registry; verificar versión):
  https://components.espressif.com/components/espressif/esp-dl
- **Rendimiento de operadores** (SIMD vs C en S3/P4):
  https://github.com/espressif/esp-dl/blob/master/operator_performance.md
- **Rendimiento por SoC** (S3 vs S31 vs P4):
  https://github.com/espressif/esp-dl/blob/master/benchmark_report.md
- **Configuración de flash y PSRAM en ESP32-S3** (incluye el riesgo de 120 MHz):
  https://docs.espressif.com/projects/esp-idf/en/v6.1/esp32s3/api-guides/flash_psram_config.html
- **Soporte de RAM externa en ESP32-S3** (advertencias sobre DMA):
  https://docs.espressif.com/projects/esp-idf/en/v5.2.2/esp32s3/api-guides/external-ram.html

### Dataset

- **Roboflow**: https://roboflow.com/ — fuente del dataset y del etiquetado.

### Issues upstream revisados

| Issue | Qué cubre |
|---|---|
| [#329](https://github.com/espressif/esp-dl/issues/329) | `CaliDataset` solo convierte imágenes `"L"`; `"RGBA"`/`"P"` rompen la calibración |
| [#325](https://github.com/espressif/esp-dl/issues/325) | Pérdida de `tracing_operation_meta` al recargar celdas |
| [#288](https://github.com/espressif/esp-dl/issues/288) | El ONNX de YOLO26 incluye el postproceso de bbox y rompe `/model.23/Concat` |
| [#328](https://github.com/espressif/esp-dl/issues/328) | `AddLUTPattern` ignora el nombre canónico que devuelve `add_lut()` |
| [#322](https://github.com/espressif/esp-dl/issues/322) | `ImportError` de `TrainedQuantizationThresholdPass` por import circular |
| [#336](https://github.com/espressif/esp-dl/issues/336) | `validator.py` asume NCHW con box+cls en un solo Concat; el contrato real es 6 tensores NHWC separados |
