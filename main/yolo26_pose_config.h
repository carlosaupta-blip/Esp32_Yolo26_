#ifndef YOLO26_POSE_CONFIG_H
#define YOLO26_POSE_CONFIG_H

/* ---------------------------------------------------------------------------
 * Constantes del contrato de POSE. C puro, sin nada de C++.
 * ---------------------------------------------------------------------------
 * ESTE ARCHIVO EXISTE POR UNA RAZON CONCRETA.
 *
 * main.c es un archivo C, y lo compila xtensa-esp-elf-gcc con -std=gnu23. Si
 * el header que el incluye arrastra cabeceras de C++ de ESP-DL, la cadena
 * acaba pidiendo <cstdint>, que es un header de C++ que el driver de C no
 * tiene en su path de busqueda. El error sale en un sitio que no tiene nada
 * que ver:
 *
 *   dl_image_color_isa.hpp:2:10: fatal error: cstdint: No such file or directory
 *
 * blamed a ESP-DL y a la ruta del proyecto, cuando la cadena real es:
 *
 *   main.c  (C)
 *     -> yolo_bridge.h
 *       -> yolo26_pose.hpp        <-- clase C++ que declara el procesador
 *         -> dl_image_preprocessor.hpp
 *           -> dl_image_process.hpp
 *             -> dl_image_color.hpp
 *               -> dl_image_color_gray2gray.hpp
 *                 -> isa/dl_image_color_isa.hpp -> <cstdint>   <-- BOOM
 *
 * main.c solo necesita el struct y las declaraciones, que son C puro. Las dos
 * macros viven aqui para poder dimensionar los arrays de yolo_pose_t sin
 * incluir nada de C++. yolo26_pose.hpp y yolo_bridge.h incluyen este archivo.
 * ---------------------------------------------------------------------------
 */

/* Numero de keypoints que predice el modelo.
 *
 * ESTE NUMERO TIENE QUE COINCIDIR con head.kpt_shape del best.pt. No se deduce
 * ni del pretrained ni del dataset: el cuaderno lo lee del modelo e imprime
 *   POSE DETECTADO: kpt_shape=(6, 3) -> NKPT=6, KPT_DIMS=3, nk=18
 * y si el .espdl trae otra cosa, yolo_init() aborta nombrando la salida que
 * falta en vez de dejar que el post-proceso lea basura.
 *
 * Para este proyecto: 6 keypoints de mano (muneca + punta de cada dedo),
 * KPT_DIMS = 3 (x, y, visibilidad) -> 18 canales por nivel. */
#define YOLO26_POSE_NKPT 6

/* Canales de salida de la rama de keypoints: x, y, visibilidad. */
#define YOLO26_POSE_KPT_DIMS 3

#endif // YOLO26_POSE_CONFIG_H
