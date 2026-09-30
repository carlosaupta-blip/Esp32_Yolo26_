document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('cam-view')) initCameraPage();
  if (document.getElementById('wifi-form')) initConfigPage();
});

function initCameraPage() {
  const camView = document.getElementById('cam-view');
  const noSignal = document.getElementById('no-signal');
  const btnStream = document.getElementById('btn-stream');
  let streamActive = false;
  let camDebounce = null;

  const overlayCanvas = document.getElementById('overlay-canvas');
  const ctx = overlayCanvas.getContext('2d');

  function resizeCanvas() {
    const rect = camView.getBoundingClientRect();
    overlayCanvas.width = rect.width || 640;
    overlayCanvas.height = rect.height || 480;
    console.log('📐 Canvas redimensionado:', overlayCanvas.width, 'x', overlayCanvas.height);
  }

  window.addEventListener('resize', resizeCanvas);
  camView.addEventListener('load', () => {
    console.log('🖼️ Imagen cargada');
    resizeCanvas();
  });

  btnStream.addEventListener('click', () => {
    if (!streamActive) {
      camView.src = `http://${location.hostname}:81/stream`;
      camView.style.display = 'block';
      noSignal.style.display = 'none';
      btnStream.textContent = '⏸ Detener';
      streamActive = true;
      setTimeout(resizeCanvas, 200);
      startDetections();
    } else {
      camView.src = '';
      camView.style.display = 'none';
      noSignal.style.display = 'block';
      noSignal.innerHTML = '<span>📷</span>Stream detenido';
      btnStream.textContent = '▶ Iniciar';
      streamActive = false;
      stopDetections();
    }
  });

  document.getElementById('btn-capture').addEventListener('click', async () => {
    try {
      const r = await fetch('/api/capture');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.href = url;
      a.download = `esp_cam_${ts}.jpg`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 1000);
    } catch (e) {
      alert('Error capturando foto: ' + e.message);
    }
  });

  const thresholdSlider = document.getElementById('s-threshold');
  const thresholdLabel = document.getElementById('v-threshold');
  thresholdSlider.addEventListener('input', () => {
    thresholdLabel.textContent = thresholdSlider.value;
    clearTimeout(window.thresholdDebounce);
    window.thresholdDebounce = setTimeout(() => {
      fetch('/api/threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: parseFloat(thresholdSlider.value) })
      });
    }, 200);
  });

  const panelToggle = document.getElementById('cam-panel-toggle');
  const panel = document.getElementById('cam-panel');
  const panelClose = document.getElementById('panel-close');
  panelToggle.addEventListener('click', () => panel.classList.toggle('open'));
  panelClose.addEventListener('click', () => panel.classList.remove('open'));

  document.querySelectorAll('[data-cam]').forEach(el => {
    const eventType = el.type === 'range' ? 'input' : 'change';
    el.addEventListener(eventType, () => updateCam(el.dataset.cam, el));
  });

  async function updateCam(name, el) {
    const valEl = document.getElementById('v-' + name);
    if (valEl) {
      valEl.textContent = el.type === 'checkbox' ? (el.checked ? '✓' : '✗') : el.value + (name === 'flash_intensity' ? '%' : '');
    }
    clearTimeout(camDebounce);
    camDebounce = setTimeout(async () => {
      const val = el.type === 'checkbox' ? (el.checked ? 1 : 0) : parseInt(el.value);
      try {
        const r = await fetch('/api/camera', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ [name]: val })
        });
        if (!r.ok) console.warn('Error en /api/camera', r.status);
      } catch (e) {
        console.warn('cam ctrl error:', e);
      }
    }, 150);
  }

  async function updateMetrics() {
    try {
      const r = await fetch('/api/metrics');
      if (!r.ok) return;
      const d = await r.json();
      document.getElementById('m-ram').textContent = d.free_heap_kb + ' KB';
      document.getElementById('m-psram').textContent = d.free_psram_kb + ' KB';
      document.getElementById('m-mode').textContent = d.wifi_mode;
      document.getElementById('m-ip').textContent = d.ip_addr;
      document.getElementById('m-clients').textContent = d.ws_clients;
      const s = d.uptime_s;
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
      document.getElementById('m-up').textContent =
        `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
    } catch (e) { /* silencioso */ }
  }

  updateMetrics();
  setInterval(updateMetrics, 3000);

  // ---- Polling de detecciones ----
  let currentVideoWidth = 640;
let currentVideoHeight = 480;

// 1. Redimensionar y posicionar el Canvas exactamente sobre el video real [cite: 366]
function resizeCanvas() {
  const rect = camView.getBoundingClientRect();
  
  // 🟢 SOLUCIÓN DE ANCLAJE: Forzamos al contenedor padre a ser el marco de referencia
  if (camView.parentElement) {
    camView.parentElement.style.position = 'relative';
    camView.parentElement.style.display = 'inline-block'; // Se encoge para abrazar exactamente la imagen
  }
  
  // Ajustamos la resolución de dibujo interno del canvas para mapear 1:1 con el tamaño mostrado [cite: 366]
  overlayCanvas.width = rect.width;
  overlayCanvas.height = rect.height;
  
  // Posicionamos el canvas en el origen exacto (0,0) del contenedor relativo [cite: 366]
  overlayCanvas.style.position = 'absolute';
  overlayCanvas.style.left = '0px';
  overlayCanvas.style.top = '0px';
  overlayCanvas.style.width = '100%';
  overlayCanvas.style.height = '100%';
  
  // 🟢 SÚPER IMPORTANTE: Evita que el canvas intercepte clics (permite pulsar botones detrás de él)
  overlayCanvas.style.pointerEvents = 'none'; 
  
  console.log('📐 Canvas alineado y redimensionado perfectamente:', rect.width, 'x', rect.height);
}

window.addEventListener('resize', resizeCanvas);
camView.addEventListener('load', () => {
  resizeCanvas();
});

// ============================================================================
// 2. Detecciones de POSE: esqueleto de keypoints
// ============================================================================
//
// El ESP32 devuelve, por cada deteccion, la caja Y los 6 keypoints de la mano
// como [x, y, visibilidad]. Aqui se dibuja el esqueleto, no una caja.
//
// 6 KEYPOINTS DE MANO: muneca + punta de cada dedo.
// El conteo sale del modelo: kpt_shape = (6, 3) -> nk = 18 canales por nivel.
// No son los 21 landmarks de MediaPipe ni los 17 de cuerpo de COCO.
//
//   0 muneca          3 punta_medio
//   1 punta_pulgar    4 punta_anular
//   2 punta_indice    5 punta_menique
//
// ESTE ORDEN TIENE QUE COINCIDIR con el del .txt del dataset. Si tu
// anotacion lista los puntos en otro orden, el abanico se dibuja cruzado:
// se reconoce enseguida porque las lineas se salen de la mano. Se arregla
// cambiando SKELETON y KPT_COLORS aqui, y KPT_NAMES/KPT_SKELETON en la
// celda 2 del cuaderno. El numero de puntos NO hay que tocarlo: la celda 5
// lo lee del modelo.
//
// DIBUJAR UN ESQUELETO ES MAS EXIGENTE QUE DIBUJAR UNA CAJA:
//
//   - Una caja es convexa: cualquier par de puntos cae dentro y el rectangulo
//     siempre se ve bien. Un esqueleto NO: un keypoint mal decodificado se
//     traduce en una linea que atraviesa la mano o sale de la imagen.
//   - Por eso NO se dibuja un segmento si alguno de sus dos extremos esta por
//     debajo del umbral de visibilidad. Un keypoint no visto puede estar en
//     cualquier sitio, y "unirlo" produce el efecto de antena que delata que
//     el modelo no vio nada. Mejor que se note el hueco.
const KPT_VIS_TH = 0.5;   // debe coincidir con KPT_VIS_TH del cuaderno
const NUM_KPT = 6;

// El esqueleto es un abanico: la muneca es el centro y de ella cuelgan las cinco
// puntas. Coincide con KPT_SKELETON de la celda 2 del cuaderno.
const SKELETON = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
];

// Un color por punto: la muneca en cian, y cada dedo en un tono distinto para
// que se distinguan sin tener que leer la etiqueta. Coincide con KPT_COLORS
// de la celda 2.
const KPT_COLORS = [
  '#00e5ff',
  '#ffd400', '#ff8c00', '#00ff88', '#c56bff', '#8f5bff',
];

const KPT_NAMES = [
  'muneca',
  'pulgar', 'indice', 'medio', 'anular', 'menique',
];

// Intervalo de sondeo de /api/detections.
//
// SUBIDO de 500 a 3800 ms para igualarlo al periodo de inferencia medido
// (Inf 3.4 s + Pre + JPEG = ~3.7 s). Con 500 ms la UI preguntaba 7 veces por
// cada inferencia y recibia 7 veces el MISMO resultado: 6 peticiones
// inutiles por frame, sin ganar nada de latencia.
//
// Esto NO reduce la latencia percibida. El overlay siempre va a tener hasta un
// ciclo de inferencia de retraso porque la inferencia es de 3.4 s: la deteccion
// que se ve es la de hace 3-4 segundos, no la del instante. Es una limitacion
// del modelo en este microcontrolador, no del polling.
const POLL_MS = 3800;

// Dibuja un esqueleto. Devuelve cuantos keypoints se pintaron.
function drawSkeleton(det, view) {
  const kpts = det.kpts;
  if (!Array.isArray(kpts) || kpts.length === 0) return 0;

  // Proyectar a pixeles de pantalla una sola vez.
  const px = new Array(kpts.length);
  const py = new Array(kpts.length);
  const vis = new Array(kpts.length);
  let visibleCount = 0;
  for (let i = 0; i < kpts.length; i++) {
    const k = kpts[i];
    px[i] = view.offsetX + k[0] * view.scaleX;
    py[i] = view.offsetY + k[1] * view.scaleY;
    vis[i] = (typeof k[2] === 'number') ? k[2] : 0;
    if (vis[i] >= KPT_VIS_TH) visibleCount++;
  }

  // 1) Segmentos. Solo si AMBOS extremos son visibles.
  ctx.lineCap = 'round';
  ctx.lineWidth = 4;
  for (const [a, b] of SKELETON) {
    if (a >= px.length || b >= px.length) continue;
    if (vis[a] < KPT_VIS_TH || vis[b] < KPT_VIS_TH) continue;
    const cx = (px[a] + px[b]) / 2;
    ctx.strokeStyle = KPT_COLORS[a] || KPT_COLORS[0];
    ctx.beginPath();
    ctx.moveTo(px[a], py[a]);
    ctx.lineTo(px[b], py[b]);
    ctx.stroke();
  }

  // 2) Puntos. Radio segun cuan visible es, para que un keypoint al limite
  //    se vea como "poco fiable" y no como igual de firme que el resto.
  for (let i = 0; i < px.length; i++) {
    if (vis[i] < KPT_VIS_TH) continue;
    const t = Math.min(1, (vis[i] - KPT_VIS_TH) / (1 - KPT_VIS_TH));
    const r = 3 + 2 * t;
    ctx.fillStyle = KPT_COLORS[i] || KPT_COLORS[0];
    ctx.beginPath();
    ctx.arc(px[i], py[i], r, 0, Math.PI * 2);
    ctx.fill();
    // Contorno oscuro para que el punto se lea sobre cualquier fondo.
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.stroke();
  }

  // 3) Caja de la deteccion. Con este modelo los keypoints llegan a 3 de 6
  // visibles, y entonces el abanico queda ralo: la caja es la referencia de que
  // HAY una deteccion aunque el esqueleto no se complete. Se sube la opacidad
  // porque al 35% sobre video era practicamente invisible.
  if (det.box) {
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.75)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(
      view.offsetX + det.box[0] * view.scaleX,
      view.offsetY + det.box[1] * view.scaleY,
      det.box[2] * view.scaleX,
      det.box[3] * view.scaleY
    );
    ctx.setLineDash([]);
  }

  return visibleCount;
}

// Escala del lienzo respecto a la imagen, saltando las barras negras de
// letterbox/pillarbox.
function computeViewTransform() {
  const imgRect = camView.getBoundingClientRect();
  const containerWidth = imgRect.width || 1;
  const containerHeight = imgRect.height || 1;
  const imgRatio = currentVideoWidth / currentVideoHeight;
  const containerRatio = containerWidth / containerHeight;

  let renderWidth, renderHeight, offsetX = 0, offsetY = 0;
  if (containerRatio > imgRatio) {
    renderHeight = containerHeight;
    renderWidth = renderHeight * imgRatio;
    offsetX = (containerWidth - renderWidth) / 2;
  } else {
    renderWidth = containerWidth;
    renderHeight = renderWidth / imgRatio;
    offsetY = (containerHeight - renderHeight) / 2;
  }
  return {
    offsetX,
    offsetY,
    scaleX: renderWidth / currentVideoWidth,
    scaleY: renderHeight / currentVideoHeight,
  };
}

async function fetchAndDrawDetections() {
  if (!streamActive) return;
  try {
    const response = await fetch('/api/detections', { cache: 'no-store' });
    if (!response.ok) return;
    const data = await response.json();
    const dets = data.detections || [];

    const origWidth = data.orig_width || 640;
    const origHeight = data.orig_height || 480;
    if (origWidth !== currentVideoWidth || origHeight !== currentVideoHeight) {
      currentVideoWidth = origWidth;
      currentVideoHeight = origHeight;
      resizeCanvas();
    }

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (dets.length === 0) return;

    const view = computeViewTransform();

    for (const det of dets) {
      const drawn = drawSkeleton(det, view);

      // Etiqueta: clase, score y cuantos keypoints se vieron de verdad. Un
      // "0/17 visibles" con un score alto es la senal de que el threshold de
      // visibilidad esta demasiado alto o de que la persona esta de perfil.
      const scorePct = (det.score * 100).toFixed(0);
      const label = `${det.class_name || 'mano'} ${scorePct}% ${drawn}/${NUM_KPT}`;
      ctx.font = 'bold 12px sans-serif';
      const textW = ctx.measureText(label).width;
      const bx = view.offsetX + det.box[0] * view.scaleX;
      const by = view.offsetY + det.box[1] * view.scaleY;
      const labelH = 18;
      const labelY = by - labelH >= 0 ? by - labelH : by;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(bx - 1.5, labelY, textW + 10, labelH);
      ctx.fillStyle = drawn > 0 ? '#00ff88' : '#ff6b6b';
      ctx.fillText(label, bx + 3, labelY + 13);
    }
  } catch (e) {
    console.error('Error en fetchAndDrawDetections:', e);
  }
}

let pollTimer = null;

function startDetections() {
  if (pollTimer !== null) return;
  pollTimer = setInterval(fetchAndDrawDetections, POLL_MS);
  fetchAndDrawDetections();
}

function stopDetections() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}
}   // fin de initCameraPage()

function initConfigPage() {
  const useDhcp = document.getElementById('use-dhcp');
  const form = document.getElementById('wifi-form');
  const statusMsg = document.getElementById('status-msg');

  function toggleStaticIP() {
    const sec = document.getElementById('static-ip-section');
    useDhcp.checked ? sec.classList.remove('visible') : sec.classList.add('visible');
  }

  useDhcp.addEventListener('change', toggleStaticIP);
  form.addEventListener('submit', saveWifi);
  toggleStaticIP();

  function parseIP(str) {
    return (str || '0.0.0.0').split('.').map(Number);
  }

  async function saveWifi(e) {
    e.preventDefault();
    const payload = {
      ssid: document.getElementById('ssid').value.trim(),
      password: document.getElementById('password').value,
      use_dhcp: useDhcp.checked,
      static_ip: parseIP(document.getElementById('static-ip').value),
      static_gw: parseIP(document.getElementById('static-gw').value),
      static_nm: parseIP(document.getElementById('static-nm').value),
    };

    statusMsg.style.display = 'block';
    statusMsg.className = '';
    statusMsg.textContent = 'Guardando...';

    try {
      const r = await fetch('/api/wifi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (r.ok) {
        statusMsg.className = 'ok';
        statusMsg.textContent = '✓ Guardado! Puedes reiniciar manualmente el equipo.';
      } else {
        throw new Error('HTTP ' + r.status);
      }
    } catch (err) {
      statusMsg.className = 'err';
      statusMsg.textContent = '✗ Error: ' + err.message;
    }
  }
}