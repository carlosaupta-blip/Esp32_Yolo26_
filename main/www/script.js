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
    } else {
      camView.src = '';
      camView.style.display = 'none';
      noSignal.style.display = 'block';
      noSignal.innerHTML = '<span>📷</span>Stream detenido';
      btnStream.textContent = '▶ Iniciar';
      streamActive = false;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
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

// 2. Polling de detecciones con compensación de relación de aspecto (Letterbox/Pillarbox)
async function fetchAndDrawDetections() {
  if (!streamActive) {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    return;
  }
  try {
    const response = await fetch('/api/detections');
    if (!response.ok) return;
    const data = await response.json();
    const dets = data.detections || [];
    
    // Actualizar resoluciones del modelo dinámicamente según el JSON [cite: 381]
    const origWidth = data.orig_width || 640;
    const origHeight = data.orig_height || 480;

    // Si la resolución cambió, reajustamos el lienzo
    if (origWidth !== currentVideoWidth || origHeight !== currentVideoHeight) {
      currentVideoWidth = origWidth;
      currentVideoHeight = origHeight;
      resizeCanvas();
    }
    
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (dets.length === 0) {
      return;
    }

    const imgRect = camView.getBoundingClientRect();
    const containerWidth = imgRect.width;
    const containerHeight = imgRect.height;

    // --- CÁLCULO DE COMPENSACIÓN DE ASPECT RATIO (Mapeo Inteligente) ---
    const imgRatio = origWidth / origHeight;
    const containerRatio = containerWidth / containerHeight;
    
    let renderWidth, renderHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (containerRatio > imgRatio) {
      // Caso Pillarbox: Barras negras a la izquierda y derecha
      renderHeight = containerHeight;
      renderWidth = renderHeight * imgRatio;
      offsetX = (containerWidth - renderWidth) / 2;
    } else {
      // Caso Letterbox: Barras negras arriba y abajo
      renderWidth = containerWidth;
      renderHeight = renderWidth / imgRatio;
      offsetY = (containerHeight - renderHeight) / 2;
    }

    // Escalas calculadas sobre la imagen real proyectada (sin las barras negras)
    const scaleX = renderWidth / origWidth;
    const scaleY = renderHeight / origHeight;

    // Dibujar cada detección proyectada
    dets.forEach(det => {
      // Sumamos los offsets para saltar las barras negras y centrar la caja [cite: 381]
      const x = offsetX + (det.x * scaleX);
      const y = offsetY + (det.y * scaleY);
      const w = det.w * scaleX;
      const h = det.h * scaleY;

      if (w <= 0 || h <= 0 || isNaN(w) || isNaN(h)) {
        return;
      }

      // 1. Dibujar Bounding Box [cite: 381]
      ctx.strokeStyle = '#00ff00'; // Verde brillante
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // 2. Dibujar Etiqueta con Score [cite: 381]
      const scorePercent = (det.score * 100).toFixed(0);
      const labelText = `Rostro: ${scorePercent}%`;
      
      ctx.font = 'bold 12px sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const labelHeight = 18;

      // Fondo de la etiqueta [cite: 381]
      ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
      ctx.fillRect(x - 1.5, y - labelHeight, textWidth + 10, labelHeight);

      // Texto de la etiqueta [cite: 381]
      ctx.fillStyle = '#000000';
      ctx.fillText(labelText, x + 3, y - 5);
    });

  } catch (e) {
    console.error('❌ Error en fetchAndDrawDetections:', e);
  }
}

  setInterval(fetchAndDrawDetections, 500);
}

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