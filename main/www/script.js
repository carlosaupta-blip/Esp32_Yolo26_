document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('cam-view')) initCameraPage();
  if (document.getElementById('wifi-form')) initConfigPage();
});

/* ========== PÁGINA DE CÁMARA ========== */
function initCameraPage() {
  const camView = document.getElementById('cam-view');
  const noSignal = document.getElementById('no-signal');
  const btnStream = document.getElementById('btn-stream');
  let streamActive = false;
  let camDebounce = null;

  /* ---------- Botón Iniciar/Detener stream (MJPEG) ---------- */
  btnStream.addEventListener('click', () => {
    if (!streamActive) {
      // Iniciar streaming: apuntar la imagen al endpoint MJPEG (puerto 81)
      camView.src = `http://${location.hostname}:81/stream`;
      camView.style.display = 'block';
      noSignal.style.display = 'none';
      btnStream.textContent = '⏸ Detener';
      streamActive = true;
    } else {
      // Detener streaming: limpiar src para cortar la conexión
      camView.src = '';
      camView.style.display = 'none';
      noSignal.style.display = 'block';
      noSignal.innerHTML = '<span>📷</span>Stream detenido';
      btnStream.textContent = '▶ Iniciar';
      streamActive = false;
    }
  });

  /* ---------- Captura de foto ---------- */
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

  /* ---------- Panel lateral ---------- */
  const panelToggle = document.getElementById('cam-panel-toggle');
  const panel = document.getElementById('cam-panel');
  const panelClose = document.getElementById('panel-close');
  panelToggle.addEventListener('click', () => panel.classList.toggle('open'));
  panelClose.addEventListener('click', () => panel.classList.remove('open'));

  /* ---------- Controles de cámara (data-cam) ---------- */
  document.querySelectorAll('[data-cam]').forEach(el => {
    const eventType = el.type === 'range' ? 'input' : 'change';
    el.addEventListener(eventType, () => updateCam(el.dataset.cam, el));
  });

  async function updateCam(name, el) {
    // Actualizar label si existe
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

  /* ---------- Métricas periódicas ---------- */
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
}

/* ========== PÁGINA DE CONFIGURACIÓN ========== */
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
  toggleStaticIP(); // estado inicial

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