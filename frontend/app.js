const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const phoneIpInput = document.getElementById('phone-ip');
const badge = document.getElementById('badge');

const statCount = document.getElementById('stat-count');
const statConfidence = document.getElementById('stat-confidence');
const statFps = document.getElementById('stat-fps');
const statStatus = document.getElementById('stat-status');
const statEmails = document.getElementById('stat-emails');

let ipFieldTouched = false;
phoneIpInput.addEventListener('input', () => { ipFieldTouched = true; });

async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();

    statCount.textContent = data.stable_count;
    statConfidence.textContent = (data.avg_confidence * 100).toFixed(1) + '%';
    statFps.textContent = data.fps;
    statStatus.textContent = data.status;
    statEmails.textContent = data.emails_sent_today;

    if (data.running && data.camera_connected) {
      badge.textContent = 'Live';
      badge.className = 'badge live';
    } else if (data.error) {
      badge.textContent = data.error;
      badge.className = 'badge error';
    } else {
      badge.textContent = 'Stopped';
      badge.className = 'badge stopped';
    }

    if (data.phone_ip && !ipFieldTouched && !phoneIpInput.value) {
      phoneIpInput.value = data.phone_ip;
    }
  } catch (e) {
    badge.textContent = 'Server unreachable';
    badge.className = 'badge error';
  }
}

async function refreshHistory() {
  try {
    const res = await fetch('/api/detections?limit=20');
    const rows = await res.json();
    const tbody = document.querySelector('#history-table tbody');
    tbody.innerHTML = '';

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" style="color:#8CA0A8;">No detections logged yet.</td></tr>';
      return;
    }

    rows.forEach(r => {
      const tr = document.createElement('tr');
      const time = r.timestamp ? new Date(r.timestamp).toLocaleString() : '–';
      const conf = r.confidence_avg != null ? (r.confidence_avg * 100).toFixed(0) + '%' : '–';
      tr.innerHTML = `
        <td>${time}</td>
        <td>${r.location || '–'}</td>
        <td>${r.waste_count}</td>
        <td>${conf}</td>
        <td>${r.alert_sent ? '✅' : '—'}</td>
        <td>${r.image_url ? `<a href="${r.image_url}" target="_blank">view</a>` : '–'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    // keep last known table on transient errors
  }
}

startBtn.addEventListener('click', async () => {
  const phone_ip = phoneIpInput.value.trim();
  startBtn.disabled = true;
  try {
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone_ip: phone_ip || null })
    });
    const data = await res.json();
    if (!data.ok) alert(data.message);
  } catch (e) {
    alert('Could not reach the server.');
  } finally {
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener('click', async () => {
  await fetch('/api/stop', { method: 'POST' });
});

refreshStats();
refreshHistory();
setInterval(refreshStats, 2000);
setInterval(refreshHistory, 5000);
