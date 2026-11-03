/* Interactive ASCII water background.
 *
 * The mood (clean / slightly polluted / polluted) comes from the live WQI:
 * app.js calls WaterBG.setScore(averageWQI, scopeLabel) whenever live data loads
 * or the State filter changes. DOE classes: Clean >= 81, Slightly polluted 60-80,
 * Polluted < 60.
 *
 * - Pure canvas, no libraries. Sits behind the page (pointer-events: none) and listens
 *   to the pointer on the window, so it never blocks clicking the dashboard.
 * - Respects "reduce motion" (draws one still frame) and pauses when the tab is hidden.
 * - The on/off choice is remembered in localStorage when available.
 */
(function () {
  'use strict';
  var canvas = document.getElementById('water-bg');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ---------------------------------------------------------------- moods
  var MOODS = {
    clean:    { rgb: [14, 116, 190], tint: 0.06, amp: 1.0,  noise: 0,    fish: 9, bubbles: 10, debris: 0,  speed: 60, name: 'Clean' },
    slight:   { rgb: [190, 140, 8],  tint: 0.09, amp: 1.15, noise: 0.10, fish: 4, bubbles: 0,  debris: 24, speed: 32, name: 'Slightly polluted' },
    polluted: { rgb: [185, 35, 35],  tint: 0.12, amp: 1.3,  noise: 0.28, fish: 1, bubbles: 0,  debris: 85, speed: 20, name: 'Polluted' },
    none:     { rgb: [100, 116, 139], tint: 0.04, amp: 0.8, noise: 0,    fish: 0, bubbles: 0,  debris: 0,  speed: 40, name: 'waiting for data' }
  };
  var DEBRIS_CHARS = { slight: ['.', ',', '*', '`', "'"], polluted: ['%', '#', 'x', '*', ';', ':', '~', ','] };
  var BUBBLE_CHARS = ['o', '°', '.'];
  var FISH_R = ['><>', '><(((°>'];
  var FISH_L = ['<><', '<°)))><'];
  var WAVE_CHARS = ['.', '~', '≈'];

  // ---------------------------------------------------------------- state
  var enabled = true;
  try { enabled = localStorage.getItem('waterbg:on') !== '0'; } catch (e) { /* storage blocked */ }
  var preview = 'live', liveScore = null, scope = '';
  var cur = { r: 100, g: 116, b: 139, tint: 0.04, noise: 0, amp: 0.8 };
  var w = 0, h = 0, dpr = 1, cw = 8, lh = 20, cols = 0, rows = 0, small = false;
  var fish = [], bubbles = [], debris = [], ripples = [];
  var ptr = null, raf = 0, last = 0, t = 0, running = false;
  var lv = [[], [], []];

  function moodOf(score) { return score == null ? 'none' : score >= 81 ? 'clean' : score >= 60 ? 'slight' : 'polluted'; }
  function mood() { return preview !== 'live' ? preview : moodOf(liveScore); }
  function rnd(a, b) { return a + Math.random() * (b - a); }
  function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }
  function hash(a, b, c) { var s = Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453; return s - Math.floor(s); }
  function rgba(r, g, b, a) { return 'rgba(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ',' + a + ')'; }
  function ptrActive() { return ptr && performance.now() - ptr.t < 2500; }

  // ------------------------------------------------------------- entities
  function newFish(m) {
    var big = Math.random() < 0.4;
    return { x: rnd(0, w), y: rnd(70, Math.max(120, h - 70)), vx: (Math.random() < 0.5 ? -1 : 1) * rnd(0.6, 1.2) * m.speed, vy: 0,
             big: big, face: 1, ph: rnd(0, 6.28), sp: rnd(0.7, 1.3) };
  }
  function newBubble() { return { x: rnd(0, w), y: h + rnd(0, 80), vx: 0, vy: -rnd(18, 40), ch: pick(BUBBLE_CHARS), ph: rnd(0, 6.28) }; }
  function newDebris(name) { var set = DEBRIS_CHARS[name] || DEBRIS_CHARS.slight;
    return { x: rnd(0, w), y: rnd(0, h), vx: rnd(-6, 6), vy: rnd(-2, 2), ch: pick(set), ph: rnd(0, 6.28) }; }

  function syncEntities(m, name) {
    var k = small ? 0.6 : 1;
    var wantFish = Math.round(m.fish * k), wantB = Math.round(m.bubbles * k), wantD = Math.round(m.debris * k);
    if (fish.length < wantFish) fish.push(newFish(m)); else if (fish.length > wantFish) fish.pop();
    if (bubbles.length < wantB) bubbles.push(newBubble()); else if (bubbles.length > wantB) bubbles.pop();
    if (debris.length < wantD) debris.push(newDebris(name)); else if (debris.length > wantD) debris.pop();
  }

  function push(e, fx, fy, R, power, dt) {
    var dx = e.x - fx, dy = e.y - fy, d = Math.sqrt(dx * dx + dy * dy) || 1;
    if (d < R) { var f = (1 - d / R) * power * dt; e.vx += dx / d * f; e.vy += dy / d * f; }
  }

  // --------------------------------------------------------------- update
  function update(dt) {
    var name = mood(), m = MOODS[name];
    var k = 1 - Math.exp(-dt * 2.5);
    cur.r += (m.rgb[0] - cur.r) * k; cur.g += (m.rgb[1] - cur.g) * k; cur.b += (m.rgb[2] - cur.b) * k;
    cur.tint += (m.tint - cur.tint) * k; cur.noise += (m.noise - cur.noise) * k; cur.amp += (m.amp - cur.amp) * k;
    syncEntities(m, name);

    var pOn = !reduceMotion && ptrActive();
    var i, e, j;
    for (i = ripples.length - 1; i >= 0; i--) { ripples[i].a += dt; if (ripples[i].a > 1.8) ripples.splice(i, 1); }

    function ripplePush(ent, power) {
      for (var q = 0; q < ripples.length; q++) {
        var rp = ripples[q], dx = ent.x - rp.x, dy = ent.y - rp.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
        if (Math.abs(d - rp.a * 260) < 40) { var s = (1 - rp.a / 1.8) * power * dt; ent.vx += dx / d * s; ent.vy += dy / d * s; }
      }
    }

    for (i = 0; i < fish.length; i++) {
      e = fish[i];
      var dir = e.vx >= 0 ? 1 : -1, sp = m.speed * e.sp;
      var wantVx = Math.abs(e.vx) < 6 ? dir * sp : (e.vx > 0 ? sp : -sp);
      e.ph += dt * 1.5;
      var wantVy = Math.sin(e.ph) * 9;
      var ease = Math.min(1, dt * 1.4);
      e.vx += (wantVx - e.vx) * ease; e.vy += (wantVy - e.vy) * ease;
      if (pOn) {
        var dx = ptr.x - e.x, dy = ptr.y - e.y, d = Math.sqrt(dx * dx + dy * dy) || 1;
        if (name === 'clean' && d > 70 && d < 260) { e.vx += dx / d * 55 * dt; e.vy += dy / d * 55 * dt; }  // curious
        else if (d < (name === 'clean' ? 70 : 150)) push(e, ptr.x, ptr.y, 150, 700, dt);                    // startled
      }
      ripplePush(e, 420);
      e.x += e.vx * dt; e.y += e.vy * dt;
      if (e.y < 50) e.vy += 30 * dt; if (e.y > h - 50) e.vy -= 30 * dt;
      if (e.x > w + 70) { e.x = -70; e.y = rnd(70, Math.max(120, h - 70)); }
      if (e.x < -70) { e.x = w + 70; e.y = rnd(70, Math.max(120, h - 70)); }
      if (Math.abs(e.vx) > 8) e.face = e.vx > 0 ? 1 : -1;
    }
    for (i = 0; i < bubbles.length; i++) {
      e = bubbles[i]; e.ph += dt * 2;
      if (pOn) push(e, ptr.x, ptr.y, 90, 300, dt);
      ripplePush(e, 300);
      e.vx *= 0.97; e.x += (e.vx + Math.sin(e.ph) * 6) * dt; e.y += e.vy * dt;
      if (e.y < -20) { e.y = h + 20; e.x = rnd(0, w); e.vx = 0; }
    }
    for (i = 0; i < debris.length; i++) {
      e = debris[i]; e.ph += dt;
      if (pOn) push(e, ptr.x, ptr.y, 100, 450, dt);
      ripplePush(e, 450);
      e.vx *= 0.985; e.vy *= 0.985;
      e.x += (e.vx + 5 + Math.sin(e.ph) * 4) * dt; e.y += (e.vy + Math.cos(e.ph * 0.8) * 3) * dt;
      if (e.x > w + 20) e.x = -20; if (e.x < -20) e.x = w + 20;
      if (e.y > h + 20) e.y = -20; if (e.y < -20) e.y = h + 20;
    }
  }

  // ----------------------------------------------------------------- draw
  function draw() {
    var r = cur.r, g = cur.g, b = cur.b;
    ctx.clearRect(0, 0, w, h);
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, rgba(r, g, b, cur.tint * 0.6));
    grad.addColorStop(1, rgba(r, g, b, cur.tint * 1.7));
    ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);

    ctx.font = (small ? 13 : 14) + 'px Consolas, Menlo, "DejaVu Sans Mono", monospace';
    ctx.textBaseline = 'alphabetic';
    lv[0].length = 0; lv[1].length = 0; lv[2].length = 0;
    var tt = t * cur.amp, tick = Math.floor(t * 3);
    var pOn = !reduceMotion && ptrActive(), nRip = ripples.length;
    for (var row = 0; row < rows; row++) {
      var y = row * lh + lh * 0.8;
      for (var c = 0; c < cols; c++) {
        var x = c * cw;
        var v = Math.sin(c * 0.22 + tt * 1.1) + Math.sin(row * 0.45 - tt * 0.7 + c * 0.08) + Math.sin((c + row) * 0.12 + tt * 0.5);
        var n = (v + 3) / 6;
        if (cur.noise > 0.01) n += (hash(c, row, tick) - 0.5) * cur.noise;
        var boost = 0;
        if (pOn) { var px = x - ptr.x, py = y - ptr.y, d2 = px * px + py * py; if (d2 < 10000) boost += 1.2 * (1 - Math.sqrt(d2) / 100); }
        for (var q = 0; q < nRip; q++) {
          var rp = ripples[q], rx = x - rp.x, ry = y - rp.y, rd = Math.abs(Math.sqrt(rx * rx + ry * ry) - rp.a * 260);
          if (rd < 45) boost += 2.2 * (1 - rd / 45) * (1 - rp.a / 1.8);
        }
        var level = n > 0.80 ? 3 : n > 0.62 ? 2 : n > 0.46 ? 1 : 0;
        if (boost > 0) level = Math.min(3, level + Math.round(boost));
        if (level) { var arr = lv[level - 1]; arr.push(x, y); }
      }
    }
    var alphas = [0.28, 0.40, 0.55];
    ctx.fillStyle = rgba(r, g, b, 1);
    for (var L = 0; L < 3; L++) {
      ctx.globalAlpha = alphas[L];
      var ch = WAVE_CHARS[L], arr2 = lv[L];
      for (var k = 0; k < arr2.length; k += 2) ctx.fillText(ch, arr2[k], arr2[k + 1]);
    }

    var name = mood(), i, e;
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = rgba(r * 0.55, g * 0.5, b * 0.45, 1);
    for (i = 0; i < debris.length; i++) { e = debris[i]; ctx.fillText(e.ch, e.x, e.y); }
    ctx.globalAlpha = 0.5; ctx.fillStyle = rgba(r, g, b, 1);
    for (i = 0; i < bubbles.length; i++) { e = bubbles[i]; ctx.fillText(e.ch, e.x, e.y); }
    ctx.globalAlpha = name === 'polluted' ? 0.6 : 0.9;
    ctx.fillStyle = rgba(r * 0.7, g * 0.7, b * 0.7, 1);
    for (i = 0; i < fish.length; i++) {
      e = fish[i]; var s = (e.face > 0 ? FISH_R : FISH_L)[e.big ? 1 : 0];
      ctx.font = 'bold ' + (e.big ? (small ? 20 : 26) : (small ? 15 : 19)) + 'px Consolas, Menlo, "DejaVu Sans Mono", monospace';
      ctx.fillText(s, e.x - ctx.measureText(s).width / 2, e.y);
    }
    ctx.globalAlpha = 1;
  }

  // ----------------------------------------------------------------- loop
  function frame(ts) {
    if (!running) return;
    raf = requestAnimationFrame(frame);
    if (ts - last < (small ? 1000 / 24 : 1000 / 30)) return;
    var dt = Math.min((ts - last) / 1000, 0.05) || 0.033;
    last = ts; t += dt; update(dt); draw();
  }
  function start() {
    if (running || !enabled || reduceMotion || document.hidden) return;
    running = true; last = performance.now(); raf = requestAnimationFrame(frame);
  }
  function stop() { running = false; cancelAnimationFrame(raf); }
  function stillFrame() {            // used for "reduce motion": snap to the mood, draw once
    var m = MOODS[mood()];
    cur.r = m.rgb[0]; cur.g = m.rgb[1]; cur.b = m.rgb[2]; cur.tint = m.tint; cur.noise = m.noise; cur.amp = m.amp;
    for (var i = 0; i < 80; i++) update(0.05);
    draw();
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2); w = window.innerWidth; h = window.innerHeight; small = w < 700;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = (small ? 13 : 14) + 'px Consolas, Menlo, "DejaVu Sans Mono", monospace';
    cw = ctx.measureText('M').width || 8; lh = small ? 22 : 20;
    cols = Math.ceil(w / cw) + 1; rows = Math.ceil(h / lh) + 1;
    [fish, bubbles, debris].forEach(function (list) { list.forEach(function (e) { e.x = Math.min(e.x, w); e.y = Math.min(e.y, h); }); });
    if (reduceMotion && enabled) stillFrame();
  }

  // ------------------------------------------------------------------- UI
  var elMood = document.getElementById('bg-mood');
  var elPrev = document.getElementById('bg-preview');
  var elTog = document.getElementById('bg-toggle');
  function refreshLabel() {
    var name = mood(), m = MOODS[name];
    if (elMood) {
      elMood.textContent = preview !== 'live' ? 'Water mood (preview): ' + m.name
        : name === 'none' ? 'Water mood: waiting for live data'
        : 'Water mood: ' + m.name + ' · avg WQI ' + Math.round(liveScore) + (scope ? ' · ' + scope : '');
    }
    if (reduceMotion && enabled) stillFrame();
  }
  function applyEnabled() {
    canvas.style.display = enabled ? '' : 'none';
    if (elTog) { elTog.textContent = 'Water: ' + (enabled ? 'on' : 'off'); elTog.setAttribute('aria-pressed', String(enabled)); }
    if (enabled) { if (reduceMotion) stillFrame(); else start(); } else stop();
  }
  if (elPrev) elPrev.addEventListener('change', function () { preview = elPrev.value; refreshLabel(); });
  if (elTog) elTog.addEventListener('click', function () {
    enabled = !enabled;
    try { localStorage.setItem('waterbg:on', enabled ? '1' : '0'); } catch (e) { /* ignore */ }
    applyEnabled();
  });

  // ---------------------------------------------------------- interaction
  if (!reduceMotion) {
    window.addEventListener('pointermove', function (ev) { ptr = { x: ev.clientX, y: ev.clientY, t: performance.now() }; }, { passive: true });
    window.addEventListener('pointerdown', function (ev) {
      if (!enabled || (ev.target && ev.target.closest && ev.target.closest('#bg-controls'))) return;
      ptr = { x: ev.clientX, y: ev.clientY, t: performance.now() };
      if (ripples.length < 6) ripples.push({ x: ev.clientX, y: ev.clientY, a: 0 });
    }, { passive: true });
    document.documentElement.addEventListener('pointerleave', function () { ptr = null; });
  }
  document.addEventListener('visibilitychange', function () { if (document.hidden) stop(); else if (enabled) start(); });
  var rz = 0;
  window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(resize, 150); });

  // ----------------------------------------------------------- public API
  window.WaterBG = {
    setScore: function (score, label) { liveScore = (score == null || isNaN(score)) ? null : Number(score); scope = label || ''; refreshLabel(); },
    setPreview: function (v) { preview = v || 'live'; if (elPrev) elPrev.value = preview; refreshLabel(); },
    state: function () { return { mood: mood(), fish: fish.length, bubbles: bubbles.length, debris: debris.length, running: running, enabled: enabled }; }
  };

  resize(); refreshLabel(); applyEnabled();
})();
