/* DIP Lab Studio front end.
   Sidebar, controls and result blocks are generated from what the backend
   sends, so a new tool only needs new Python code. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const state = { image: null, groups: [], op: null, params: {}, seq: 0, timer: null, q: "" };
  const el = {
    nav: $("#nav"), title: $("#opTitle"), desc: $("#opDesc"), formula: $("#opFormula"),
    controls: $("#controls"), drop: $("#drop"), out: $("#out"), alert: $("#alert"), home: $("#home"),
    chip: $("#fileChip"), thumb: $("#fileThumb"), fname: $("#fileName"), fmeta: $("#fileMeta"),
    input: $("#fileInput"), veil: $("#veil"), rail: $("#rail"), print: $("#printBtn"), search: $("#search"),
  };

  const coarse = matchMedia("(pointer: coarse)").matches;
  const scrim = $("#scrim"), menuBtn = $("#menuBtn");
  function setRail(open) {
    el.rail.classList.toggle("open", open);
    scrim.classList.toggle("on", open);
    menuBtn.setAttribute("aria-expanded", String(open));
  }
  scrim.onclick = () => setRail(false);
  addEventListener("keydown", (e) => { if (e.key === "Escape") setRail(false); });
  matchMedia("(min-width: 901px)").addEventListener("change", () => setRail(false));

  /* ---------- sidebar ---------- */
  function buildNav() {
    const q = state.q.trim().toLowerCase();
    el.nav.innerHTML = "";
    state.groups.forEach((g) => {
      const items = g.items.filter((i) => !q || (i.title + " " + i.desc + " " + g.name).toLowerCase().includes(q));
      if (!items.length) return;
      const d = document.createElement("details");
      d.open = !!q || (state.op && items.some((i) => i.id === state.op.id)) || g === state.groups[0];
      d.innerHTML = `<summary><span>${esc(g.name)}</span><span class="tag">${esc(g.lab)}</span></summary>`;
      items.forEach((i) => {
        const b = document.createElement("button");
        b.className = "tool" + (state.op && state.op.id === i.id ? " on" : "");
        b.textContent = i.title;
        b.dataset.id = i.id;
        b.onclick = () => { selectOp(i.id); setRail(false); scrollTo(0, 0); };
        d.appendChild(b);
      });
      el.nav.appendChild(d);
    });
  }
  el.search.oninput = () => { state.q = el.search.value; buildNav(); };

  function findOp(id) {
    for (const g of state.groups) for (const i of g.items) if (i.id === id) return i;
  }

  /* ---------- overview ---------- */
  function showHome() {
    state.op = null; state.seq++;
    document.querySelectorAll(".tool").forEach((b) => b.classList.remove("on"));
    el.title.textContent = "DIP Lab Studio";
    el.desc.textContent = "Upload one image, then open any tool to see its result as pictures, graphs, numbers and matrices.";
    el.formula.hidden = true; el.controls.hidden = true; el.out.innerHTML = ""; el.print.hidden = true;
    const total = state.groups.reduce((n, g) => n + g.items.length, 0);
    el.home.hidden = false;
    el.home.innerHTML = `<p class="lead">${total} tools, grouped by topic. Pick one below or use the search box.</p>` +
      state.groups.map((g) => `<div class="home-group"><h3>${esc(g.name)} <small>${esc(g.lab)}</small></h3><div class="home-list">` +
        g.items.map((i) => `<button class="home-item" data-id="${i.id}"><b>${esc(i.title)}</b><span>${esc(i.desc)}</span></button>`).join("") +
        `</div></div>`).join("");
    el.home.querySelectorAll(".home-item").forEach((b) => b.onclick = () => selectOp(b.dataset.id));
    el.drop.hidden = !!state.image;
    scrollTo(0, 0);
  }
  $("#homeBtn").onclick = () => { showHome(); setRail(false); };

  function selectOp(id) {
    const o = findOp(id);
    if (!o) return;
    state.op = o;
    state.params = Object.fromEntries(o.params.map((p) => [p.name, p.default]));
    document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
    el.home.hidden = true;
    el.title.textContent = o.title;
    el.desc.textContent = o.desc;
    el.formula.hidden = !o.formula;
    el.formula.textContent = o.formula;
    el.out.innerHTML = "";
    buildControls(o);
    el.drop.hidden = !!state.image;
    run();
  }

  /* ---------- controls ---------- */
  const queue = (ms) => {
    clearTimeout(state.timer);
    state.timer = setTimeout(run, ms ?? (state.op && state.op.heavy ? 450 : 160));
  };

  function buildControls(o) {
    el.controls.innerHTML = "";
    el.controls.hidden = !o.params.length || !state.image;
    o.params.forEach((p) => {
      const w = document.createElement("label");
      w.className = "ctl";
      let input;
      if (p.type === "slider") {
        w.innerHTML = `<span>${esc(p.label)}</span><input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${state.params[p.name]}"><output>${state.params[p.name]}</output>`;
        input = $("input", w);
        input.oninput = () => { $("output", w).textContent = input.value; state.params[p.name] = +input.value; queue(); };
      } else if (p.type === "select") {
        w.innerHTML = `<span>${esc(p.label)}</span><select>${p.options.map(([v, l]) => `<option value="${esc(v)}" ${v === state.params[p.name] ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
        input = $("select", w);
        input.onchange = () => { state.params[p.name] = input.value; queue(0); };
      } else if (p.type === "checkbox") {
        w.innerHTML = `<input type="checkbox" ${state.params[p.name] ? "checked" : ""}><span>${esc(p.label)}</span>`;
        input = $("input", w);
        input.onchange = () => { state.params[p.name] = input.checked; queue(0); };
      } else if (p.type === "image") {
        w.innerHTML = `<span>${esc(p.label)}</span><span class="refpick"><img alt="" hidden><button type="button" class="ghost">${esc(p.button || "Choose image")}</button><small class="sub"></small></span><input type="file" accept="image/*" hidden>`;
        const f = $("input", w), btn = $("button", w), im = $("img", w), sub = $("small", w);
        btn.onclick = () => f.click();
        f.onchange = async () => {
          if (!f.files[0]) return;
          const fd = new FormData(); fd.append("image", f.files[0]);
          try {
            const r = await fetch("/api/upload", { method: "POST", body: fd });
            const j = await r.json();
            if (!r.ok) throw new Error(j.error || "Upload failed.");
            state.params[p.name] = j.id;
            im.src = URL.createObjectURL(f.files[0]); im.hidden = false;
            sub.textContent = j.name;
            queue(0);
          } catch (e) { showAlert(e.message); }
        };
      }
      el.controls.appendChild(w);
    });
  }

  /* ---------- upload ---------- */
  async function send(promise) {
    try {
      const r = await promise;
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Upload failed.");
      state.image = j;
      showFileChip();
      showAlert("");
      el.drop.hidden = true;
      if (state.op) { buildControls(state.op); run(); }
    } catch (e) { showAlert(e.message); }
  }
  const upload = (file) => {
    const fd = new FormData(); fd.append("image", file);
    state.thumbURL && URL.revokeObjectURL(state.thumbURL);
    state.thumbURL = URL.createObjectURL(file);
    send(fetch("/api/upload", { method: "POST", body: fd }));
  };
  function showFileChip() {
    const im = state.image;
    el.chip.hidden = false;
    el.fname.textContent = im.name;
    el.fmeta.textContent = `${im.width} x ${im.height}, ${im.kb} KB`;
    el.thumb.src = state.thumbURL || "";
    el.thumb.hidden = !state.thumbURL;
  }

  $("#pickBtn").onclick = $("#changeBtn").onclick = () => el.input.click();
  el.input.onchange = () => { if (el.input.files[0]) upload(el.input.files[0]); el.input.value = ""; };
  const cam = $("#camInput");
  $("#camBtn").onclick = () => cam.click();
  cam.onchange = () => { if (cam.files[0]) upload(cam.files[0]); cam.value = ""; };
  $("#sampleBtn").onclick = () => { state.thumbURL = null; send(fetch("/api/sample")); };
  menuBtn.onclick = () => setRail(!el.rail.classList.contains("open"));
  el.print.onclick = () => print();

  let dragDepth = 0;
  addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; el.veil.classList.add("on"); });
  addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; el.veil.classList.remove("on"); } });
  addEventListener("dragover", (e) => e.preventDefault());
  addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; el.veil.classList.remove("on");
    const f = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (f) upload(f); else showAlert("That file is not an image.");
  });
  addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
    if (f) upload(f);
  });

  function showAlert(msg) { el.alert.hidden = !msg; el.alert.textContent = msg; }

  /* ---------- run ---------- */
  async function run() {
    if (!state.op || !state.image) { el.out.innerHTML = ""; return; }
    const my = ++state.seq;
    el.out.classList.add("busy");
    try {
      const r = await fetch("/api/run", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: state.image.id, op: state.op.id, params: state.params }),
      });
      const j = await r.json();
      if (my !== state.seq) return;
      if (!r.ok) throw new Error(j.error || "Something went wrong.");
      showAlert("");
      render(j);
    } catch (e) {
      if (my === state.seq) { showAlert(e.message); el.out.innerHTML = ""; }
    } finally {
      if (my === state.seq) el.out.classList.remove("busy");
    }
  }

  /* ---------- render ---------- */
  function render(r) {
    let h = "";
    if (r.headline) h += `<div class="headline"><strong>${esc(r.headline.value)}</strong><span>${esc(r.headline.text)}</span></div>`;
    if (r.images.length) {
      h += `<div class="shots ${r.layout === "gallery" ? "gallery" : ""}">` + r.images.map((im, i) => `
        <figure class="shot" data-i="${i}">
          <div class="shot-img"><img alt="${esc(im.label)}" src="${im.src}"></div>
          <figcaption>
            <div class="cap-top"><b>${esc(im.label)}</b><a href="${im.src}" download="${esc(im.file)}.png">Save PNG</a></div>
            ${im.caption ? `<span class="sub">${esc(im.caption)}</span>` : ""}
            <div class="readout"></div>
          </figcaption>
        </figure>`).join("") + `</div>`;
    }
    if (r.stats.length) {
      h += `<div class="section"><div class="facts">` + r.stats.map((s) => `
        <div class="fact"><small>${esc(s.label)}</small><strong>${esc(s.value)}</strong>${s.hint ? `<em>${esc(s.hint)}</em>` : ""}</div>`).join("") + `</div></div>`;
    }
    if (r.graphs.length) h += `<div class="section graphs">` + r.graphs.map((g, i) => `<div class="graph" data-g="${i}"></div>`).join("") + `</div>`;
    if (r.palettes.length) {
      h += `<div class="section tables">` + r.palettes.map((p) => `
        <div class="pal"><h4>${esc(p.title)}</h4>${p.caption ? `<p>${esc(p.caption)}</p>` : ""}
          <div class="swatches">${p.colors.map((c, i) => `<div class="sw"><i style="background:rgb(${c})"></i><small>${p.values && p.values[i] != null ? esc(p.values[i]) + "<br>" : ""}${c.join(", ")}</small></div>`).join("")}</div></div>`).join("") + `</div>`;
    }
    if (r.charts.length) h += `<div class="section charts">` + r.charts.map((c, i) => `<div class="chart" data-c="${i}"></div>`).join("") + `</div>`;
    if (r.tables.length) h += `<div class="section tables">` + r.tables.map(tableHTML).join("") + `</div>`;
    if (r.notes.length) h += `<div class="section"><ul class="notes">${r.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>`;
    if (r.code) h += `<details class="code"><summary>Lab code for this tool</summary><pre>${esc(r.code)}</pre></details>`;
    el.out.innerHTML = h;
    el.print.hidden = false;
    el.out.querySelectorAll(".shot").forEach((f) => inspector(f, r.images[+f.dataset.i]));
    el.out.querySelectorAll(".chart").forEach((c) => chart(c, r.charts[+c.dataset.c]));
    el.out.querySelectorAll(".graph").forEach((c) => graph(c, r.graphs[+c.dataset.g]));
  }

  function tableHTML(t) {
    const [lo, hi] = t.heat || [0, 1];
    const cell = (v) => {
      if (!t.heat) return `<td>${esc(v)}</td>`;
      const n = parseFloat(v);
      const g = Math.max(0, Math.min(255, Math.round(255 * (n - lo) / ((hi - lo) || 1))));
      return `<td style="background:rgb(${g},${g},${g});color:${g > 127 ? "#000" : "#fff"}">${esc(v)}</td>`;
    };
    const head = t.columns ? `<tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` : "";
    const rows = t.rows.map((row, i) => `<tr class="${t.highlight === i ? "hl" : ""}">${row.map(cell).join("")}</tr>`).join("");
    return `<div class="tbl"><h4>${esc(t.title)}</h4>${t.caption ? `<p>${esc(t.caption)}</p>` : ""}<table class="${t.heat ? "heat" : "plain"}">${head}${rows}</table></div>`;
  }

  /* ---------- pixel inspector ---------- */
  function inspector(fig, im) {
    const img = $("img", fig), out = $(".readout", fig);
    let ctx = null, W = 0, H = 0;
    const prep = () => {
      W = img.naturalWidth; H = img.naturalHeight;
      const c = document.createElement("canvas"); c.width = W; c.height = H;
      ctx = c.getContext("2d", { willReadFrequently: true }); ctx.drawImage(img, 0, 0);
    };
    img.complete ? prep() : (img.onload = prep);
    const hint = coarse ? "Touch the image to read a pixel" : "Move over the image to read a pixel";
    out.textContent = hint;
    const probe = (e) => {
      if (!ctx) return;
      const b = img.getBoundingClientRect();
      const px = Math.min(W - 1, Math.max(0, Math.floor((e.clientX - b.left) / b.width * W)));
      const py = Math.min(H - 1, Math.max(0, Math.floor((e.clientY - b.top) / b.height * H)));
      const [r, g, bl] = ctx.getImageData(px, py, 1, 1).data;
      const x = Math.round(px * im.w / W), y = Math.round(py * im.h / H);
      out.classList.add("live");
      out.innerHTML = `<span>x ${x}</span><span>y ${y}</span>` + (im.gray
        ? `<span>value ${r}</span><i class="chip" style="background:rgb(${r},${r},${r})"></i>`
        : `<span style="color:var(--r)">R ${r}</span><span style="color:var(--g)">G ${g}</span><span style="color:var(--b)">B ${bl}</span><i class="chip" style="background:rgb(${r},${g},${bl})"></i>`);
    };
    img.onpointermove = probe;
    img.onpointerdown = probe;
    img.onpointerleave = (e) => { if (e.pointerType === "mouse") { out.classList.remove("live"); out.textContent = hint; } };
  }

  /* ---------- charts (plain SVG, works offline) ---------- */
  const fmt = (v) => {
    if (v == null || isNaN(v)) return "-";
    const a = Math.abs(v);
    if (a >= 1e6) return +(v / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return +(v / 1e3).toFixed(1) + "k";
    if (a >= 100) return String(Math.round(v * 10) / 10);
    if (a >= 1) return String(+v.toFixed(3));
    if (a === 0) return "0";
    return String(+v.toPrecision(3));
  };
  function niceCeil(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); const f = v / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }

  function chart(host, c) {
    const W = 640, H = 240, m = { l: 54, r: 12, t: 12, b: 40 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const n = c.series[0].values.length;
    const all = c.series.flatMap((s) => s.values.filter((v) => v != null && isFinite(v)));
    const dmax = Math.max(1e-9, ...all), dmin = Math.min(0, ...all);
    const top = niceCeil(dmax), bot = dmin < 0 ? -niceCeil(-dmin) : 0;
    const bars = c.kind === "bars" || (c.kind === "auto" && n <= 32);
    const X = (i) => bars ? m.l + (i + .5) * (pw / n) : m.l + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
    const Y = (v) => m.t + ph - ((v - bot) / (top - bot)) * ph;
    let g = `<g class="grid">`;
    for (let k = 0; k <= 4; k++) { const y = m.t + ph - (k / 4) * ph; g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}"/>`; }
    g += `</g><g class="axis">`;
    for (let k = 0; k <= 4; k++) g += `<text x="${m.l - 8}" y="${m.t + ph - (k / 4) * ph + 4}" text-anchor="end">${fmt(bot + (top - bot) * k / 4)}</text>`;
    let xt;
    if (c.labels) xt = n > 12 ? [...Array(n).keys()].filter((i) => i % Math.ceil(n / 8) === 0) : [...Array(n).keys()];
    else xt = n > 32 ? [0, Math.round(n / 4), Math.round(n / 2), Math.round(3 * n / 4), n - 1] : [...Array(n).keys()].filter((i) => n <= 16 || i % Math.ceil(n / 12) === 0);
    xt.forEach((i) => g += `<text x="${X(i)}" y="${H - 20}" text-anchor="middle">${esc(c.labels ? c.labels[i] : i)}</text>`);
    g += `<text x="${m.l + pw / 2}" y="${H - 4}" text-anchor="middle">${esc(c.xLabel)}</text></g>`;

    let body = "";
    const y0 = Y(0);
    if (!bars) {
      c.series.forEach((s) => {
        const pts = s.values.map((v, i) => `${X(i).toFixed(1)},${Y(v ?? 0).toFixed(1)}`).join(" ");
        if (n > 40) body += `<polygon points="${X(0)},${y0} ${pts} ${X(n - 1)},${y0}" fill="${s.color}" fill-opacity="${c.series.length > 1 ? .2 : .32}"/>`;
        body += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.5" stroke-linejoin="round"/>`;
        if (n <= 40) body += s.values.map((v, i) => `<circle cx="${X(i)}" cy="${Y(v ?? 0)}" r="3" fill="${s.color}"/>`).join("");
      });
    } else {
      const slot = pw / n, bw = slot * .8 / c.series.length;
      c.series.forEach((s, k) => s.values.forEach((v, i) => {
        const y = Y(v ?? 0);
        body += `<rect x="${m.l + i * slot + slot * .1 + k * bw}" y="${Math.min(y, y0)}" width="${Math.max(1, bw - 1)}" height="${Math.abs(y0 - y)}" fill="${s.color}"/>`;
      }));
    }
    let marker = "";
    const marks = c.marker == null ? [] : Array.isArray(c.marker) ? c.marker : [[c.marker, "T = " + c.marker]];
    marks.forEach(([mx, lab], j) => {
      const x = X(Math.max(0, Math.min(n - 1, mx)));
      marker += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ph}" stroke="#ffc933" stroke-dasharray="4 4"/><text x="${x + 5}" y="${m.t + 12 + j * 13}" fill="#ffc933" font-size="11" font-family="var(--mono)">${esc(lab)}</text>`;
    });
    const named = c.series.length > 1 || c.series[0].name;
    host.innerHTML = `<h4><span>${esc(c.title)}</span>${named ? `<span class="legend">${c.series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</span>` : ""}</h4>
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(c.title)}">${g}${body}${marker}<line class="hair" y1="${m.t}" y2="${m.t + ph}" stroke="#777" stroke-width="1" visibility="hidden"/><rect class="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent"/></svg>
      <div class="readline">${coarse ? "Touch the chart to read values" : "Hover the chart to read values"}</div>`;
    const hint = coarse ? "Touch the chart to read values" : "Hover the chart to read values";
    const svg = $("svg", host), hair = $(".hair", host), hit = $(".hit", host), rl = $(".readline", host);
    const scrub = (e) => {
      const b = svg.getBoundingClientRect();
      const x = (e.clientX - b.left) / b.width * W;
      const i = Math.max(0, Math.min(n - 1, bars ? Math.floor((x - m.l) / (pw / n)) : Math.round((x - m.l) / pw * (n - 1))));
      hair.setAttribute("x1", X(i)); hair.setAttribute("x2", X(i)); hair.setAttribute("visibility", "visible");
      rl.textContent = `${c.labels ? c.labels[i] : c.xLabel + " " + i}:  ` + c.series.map((s) => `${s.name || c.yLabel} ${s.values[i] == null ? "-" : (+s.values[i]).toLocaleString(undefined, { maximumFractionDigits: 4 })}`).join("   ");
    };
    hit.onpointermove = scrub;
    hit.onpointerdown = scrub;
    hit.onpointerleave = (e) => { if (e.pointerType === "mouse") { hair.setAttribute("visibility", "hidden"); rl.textContent = hint; } };
  }

  /* ---------- graph (pixel connectivity) ----------
     nodes: [{x, y, label, on}]  edges: [{a, b, kind}] kind = n4 | nd | m */
  function graph(host, gr) {
    const xs = gr.nodes.map((n) => n.x), ys = gr.nodes.map((n) => n.y);
    const cols = Math.max(...xs) + 1, rows = Math.max(...ys) + 1;
    const cell = Math.max(34, Math.min(70, 520 / Math.max(cols, rows)));
    const pad = 30, W = cols * cell + pad, H = rows * cell + pad;
    const P = (n) => [pad / 2 + cell / 2 + n.x * cell, pad / 2 + cell / 2 + n.y * cell];
    const col = { n4: "#ffc933", nd: "#4d8dff", m: "#3ddc84" };
    let s = "";
    gr.edges.forEach((e) => {
      const [x1, y1] = P(gr.nodes[e.a]), [x2, y2] = P(gr.nodes[e.b]);
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col[e.kind] || "#888"}" stroke-width="2" ${e.kind === "nd" ? 'stroke-dasharray="5 4"' : ""}/>`;
    });
    const r = Math.min(15, cell * .33);
    gr.nodes.forEach((n) => {
      const [x, y] = P(n);
      s += `<circle cx="${x}" cy="${y}" r="${r}" fill="${n.on ? "#fff" : "#000"}" stroke="${n.on ? "#fff" : "#555"}" stroke-width="1.5"/>` +
           `<text x="${x}" y="${y + 4}" text-anchor="middle" font-size="${r > 11 ? 12 : 10}" font-family="var(--mono)" fill="${n.on ? "#000" : "#888"}">${esc(n.label)}</text>`;
    });
    host.innerHTML = `<h4>${esc(gr.title)}</h4>${gr.caption ? `<p>${esc(gr.caption)}</p>` : ""}
      <svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="${esc(gr.title)}">${s}</svg>
      <div class="glegend"><span><i style="border-color:${col.n4}"></i>4-adjacent</span><span><i style="border-color:${col.nd};border-top-style:dashed"></i>diagonal</span><span><i style="border-color:${col.m}"></i>m-adjacent</span></div>`;
  }

  /* ---------- start ---------- */
  fetch("/api/catalog").then((r) => r.json()).then((g) => {
    state.groups = g;
    buildNav();
    showHome();
  });
})();
