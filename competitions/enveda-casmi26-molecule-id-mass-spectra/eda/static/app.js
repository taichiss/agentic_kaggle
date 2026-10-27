"use strict";

const $ = (id) => document.getElementById(id);
const state = { offset: 0, limit: 25, total: 0, selected: null, comparison: null, related: [], listRequest: 0, spectrumRequest: 0, structureRequest: 0 };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function value(x) {
  if (x === null || x === undefined || x === "") return "—";
  return typeof x === "object" ? JSON.stringify(x) : String(x);
}

function number(x, digits = 4) {
  const n = Number(x);
  return x === null || x === undefined || x === "" || !Number.isFinite(n) ? "—" : n.toFixed(digits);
}

function showError(id, message) {
  const el = $(id);
  el.textContent = message;
  el.hidden = !message;
}

function svgImage(svg, alt) {
  const image = document.createElement("img");
  image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  image.alt = alt;
  return image;
}

function placeholder(id, message) {
  const span = document.createElement("span");
  span.textContent = message;
  $(id).replaceChildren(span);
}

function setStructure(id, svg, alt) {
  $(id).replaceChildren(svgImage(svg, alt));
}

async function loadDataset() {
  const data = await api("/api/dataset");
  $("source-label").textContent = value(data.label || data.source);
  $("stat-spectra").textContent = Number(data.rows).toLocaleString();
  $("stat-molecules").textContent = data.molecules == null ? "—" : Number(data.molecules).toLocaleString();
  $("stat-label").textContent = data.has_structures ? "あり" : "なし";
  $("demo-notice").hidden = !data.demo;
  if (data.provenance) {
    const provenance = data.provenance;
    const checks = { verified: "一致", mismatch: "不一致", unavailable: "未確認", invalid_sidecar: "記録が不正" };
    const status = checks[provenance.checksum_status] || "未確認";
    const sourceKind = String(provenance.source_kind || "");
    const mirrorWarning = /third.party|mirror/i.test(sourceKind) && /not verified|unverified/i.test(sourceKind) ? "第三者ミラーの内容と公式データとの同一性は未確認です。" : "";
    $("provenance-notice").textContent = `データ出所（提供メタデータ）: ${value(provenance.source_kind)} / ${value(provenance.dataset)}。${mirrorWarning}ローカルファイルと付属記録のチェックサム: ${status}（一致しても出所の正しさを保証しません）。`;
    $("provenance-notice").hidden = false;
  }
  if (data.truncated) {
    $("limit-notice").textContent = `先頭 ${Number(data.rows).toLocaleString()} 件を読み込み中（全 ${Number(data.total_rows).toLocaleString()} 件）。検索と類似分子探索は読み込んだ範囲が対象です。`;
    $("limit-notice").hidden = false;
  }
  if (!data.has_structures) $("structure-context").textContent = "このデータに正解構造は含まれていません。候補 SMILES を入力して構造を比較できます。";
}

async function loadList() {
  const request = ++state.listRequest;
  const query = new URLSearchParams({ q: $("search").value.trim(), offset: state.offset, limit: state.limit });
  $("spectrum-list").replaceChildren(Object.assign(document.createElement("div"), { className: "empty", textContent: "読み込み中…" }));
  try {
    const data = await api(`/api/spectra?${query}`);
    if (request !== state.listRequest) return;
    state.total = data.total;
    $("result-count").textContent = `${data.total.toLocaleString()} 件`;
    $("page-info").textContent = data.total ? `${state.offset + 1}–${state.offset + data.items.length} / ${data.total.toLocaleString()}` : "0 件";
    $("prev").disabled = state.offset === 0;
    $("next").disabled = state.offset + data.items.length >= data.total;
    const nodes = data.items.map((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `spectrum-item${state.selected && String(state.selected.id) === String(item.id) ? " selected" : ""}`;
      button.setAttribute("aria-pressed", state.selected && String(state.selected.id) === String(item.id) ? "true" : "false");
      const name = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = value(item.molecule_id);
      const sub = document.createElement("small");
      sub.textContent = `${value(item.id)} · ${value(item.adduct)}`;
      name.append(title, sub);
      const mass = document.createElement("span");
      mass.className = "mass";
      mass.textContent = number(item.precursor_mz);
      button.append(name, mass);
      button.addEventListener("click", () => selectSpectrum(item.id));
      return button;
    });
    if (!nodes.length) nodes.push(Object.assign(document.createElement("div"), { className: "empty", textContent: "一致するスペクトルがありません。" }));
    $("spectrum-list").replaceChildren(...nodes);
    if (!state.selected && data.items.length) await selectSpectrum(data.items[0].id);
  } catch (error) {
    if (request === state.listRequest) showError("app-error", `一覧を読み込めませんでした: ${error.message}`);
  }
}

async function selectSpectrum(id) {
  const request = ++state.spectrumRequest;
  try {
    const item = await api(`/api/spectra/${encodeURIComponent(id)}`);
    if (request !== state.spectrumRequest) return;
    state.selected = item;
    state.comparison = null;
    ++state.structureRequest;
    $("smiles-form").querySelector("button").disabled = false;
    showError("structure-error", "");
    $("spectrum-id").textContent = value(item.id);
    $("peak-count").textContent = `${item.peaks.length.toLocaleString()} peaks · ${value(item.instrument)}`;
    const fields = [["PRECURSOR m/z", number(item.precursor_mz)], ["ADDUCT", value(item.adduct)], ["ION MODE", value(item.ion_mode)], ["COLLISION ENERGY (eV)", value(item.collision_energy)]];
    $("spectrum-metadata").replaceChildren(...fields.map(([label, text]) => {
      const div = document.createElement("div");
      const span = document.createElement("span"); span.textContent = label;
      const strong = document.createElement("strong"); strong.textContent = text;
      div.append(span, strong); return div;
    }));
    const entries = Object.entries(item.metadata || {}).sort(([a], [b]) => a.localeCompare(b));
    $("all-metadata").replaceChildren(...entries.flatMap(([key, val]) => {
      const dt = document.createElement("dt"); dt.textContent = key;
      const dd = document.createElement("dd"); dd.textContent = value(val);
      return [dt, dd];
    }));
    for (const button of $("spectrum-list").querySelectorAll("button")) {
      const selected = button.querySelector("small").textContent.startsWith(`${id} ·`);
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    clearComparison();
    drawChart();
    if (item.smiles) {
      $("smiles").value = item.smiles;
      $("query-origin").textContent = "データの構造";
      await findRelated();
    } else {
      $("smiles").value = "";
      $("query-origin").textContent = "候補 SMILES";
      $("query-smiles").textContent = "—";
      placeholder("query-structure", "正解構造は提供されていません");
      $("related-list").replaceChildren(Object.assign(document.createElement("div"), { className: "empty", textContent: "基準構造を入力すると、データ内と入力候補の類似分子を表示します。" }));
      $("related-count").textContent = "—";
    }
  } catch (error) {
    if (request === state.spectrumRequest) showError("app-error", `スペクトルを読み込めませんでした: ${error.message}`);
  }
}

function clearComparison() {
  state.comparison = null;
  $("compare-spectrum-control").hidden = true;
  $("compare-spectrum").replaceChildren();
  placeholder("compare-structure", "下の類似分子から選択");
  $("compare-smiles").textContent = "—";
  $("compare-score").textContent = "未選択";
}

async function findRelated(event) {
  if (event) event.preventDefault();
  const request = ++state.structureRequest;
  const smiles = $("smiles").value.trim();
  const candidates = $("candidates").value.split(/\r?\n/).map((v) => v.trim()).filter(Boolean);
  showError("structure-error", "");
  if (!smiles) { showError("structure-error", "基準となる SMILES を入力してください。"); return; }
  if (candidates.length > 100) { showError("structure-error", "比較候補は 100 件以内にしてください。"); return; }
  const button = $("smiles-form").querySelector("button");
  button.disabled = true;
  clearComparison(); drawChart();
  $("related-list").replaceChildren(Object.assign(document.createElement("div"), { className: "empty", textContent: "類似分子を検索中…" }));
  $("related-count").textContent = "—";
  try {
    const [structure, related] = await Promise.all([post("/api/structure", { smiles }), post("/api/related", { smiles, candidates, limit: 12 })]);
    if (request !== state.structureRequest) return;
    setStructure("query-structure", structure.svg, "基準分子の構造式");
    $("query-smiles").textContent = structure.smiles;
    if (event) $("query-origin").textContent = "入力した候補";
    state.related = related.items;
    $("related-count").textContent = `${related.items.length} 件`;
    renderRelated();
  } catch (error) {
    if (request !== state.structureRequest) return;
    placeholder("query-structure", "構造を表示できません");
    $("query-smiles").textContent = "—";
    $("related-list").replaceChildren();
    $("related-count").textContent = "—";
    showError("structure-error", `構造を読み込めませんでした: ${error.message}`);
  } finally { if (request === state.structureRequest) button.disabled = false; }
}

function renderRelated() {
  const nodes = state.related.map((item, index) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "related-item";
    button.setAttribute("aria-label", `比較候補 ${index + 1}、類似度 ${number(item.similarity, 3)}`);
    if (item.svg) button.append(svgImage(item.svg, ""));
    const score = document.createElement("strong"); score.textContent = `類似度 ${number(item.similarity, 3)}`;
    const info = document.createElement("small"); info.textContent = item.molecule_id ? value(item.molecule_id) : "入力した候補";
    const avail = document.createElement("small"); avail.textContent = item.spectrum_id != null ? "スペクトルあり" : "スペクトルなし";
    button.append(score, info, avail);
    button.addEventListener("click", () => selectRelated(item, button));
    return button;
  });
  if (!nodes.length) nodes.push(Object.assign(document.createElement("div"), { className: "empty", textContent: "類似分子が見つかりません。比較候補 SMILES を追加してください。" }));
  $("related-list").replaceChildren(...nodes);
}

async function selectRelated(item, button) {
  ++state.structureRequest;
  for (const element of $("related-list").querySelectorAll("button")) element.classList.remove("selected");
  button.classList.add("selected");
  if (item.svg) setStructure("compare-structure", item.svg, "比較分子の構造式");
  $("compare-smiles").textContent = value(item.smiles);
  $("compare-score").textContent = `Tanimoto ${number(item.similarity, 3)}`;
  state.comparison = null; drawChart();
  const spectra = item.spectra || (item.spectrum_id != null ? [{ id: item.spectrum_id }] : []);
  $("compare-spectrum-control").hidden = spectra.length === 0;
  $("compare-spectrum").replaceChildren(...spectra.map((spectrum) => {
    const option = document.createElement("option"); option.value = spectrum.id;
    option.textContent = `${value(spectrum.id)} · ${value(spectrum.adduct)} · ${value(spectrum.ion_mode)} · CE ${value(spectrum.collision_energy)} eV · ${value(spectrum.instrument)}`;
    return option;
  }));
  if (spectra.length) await loadComparison(spectra[0].id);
}

async function loadComparison(id) {
  const request = ++state.structureRequest;
  state.comparison = null; drawChart();
  showError("structure-error", "");
  try {
    const spectrum = await api(`/api/spectra/${encodeURIComponent(id)}`);
    if (request !== state.structureRequest) return;
    state.comparison = spectrum; drawChart();
  } catch (error) { if (request === state.structureRequest) showError("structure-error", `比較スペクトルを読み込めませんでした: ${error.message}`); }
}

const NS = "http://www.w3.org/2000/svg";
function svgElement(name, attributes = {}, content) {
  const el = document.createElementNS(NS, name);
  for (const [key, val] of Object.entries(attributes)) el.setAttribute(key, val);
  if (content !== undefined) el.textContent = content;
  return el;
}

function drawChart() {
  if (!state.selected) return;
  const normalize = (spectrum) => {
    const input = spectrum?.peaks || [];
    const peaks = input.filter((peak) => Array.isArray(peak) && peak.length >= 2 && typeof peak[0] === "number" && typeof peak[1] === "number" && Number.isFinite(peak[0]) && peak[0] >= 0 && Number.isFinite(peak[1]) && peak[1] >= 0);
    const valid = peaks.length;
    const max = peaks.reduce((n, peak) => Math.max(n, peak[1]), 0);
    if (peaks.length > 5000) peaks.sort((a, b) => b[1] - a[1] || a[0] - b[0]).length = 5000;
    peaks.sort((a, b) => a[0] - b[0]);
    return { original: input.length, valid, peaks: peaks.map(([mz, intensity]) => [mz, max ? intensity / max * 100 : 0, intensity]) };
  };
  const mainData = normalize(state.selected), comparisonData = normalize(state.comparison);
  const main = mainData.peaks, compare = comparisonData.peaks;
  const countDescription = (name, data, spectrum) => {
    const diagnostic = spectrum?.peak_diagnostics;
    const sourceCounts = diagnostic ? `元データ m/z ${value(diagnostic.mz_count)} 件・強度 ${value(diagnostic.intensity_count)} 件、無効ペア ${value(diagnostic.invalid_pair_count)} 件・対応なし ${value(diagnostic.unpaired_count)} 件。` : "";
    return `${name}: ${sourceCounts}取得 ${data.original.toLocaleString()} 件、表示 ${data.peaks.length.toLocaleString()} 件${data.valid < data.original ? `、表示できない値 ${data.original - data.valid} 件を除外` : ""}${data.valid > 5000 ? "（強度上位 5,000 件）" : ""}`;
  };
  $("chart-note").textContent = `強度は各スペクトルの最大ピークを 100% として表示。下向きは比較対象です。${countDescription("選択", mainData, state.selected)}${state.comparison ? ` / ${countDescription("比較", comparisonData, state.comparison)}` : ""}。表も選択スペクトルの表示分です。`;
  $("peak-table").replaceChildren(...main.map(([mz, pct, raw]) => {
    const row = document.createElement("tr");
    for (const text of [value(mz), value(raw), `${number(pct, 1)}%`]) {
      const cell = document.createElement("td"); cell.textContent = text; row.append(cell);
    }
    return row;
  }));
  const maxMz = [...main, ...compare].reduce((max, peak) => Math.max(max, peak[0]), 100);
  const ceiling = Math.ceil(maxMz / 50) * 50;
  const svg = svgElement("svg", { viewBox: "0 0 800 250", role: "img", "aria-label": "選択した MS/MS スペクトル。ピークは最大強度を 100% として表示。" });
  const left = 53, right = 780, top = 15, bottom = 218, baseline = compare.length ? 116 : bottom;
  const scaleX = (v) => left + v / ceiling * (right - left);
  const axis = (x1,y1,x2,y2,stroke = "#d9e5e3") => svg.append(svgElement("line", { x1,y1,x2,y2,stroke,"stroke-width":1 }));
  const label = (x,y,text,anchor = "middle") => svg.append(svgElement("text", { x,y,"text-anchor":anchor,fill:"#73888a","font-size":10,"font-family":"system-ui, sans-serif" }, text));
  for (let i = 0; i <= 5; i++) { const x = left + (right-left)*i/5; axis(x,top,x,bottom,"#edf2f1"); label(x,234,Math.round(ceiling*i/5)); }
  axis(left,baseline,right,baseline,"#9fb3b1");
  axis(left,top,left,bottom,"#9fb3b1");
  label(12, top + 5,"100", "start"); label(20,baseline + 4,"0", "start");
  if (compare.length) label(12,bottom,"100", "start");
  label(right,248,"m/z", "end");
  const tooltip = document.createElement("div"); tooltip.className = "chart-tooltip"; tooltip.hidden = true;
  function addPeaks(peaks, color, isComparison) {
    peaks.forEach(([mz, pct, raw]) => {
      const x = scaleX(mz);
      const extent = isComparison ? bottom-baseline : baseline-top;
      const y = baseline + (isComparison ? 1 : -1) * pct / 100 * extent;
      const line = svgElement("line", { x1:x, y1:baseline, x2:x, y2:y, stroke:color, "stroke-width":1.5, "aria-hidden":"true" });
      line.append(svgElement("title", {}, `m/z ${mz} · 相対強度 ${pct.toFixed(1)}% · 元の強度 ${raw}`));
      svg.append(line);
      const hit = svgElement("line", { x1:x, y1:baseline, x2:x, y2:y, stroke:"transparent", "stroke-width":8, "aria-hidden":"true" });
      const show = (event) => { tooltip.hidden = false; tooltip.textContent = `m/z ${mz}  ·  ${pct.toFixed(1)}%  (元値 ${raw})`; const rect = $("chart").getBoundingClientRect(); const px = event.clientX ? event.clientX-rect.left : x/800*rect.width; tooltip.style.left = `${Math.max(0,Math.min(rect.width-180,px+10))}px`; tooltip.style.top = "10px"; };
      hit.addEventListener("pointermove", show); hit.addEventListener("pointerleave", () => { tooltip.hidden = true; });
      svg.append(hit);
    });
  }
  addPeaks(main,"#087b75",false); addPeaks(compare,"#ba6633",true);
  $("chart").replaceChildren(svg,tooltip);
  const legend = [Object.assign(document.createElement("span"), { innerHTML: '<i class="swatch primary"></i>選択スペクトル' })];
  if (state.comparison) { const item = document.createElement("span"); const swatch = document.createElement("i"); swatch.className = "swatch comparison"; item.append(swatch, `比較: ${value(state.comparison.id)}`); legend.push(item); }
  $("chart-legend").replaceChildren(...legend);
  const comparisonMetadata = $("comparison-metadata");
  comparisonMetadata.hidden = !state.comparison;
  if (state.comparison) {
    const spectrum = state.comparison;
    const title = document.createElement("strong");
    title.textContent = `比較スペクトルの測定条件 · ${value(spectrum.id)}`;
    const detail = document.createElement("span");
    detail.textContent = `前駆体 m/z ${number(spectrum.precursor_mz)} · 付加イオン ${value(spectrum.adduct)} · イオンモード ${value(spectrum.ion_mode)} · 衝突エネルギー (eV) ${value(spectrum.collision_energy)} · 装置 ${value(spectrum.instrument)}`;
    comparisonMetadata.replaceChildren(title, detail);
  } else comparisonMetadata.replaceChildren();
}

let debounce;
$("search").addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => { state.offset = 0; loadList(); }, 250); });
$("prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadList(); });
$("next").addEventListener("click", () => { state.offset += state.limit; loadList(); });
$("smiles-form").addEventListener("submit", findRelated);
$("compare-spectrum").addEventListener("change", (event) => loadComparison(event.target.value));
Promise.resolve().then(loadDataset).then(loadList).catch((error) => showError("app-error", `データを読み込めませんでした: ${error.message}`));
