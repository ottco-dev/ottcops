// OPENCORE Analyzer front-end controller.
// Handles prompt templates, ML-only vs. hybrid runs, batch uploads, token-mode routing,
// export utilities and the automation stream dashboard.

const PROMPT_TEMPLATES = [
  {
    id: "trichomes_maturity",
    label: "Trichome-Reifegrad",
    description: "Analysiert Klar/Milchig/Amber und das Erntefenster.",
    prompt:
      "Analysiere bitte die Trichome auf Klar/Milchig/Amber, beschreibe die Verteilung und schätze das optimale Erntefenster.",
  },
  {
    id: "bud_health_mold",
    label: "Bud-Health / Schimmel",
    description: "Prüft Buds auf Schimmel, Fäulnis und generelle Gesundheit.",
    prompt:
      "Untersuche das Bild auf Schimmel, Fäulnis oder andere Probleme der Bud-Gesundheit und beschreibe die Risiken.",
  },
  {
    id: "pest_detection",
    label: "Pest-Detection",
    description: "Sucht nach sichtbaren Schädlingen oder typischen Schadmustern.",
    prompt:
      "Analysiere das Bild auf sichtbare Schädlinge oder typische Schadbilder und beschreibe die Befunde.",
  },
  {
    id: "bag_appeal",
    label: "Bag Appeal",
    description: "Bewertet Trim, Struktur, Frost und Farbspiel.",
    prompt:
      "Bewerte das visuelle Erscheinungsbild (Bag Appeal) der Buds: Trim, Struktur, Trichomdichte, Farbspiel, Gesamteindruck.",
  },
];

const STORAGE_KEYS = {
  THEME: "opencore_theme",
  TEMPLATES: "opencore_custom_templates",
  API_SETTINGS: "opencore_settings",
};

const state = {
  files: [],
  previews: [],
  customTemplates: [],
  debugEnabled: false,
  theme: "dark",
  lastResult: null,
  activeTab: "summary",
  analysisMode: "hybrid",
  streams: [],
  streamTimer: null,
  models: [],
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  initTheme();
  initTemplates();
  bindEvents();
  loadModels();
  applyDebugFromQuery();
  updatePreviewGrid();
  renderResultPlaceholder();
  handleAnalysisModeChange();
  loadStreams();
  state.streamTimer = setInterval(loadStreams, 15000);
});

function cacheDom() {
  dom.form = document.getElementById("analyze-form");
  dom.modelSelect = document.getElementById("model-select");
  dom.streamModel = document.getElementById("streamModel");
  dom.modelStatus = document.getElementById("model-status");
  dom.prompt = document.getElementById("prompt");
  dom.templateSelect = document.getElementById("template-select");
  dom.templateList = document.getElementById("customTemplateList");
  dom.saveTemplateBtn = document.getElementById("saveTemplateBtn");
  dom.templateModal = document.getElementById("templateModal");
  dom.templateName = document.getElementById("templateName");
  dom.templateDescription = document.getElementById("templateDescription");
  dom.confirmTemplate = document.getElementById("confirmTemplate");
  dom.cancelTemplate = document.getElementById("cancelTemplate");
  dom.imageInput = document.getElementById("image");
  dom.dropzone = document.getElementById("dropzone");
  dom.previewContainer = document.getElementById("imagePreviewContainer");
  dom.runStatus = document.getElementById("runStatus");
  dom.analyzeBtn = document.getElementById("analyzeBtn");
  dom.resultTabs = document.getElementById("result-tabs");
  dom.resultDisplay = document.getElementById("result-display");
  dom.resultJson = document.getElementById("result-json");
  dom.debugPanel = document.getElementById("debugPanel");
  dom.debugToggle = document.getElementById("debugToggle");
  dom.themeToggle = document.getElementById("themeToggle");
  dom.apiSettingsBtn = document.getElementById("apiSettingsBtn");
  dom.apiModal = document.getElementById("apiModal");
  dom.apiBaseUrl = document.getElementById("apiBaseUrl");
  dom.apiToken = document.getElementById("apiToken");
  dom.apiModeEnabled = document.getElementById("apiModeEnabled");
  dom.saveApiSettings = document.getElementById("saveApiSettings");
  dom.resetApiSettings = document.getElementById("resetApiSettings");
  dom.closeApiModal = document.getElementById("closeApiModal");
  dom.jsonFullscreenBtn = document.getElementById("jsonFullscreenBtn");
  dom.jsonFullscreen = document.getElementById("jsonFullscreen");
  dom.jsonFullscreenOutput = document.getElementById("jsonFullscreenOutput");
  dom.closeJsonFullscreen = document.getElementById("closeJsonFullscreen");
  dom.jsonDownloadBtn = document.getElementById("jsonDownloadBtn");
  dom.pdfExportBtn = document.getElementById("pdfExportBtn");
  dom.shareBtn = document.getElementById("shareBtn");
  dom.toast = document.getElementById("toast");
  dom.imageModal = document.getElementById("imageModal");
  dom.modalImage = document.getElementById("modalImage");
  dom.closeImageModal = document.getElementById("closeImageModal");
  dom.zoomSlider = document.getElementById("zoomSlider");
  dom.fileStatus = document.getElementById("fileStatus");
  dom.analysisMode = document.getElementById("analysisMode");
  dom.analysisHint = document.getElementById("analysisHint");
  dom.mlDebugLog = document.getElementById("mlDebugLog");
  dom.mlDebugLogText = document.getElementById("mlDebugLogText");
  dom.streamForm = document.getElementById("streamForm");
  dom.streamName = document.getElementById("streamName");
  dom.streamUrl = document.getElementById("streamUrl");
  dom.streamSourceType = document.getElementById("streamSourceType");
  dom.streamAnalysisMode = document.getElementById("streamAnalysisMode");
  dom.streamPrompt = document.getElementById("streamPrompt");
  dom.streamCaptureInterval = document.getElementById("streamCaptureInterval");
  dom.streamBatchInterval = document.getElementById("streamBatchInterval");
  dom.streamList = document.getElementById("streamList");
  dom.streamStatus = document.getElementById("streamStatus");
  dom.streamRefresh = document.getElementById("streamRefresh");
}

function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEYS.THEME);
  state.theme = saved === "light" ? "light" : "dark";
  applyTheme();
  dom.themeToggle.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme();
    localStorage.setItem(STORAGE_KEYS.THEME, state.theme);
  });
}

function applyTheme() {
  document.body.classList.toggle("theme-light", state.theme === "light");
}

function setMlDebugMessage(lines = []) {
  if (!dom.mlDebugLog || !dom.mlDebugLogText) return;
  if (!lines.length) {
    dom.mlDebugLog.style.display = "none";
    dom.mlDebugLogText.textContent = "";
    return;
  }
  dom.mlDebugLog.style.display = "block";
  dom.mlDebugLogText.textContent = lines.join("\n");
}

function initTemplates() {
  const raw = localStorage.getItem(STORAGE_KEYS.TEMPLATES);
  if (raw) {
    try {
      state.customTemplates = JSON.parse(raw) ?? [];
    } catch {
      state.customTemplates = [];
    }
  }
  renderTemplateOptions();
  renderCustomTemplateList();
}

function renderTemplateOptions() {
  if (!dom.templateSelect) return;
  dom.templateSelect.innerHTML = '<option value="">Template auswählen …</option>';
  PROMPT_TEMPLATES.forEach((tpl) => {
    const option = document.createElement("option");
    option.value = tpl.id;
    option.textContent = `${tpl.label} — ${tpl.description}`;
    dom.templateSelect.appendChild(option);
  });
  if (state.customTemplates.length) {
    const divider = document.createElement("option");
    divider.disabled = true;
    divider.textContent = "──── Eigene Templates ────";
    dom.templateSelect.appendChild(divider);
    state.customTemplates.forEach((tpl) => {
      const option = document.createElement("option");
      option.value = tpl.id;
      option.textContent = `${tpl.label} (Custom)`;
      dom.templateSelect.appendChild(option);
    });
  }
}

function renderCustomTemplateList() {
  if (!dom.templateList) return;
  dom.templateList.innerHTML = "";
  state.customTemplates.forEach((tpl) => {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.justifyContent = "space-between";
    row.style.gap = "0.5rem";
    const info = document.createElement("div");
    info.innerHTML = `<strong>${tpl.label}</strong><br /><small>${tpl.description || ""}</small>`;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "pill-button";
    removeBtn.textContent = "Löschen";
    removeBtn.addEventListener("click", () => removeCustomTemplate(tpl.id));
    row.append(info, removeBtn);
    dom.templateList.appendChild(row);
  });
}

function bindEvents() {
  dom.templateSelect.addEventListener("change", handleTemplateSelect);
  dom.saveTemplateBtn.addEventListener("click", () => openModal(dom.templateModal));
  dom.cancelTemplate.addEventListener("click", () => closeModal(dom.templateModal));
  dom.confirmTemplate.addEventListener("click", saveCustomTemplate);
  dom.form.addEventListener("submit", handleAnalyzeSubmit);
  dom.imageInput.addEventListener("change", handleFileChange);
  dom.debugToggle.addEventListener("change", () => {
    state.debugEnabled = dom.debugToggle.checked;
  });
  dom.apiSettingsBtn.addEventListener("click", openApiModal);
  dom.closeApiModal.addEventListener("click", () => closeModal(dom.apiModal));
  dom.saveApiSettings.addEventListener("click", saveApiSettings);
  dom.resetApiSettings.addEventListener("click", resetApiSettings);
  dom.jsonFullscreenBtn.addEventListener("click", () => openJsonFullscreen());
  dom.closeJsonFullscreen.addEventListener("click", () => closeModal(dom.jsonFullscreen));
  dom.jsonDownloadBtn.addEventListener("click", downloadJsonReport);
  dom.pdfExportBtn.addEventListener("click", generatePdfReport);
  dom.shareBtn.addEventListener("click", shareReport);
  dom.closeImageModal.addEventListener("click", () => closeModal(dom.imageModal));
  dom.zoomSlider.addEventListener("input", handleZoomChange);
  dom.analysisMode.addEventListener("change", handleAnalysisModeChange);
  dom.streamForm.addEventListener("submit", handleStreamSubmit);
  dom.streamRefresh.addEventListener("click", loadStreams);
  window.addEventListener("keyup", (event) => {
    if (event.key === "Escape") {
      [dom.templateModal, dom.apiModal, dom.jsonFullscreen, dom.imageModal].forEach(closeModal);
    }
  });
  setupDropzone();
}

function applyDebugFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const debug = params.get("debug");
  if (debug && ["1", "true"].includes(debug.toLowerCase())) {
    dom.debugToggle.checked = true;
    state.debugEnabled = true;
  }
}

function handleAnalysisModeChange() {
  state.analysisMode = dom.analysisMode.value;
  const isMl = state.analysisMode === "ml";
  dom.analysisHint.textContent = isMl
    ? "ML-only benötigt keinen Prompt."
    : "Hybrid kombiniert ML + GPT und benötigt einen Prompt.";
  if (dom.prompt) {
    dom.prompt.required = !isMl;
    dom.prompt.placeholder = isMl ? "Optionaler Hinweis für das ML-Log" : "Beschreibe die gewünschte Analyse";
  }
  if (isMl) {
    setMlDebugMessage(["ML-Modus aktiv. Modelle werden beim nächsten Run initialisiert."]);
  } else {
    setMlDebugMessage([]);
  }
}

function handleTemplateSelect() {
  const value = dom.templateSelect.value;
  if (!value) return;
  const builtIn = PROMPT_TEMPLATES.find((tpl) => tpl.id === value);
  const custom = state.customTemplates.find((tpl) => tpl.id === value);
  const template = builtIn || custom;
  if (template && dom.prompt) {
    dom.prompt.value = template.prompt;
  }
}

function saveCustomTemplate() {
  const label = dom.templateName.value.trim();
  const description = dom.templateDescription.value.trim();
  const prompt = dom.prompt.value.trim();
  if (!label || !prompt) {
    showToast("Name und Prompt sind erforderlich.");
    return;
  }
  const entry = {
    id: `custom_${Date.now()}`,
    label,
    description,
    prompt,
  };
  state.customTemplates.push(entry);
  persistCustomTemplates();
  renderTemplateOptions();
  renderCustomTemplateList();
  dom.templateName.value = "";
  dom.templateDescription.value = "";
  closeModal(dom.templateModal);
  showToast("Template gespeichert.");
}

function persistCustomTemplates() {
  localStorage.setItem(STORAGE_KEYS.TEMPLATES, JSON.stringify(state.customTemplates));
}

function removeCustomTemplate(id) {
  state.customTemplates = state.customTemplates.filter((tpl) => tpl.id !== id);
  persistCustomTemplates();
  renderTemplateOptions();
  renderCustomTemplateList();
  showToast("Template entfernt.");
}

function setupDropzone() {
  ["dragenter", "dragover"].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dom.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dom.dropzone.classList.remove("dragover");
    });
  });
  dom.dropzone.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer.files || []);
    if (!files.length) return;
    state.files = state.files.concat(files);
    updatePreviewGrid();
  });
}

function handleFileChange(event) {
  const files = Array.from(event.target.files || []);
  state.files = files;
  updatePreviewGrid();
}

function updatePreviewGrid() {
  dom.previewContainer.innerHTML = "";
  if (!state.files.length) {
    dom.fileStatus.textContent = "Keine Dateien ausgewählt.";
    return;
  }
  dom.fileStatus.textContent = `${state.files.length} Datei(en) ausgewählt.`;
  state.files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const item = document.createElement("div");
      item.className = "preview-item";
      item.innerHTML = `<img src="${event.target.result}" alt="Preview" /><small>${file.name || `Bild ${index + 1}`}</small>`;
      item.addEventListener("click", () => openImageModal(event.target.result));
      dom.previewContainer.appendChild(item);
    };
    reader.readAsDataURL(file);
  });
}

function openImageModal(src) {
  dom.modalImage.src = src;
  dom.zoomSlider.value = "100";
  dom.modalImage.style.transform = "scale(1)";
  openModal(dom.imageModal);
}

function handleZoomChange() {
  const scale = Number(dom.zoomSlider.value) / 100;
  dom.modalImage.style.transform = `scale(${scale})`;
}

function loadModels() {
  if (!dom.modelSelect) return;
  dom.modelSelect.disabled = true;
  dom.modelSelect.innerHTML = "<option>Modelle werden geladen …</option>";
  apiRequest("/tm-models")
    .then((payload) => {
      const models = payload.models || [];
      state.models = models;
      const hasBuiltin = payload.has_builtin;
      const defaultId = payload.default_model_id;
      populateModelSelect(dom.modelSelect, models, hasBuiltin, defaultId);
      if (dom.streamModel) {
        populateModelSelect(dom.streamModel, models, hasBuiltin, defaultId, true);
      }
      dom.modelStatus.textContent = "Die Auswahl bestimmt das aktive Teachable-Machine-Paket.";
    })
    .catch((error) => {
      dom.modelStatus.textContent = `Modelle konnten nicht geladen werden (${error.message}).`;
    })
    .finally(() => {
      dom.modelSelect.disabled = false;
    });
}

function populateModelSelect(selectEl, models, hasBuiltin, defaultId, optional = false) {
  if (!selectEl) return;
  selectEl.innerHTML = optional ? '<option value="">Standard verwenden</option>' : "";
  if (hasBuiltin) {
    const option = document.createElement("option");
    option.value = "builtin";
    option.textContent = "OPENCORE Referenz (TEACHABLE_MODEL_PATH)";
    selectEl.appendChild(option);
  }
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.name} (${model.type})`;
    selectEl.appendChild(option);
  });
  if (!optional) {
    if (defaultId && models.some((model) => model.id === defaultId)) {
      selectEl.value = defaultId;
    } else if (hasBuiltin) {
      selectEl.value = "builtin";
    } else if (models.length) {
      selectEl.value = models[0].id;
    }
  }
}

async function handleAnalyzeSubmit(event) {
  event.preventDefault();
  const prompt = dom.prompt.value.trim();
  if (state.analysisMode !== "ml" && !prompt) {
    showToast("Prompt ist erforderlich.");
    return;
  }
  if (!state.files.length) {
    showToast("Bitte mindestens ein Bild hinzufügen.");
    return;
  }
  if (state.analysisMode === "ml") {
    setMlDebugMessage(["Initialisiere Modell …", `Uploads: ${state.files.length}`]);
  }
  dom.runStatus.textContent = "Analyse läuft …";
  dom.analyzeBtn.disabled = true;
  const formData = new FormData();
  formData.append("prompt", prompt);
  const modelValue = dom.modelSelect.value;
  if (modelValue) {
    formData.append("model_id", modelValue);
  }
  if (state.debugEnabled) {
    formData.append("debug", "1");
  }
  formData.append("analysis_mode", state.analysisMode);
  let endpoint = state.files.length > 1 ? "/api/opencore/analyze-batch" : "/analyze";
  if (state.files.length > 1) {
    state.files.forEach((file) => formData.append("files[]", file));
  } else {
    formData.append("image", state.files[0]);
  }
  const clientStart = performance.now();
  try {
    const payload = await apiRequest(endpoint, { method: "POST", body: formData });
    const elapsed = Math.round(performance.now() - clientStart);
    handleResult(payload, elapsed);
    dom.runStatus.textContent = `Fertig (${elapsed} ms)`;
  } catch (error) {
    dom.runStatus.textContent = "Fehler";
    dom.resultDisplay.innerHTML = `<p class="disclaimer">Fehler: ${error.message}</p>`;
    dom.resultJson.textContent = JSON.stringify({ status: "error", message: error.message }, null, 2);
    dom.debugPanel.classList.add("active");
    dom.debugPanel.textContent = `Client-Fehler: ${error.message}`;
    if (state.analysisMode === "ml") {
      setMlDebugMessage([`Fehler: ${error.message}`]);
    }
  } finally {
    dom.analyzeBtn.disabled = false;
  }
}

function handleResult(payload, elapsedMs) {
  const normalized = normalizeResult(payload);
  state.lastResult = normalized;
  renderResultTabs(normalized);
  renderJson(normalized);
  renderDebug(normalized.debug, elapsedMs);
  renderMlDebugWidget(normalized);
}

function normalizeResult(payload) {
  if (!payload) return null;
  if (payload && Array.isArray(payload.items)) {
    return payload;
  }
  const wrapper = {
    status: payload.status || "ok",
    summary: { text: payload.summary?.text || payload.gpt_response || "Report erstellt." },
    items: [
      {
        image_id: state.files[0]?.name || "upload",
        analysis: payload,
      },
    ],
    teachable_model: payload.teachable_model || payload.items?.[0]?.analysis?.teachable_model,
    debug: payload.debug,
    analysis_mode: payload.analysis_mode || payload.items?.[0]?.analysis_mode || "hybrid",
  };
  return wrapper;
}

function renderResultTabs(result) {
  if (!result) {
    renderResultPlaceholder();
    return;
  }
  dom.resultTabs.innerHTML = "";
  const tabs = [{ key: "summary", label: "Gesamt-Report" }];
  result.items.forEach((item, index) => {
    tabs.push({ key: `item-${index}`, label: item.image_id || `Bild ${index + 1}` });
  });
  tabs.forEach((tab, index) => {
    const button = document.createElement("button");
    button.textContent = tab.label;
    button.classList.toggle("active", index === 0);
    button.addEventListener("click", () => {
      state.activeTab = tab.key;
      document.querySelectorAll(".tab-bar button").forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      renderTabContent(result, tab.key);
    });
    dom.resultTabs.appendChild(button);
  });
  state.activeTab = "summary";
  renderTabContent(result, "summary");
}

function renderTabContent(result, key) {
  if (!result) return;
  if (key === "summary") {
    const mode = result.analysis_mode || "hybrid";
    dom.resultDisplay.innerHTML = `<p><strong>Modus:</strong> ${mode.toUpperCase()}</p><p>${
      result.summary?.text || "Keine Zusammenfassung vorhanden."
    }</p>`;
    return;
  }
  const index = Number(key.split("-")[1]);
  const item = result.items[index];
  if (!item) {
    dom.resultDisplay.innerHTML = `<p class="disclaimer">Kein Eintrag gefunden.</p>`;
    return;
  }
  const classificationHtml = renderClassification(item.analysis?.classification);
  const gptHtml = item.analysis?.gpt_response
    ? `<h4>LLM-Output</h4><pre style="white-space: pre-wrap; font-family: 'Space Mono', monospace;">${
        item.analysis.gpt_response
      }</pre>`
    : `<p class="disclaimer">ML-only: Kein LLM-Text für dieses Bild.</p>`;
  dom.resultDisplay.innerHTML = `
    <h3>${item.image_id}</h3>
    <p><strong>Modell:</strong> ${item.analysis?.teachable_model?.name || "OPENCORE"}</p>
    ${classificationHtml}
    ${gptHtml}
  `;
}

function renderClassification(classification) {
  if (!classification) return "";
  const topConfidence = classification.top_confidence;
  const topPercent = typeof topConfidence === "number" ? (topConfidence * 100).toFixed(2) : topConfidence;
  const rows = (classification.all_predictions || [])
    .map((pred) => {
      const value = typeof pred.confidence === "number" ? (pred.confidence * 100).toFixed(2) + "%" : pred.confidence;
      return `<li>${pred.label}: ${value}</li>`;
    })
    .join("");
  return `
    <div class="classification-block">
      <strong>Top Label:</strong> ${classification.top_label} (${topPercent}%)
      <ul>${rows}</ul>
    </div>
  `;
}

function renderJson(result) {
  dom.resultJson.textContent = JSON.stringify(result, null, 2);
}

function renderDebug(debug, clientMs) {
  if (!debug && !state.debugEnabled) {
    dom.debugPanel.classList.remove("active");
    dom.debugPanel.textContent = "";
    return;
  }
  dom.debugPanel.classList.add("active");
  const timings = debug?.timings || {};
  dom.debugPanel.innerHTML = `
    <strong>Request-ID:</strong> ${debug?.request_id || "n/a"}<br />
    <strong>Modell:</strong> ${debug?.model_name || "OPENCORE"} (${debug?.model_version || ""})<br />
    <strong>Timings:</strong> Modell ${timings.model_ms || "?"} ms · LLM ${timings.llm_ms || "?"} ms · Gesamt ${
    timings.total_ms || "?"
  } ms<br />
    <strong>Client:</strong> ${clientMs || "?"} ms<br />
    <strong>Prompt:</strong> ${debug?.prompt_preview || "-"}
  `;
}

function renderMlDebugWidget(result) {
  if (!dom.mlDebugLog) return;
  if (!result || result.analysis_mode !== "ml") {
    setMlDebugMessage([]);
    return;
  }
  const lines = [];
  const debug = result.debug || {};
  if (debug.model_name) {
    lines.push(`Modell: ${debug.model_name}`);
  }
  if (debug.model_version) {
    lines.push(`Version: ${debug.model_version}`);
  }
  if (debug.timings) {
    const modelMs = debug.timings.model_ms ?? "?";
    const totalMs = debug.timings.total_ms ?? "?";
    lines.push(`Timings: Modell ${modelMs} ms · Gesamt ${totalMs} ms`);
  }
  if (debug.request_id) {
    lines.push(`Request-ID: ${debug.request_id}`);
  }
  if (debug.timestamp) {
    lines.push(`Zeit: ${debug.timestamp}`);
  }
  const firstClassification = result.items?.[0]?.analysis?.classification;
  if (firstClassification?.top_label) {
    const topConf = typeof firstClassification.top_confidence === "number"
      ? `${(firstClassification.top_confidence * 100).toFixed(2)}%`
      : firstClassification.top_confidence;
    lines.push(`Top Label: ${firstClassification.top_label} (${topConf})`);
  }
  setMlDebugMessage(lines.length ? lines : ["ML-only Run abgeschlossen."]); 
}

function openModal(element) {
  element.classList.remove("hidden");
}

function closeModal(element) {
  element.classList.add("hidden");
}

function openApiModal() {
  const settings = loadApiSettings();
  dom.apiBaseUrl.value = settings.baseUrl || "";
  dom.apiToken.value = settings.token || "";
  dom.apiModeEnabled.checked = Boolean(settings.enabled);
  openModal(dom.apiModal);
}

function loadApiSettings() {
  const raw = localStorage.getItem(STORAGE_KEYS.API_SETTINGS);
  if (!raw) return {};
  try {
    return JSON.parse(raw) ?? {};
  } catch {
    return {};
  }
}

function saveApiSettings() {
  const payload = {
    baseUrl: dom.apiBaseUrl.value.trim(),
    token: dom.apiToken.value.trim(),
    enabled: dom.apiModeEnabled.checked,
  };
  localStorage.setItem(STORAGE_KEYS.API_SETTINGS, JSON.stringify(payload));
  showToast("API-Einstellungen gespeichert.");
  closeModal(dom.apiModal);
}

function resetApiSettings() {
  localStorage.removeItem(STORAGE_KEYS.API_SETTINGS);
  dom.apiBaseUrl.value = "";
  dom.apiToken.value = "";
  dom.apiModeEnabled.checked = false;
  showToast("API-Einstellungen entfernt.");
}

function openJsonFullscreen(payload = null) {
  const data = payload || state.lastResult;
  if (!data) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  dom.jsonFullscreenOutput.textContent = JSON.stringify(data, null, 2);
  openModal(dom.jsonFullscreen);
}

function downloadJsonReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "opencore-report.json";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function generatePdfReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  const { jsPDF } = window.jspdf || {};
  if (!jsPDF) {
    showToast("PDF-Bibliothek nicht geladen.");
    return;
  }
  const doc = new jsPDF();
  doc.setFont("helvetica", "");
  doc.setFontSize(14);
  doc.text("OPENCORE Analyzer Report", 14, 20);
  doc.setFontSize(11);
  doc.text(`Datum: ${new Date().toLocaleString()}`, 14, 30);
  doc.text(`Modell: ${state.lastResult.teachable_model?.name || "OPENCORE"}`, 14, 38);
  doc.text(`Modus: ${(state.lastResult.analysis_mode || "hybrid").toUpperCase()}`, 14, 46);
  doc.setFontSize(12);
  doc.text("Summary:", 14, 56);
  doc.setFontSize(11);
  const summary = doc.splitTextToSize(state.lastResult.summary?.text || "", 180);
  doc.text(summary, 14, 64);
  let offset = 64 + summary.length * 5;
  state.lastResult.items.forEach((item, idx) => {
    doc.setFontSize(12);
    doc.text(`Bild ${idx + 1}: ${item.image_id}`, 14, offset);
    offset += 6;
    doc.setFontSize(10);
    const text = doc.splitTextToSize(item.analysis?.gpt_response || "ML-only", 180);
    doc.text(text, 14, offset);
    offset += text.length * 5 + 4;
    if (offset > 260) {
      doc.addPage();
      offset = 20;
    }
  });
  doc.save("opencore-report.pdf");
}

async function shareReport() {
  if (!state.lastResult) {
    showToast("Keine Analyse vorhanden.");
    return;
  }
  try {
    const payload = await apiRequest("/api/opencore/share", {
      method: "POST",
      body: { payload: state.lastResult },
    });
    const url = `${window.location.origin}${payload.url}`;
    await navigator.clipboard?.writeText(url).catch(() => {});
    showToast(`Share-Link erstellt: ${url}`);
  } catch (error) {
    showToast(`Share fehlgeschlagen: ${error.message}`);
  }
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("active");
  setTimeout(() => dom.toast.classList.remove("active"), 3200);
}

async function apiRequest(path, options = {}) {
  const settings = loadApiSettings();
  const headers = options.headers ? { ...options.headers } : {};
  const isFormData = options.body instanceof FormData;
  let url = path;
  if (settings.enabled && settings.baseUrl && !path.startsWith("http")) {
    url = `${settings.baseUrl}${path}`;
  }
  const fetchOptions = {
    method: options.method || "GET",
    headers,
    body: undefined,
  };
  if (settings.enabled && settings.token) {
    headers.Authorization = `Bearer ${settings.token}`;
  }
  if (options.body) {
    if (isFormData) {
      fetchOptions.body = options.body;
    } else {
      headers["Content-Type"] = "application/json";
      fetchOptions.body = JSON.stringify(options.body);
    }
  }
  const response = await fetch(url, fetchOptions);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  if (response.headers.get("content-type")?.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function renderResultPlaceholder() {
  dom.resultDisplay.innerHTML = `<p class="disclaimer">Noch keine Analyse durchgeführt.</p>`;
  setMlDebugMessage([]);
}

async function handleStreamSubmit(event) {
  event.preventDefault();
  const payload = {
    label: dom.streamName.value.trim(),
    source_url: dom.streamUrl.value.trim(),
    source_type: dom.streamSourceType.value,
    analysis_mode: dom.streamAnalysisMode.value,
    prompt: dom.streamPrompt.value.trim(),
    model_id: dom.streamModel.value || undefined,
    capture_interval: Number(dom.streamCaptureInterval.value) || 5,
    batch_interval: Number(dom.streamBatchInterval.value) || 30,
  };
  if (!payload.source_url) {
    showToast("Quelle ist erforderlich.");
    return;
  }
  try {
    await apiRequest("/api/opencore/streams", { method: "POST", body: payload });
    showToast("Stream aktiviert.");
    dom.streamForm.reset();
    dom.streamCaptureInterval.value = "5";
    dom.streamBatchInterval.value = "30";
    loadStreams();
  } catch (error) {
    showToast(`Stream konnte nicht gestartet werden: ${error.message}`);
  }
}

async function loadStreams() {
  try {
    dom.streamStatus.textContent = "Lade Streams …";
    const payload = await apiRequest("/api/opencore/streams");
    state.streams = payload.streams || [];
    renderStreamList();
    dom.streamStatus.textContent = state.streams.length
      ? `${state.streams.length} Stream(s) aktiv.`
      : "Noch keine Streams registriert.";
  } catch (error) {
    dom.streamStatus.textContent = `Streams konnten nicht geladen werden (${error.message}).`;
  }
}

function renderStreamList() {
  dom.streamList.innerHTML = "";
  state.streams.forEach((stream) => {
    const item = document.createElement("div");
    item.className = "stream-item";
    const summary = stream.last_result?.summary?.text || "Noch kein Report vorhanden.";
    const errorHtml = stream.last_error
      ? `<p class="disclaimer" style="color:#f87171;">Fehler: ${stream.last_error}</p>`
      : "";
    item.innerHTML = `
      <header>
        <strong>${stream.label || stream.stream_id}</strong>
        <small>${stream.analysis_mode?.toUpperCase() || "HYBRID"}</small>
      </header>
      <p class="disclaimer">${summary}</p>
      ${errorHtml}
      <small class="disclaimer">Letztes Capture: ${stream.last_capture_ts ? new Date(stream.last_capture_ts * 1000).toLocaleTimeString() : "-"}</small>
    `;
    const footer = document.createElement("footer");
    const viewBtn = document.createElement("button");
    viewBtn.textContent = "JSON";
    viewBtn.addEventListener("click", () => {
      if (!stream.last_result) {
        showToast("Noch kein Resultat für diesen Stream.");
        return;
      }
      openJsonFullscreen(stream.last_result);
    });
    const triggerBtn = document.createElement("button");
    triggerBtn.textContent = "Trigger";
    triggerBtn.addEventListener("click", () => triggerStream(stream.stream_id));
    const stopBtn = document.createElement("button");
    stopBtn.textContent = "Stop";
    stopBtn.addEventListener("click", () => stopStream(stream.stream_id));
    footer.append(viewBtn, triggerBtn, stopBtn);
    item.appendChild(footer);
    dom.streamList.appendChild(item);
  });
}

async function stopStream(streamId) {
  try {
    await apiRequest(`/api/opencore/streams/${streamId}`, { method: "DELETE" });
    showToast("Stream gestoppt.");
    loadStreams();
  } catch (error) {
    showToast(`Stop fehlgeschlagen: ${error.message}`);
  }
}

async function triggerStream(streamId) {
  try {
    await apiRequest(`/api/opencore/streams/${streamId}/trigger`, { method: "POST" });
    showToast("Analyse ausgelöst.");
    loadStreams();
  } catch (error) {
    showToast(`Trigger fehlgeschlagen: ${error.message}`);
  }
}
