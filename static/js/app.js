// OPENCORE Analyzer front-end controller.
// Handles prompt templates, ML-only vs. hybrid runs, batch uploads, token-mode routing,
// export utilities and the automation stream dashboard.

const PROMPT_TEMPLATES = [
  {
    id: "trichomes_maturity",
    label: "Trichome-Reifegrad",
    description: "Analyzes clear/milky/amber balance and harvest timing.",
    prompt:
      "Analyze the trichomes for clear/milky/amber, describe the distribution, and estimate the optimal harvest window.",
  },
  {
    id: "bud_health_mold",
    label: "Bud-Health / Schimmel",
    description: "Checks buds for mold, rot, and general health.",
    prompt:
      "Inspect the image for mold, rot, or other bud health issues and describe the risks.",
  },
  {
    id: "pest_detection",
    label: "Pest-Detection",
    description: "Looks for visible pests or characteristic damage patterns.",
    prompt:
      "Analyze the image for visible pests or typical damage patterns and describe the findings.",
  },
  {
    id: "bag_appeal",
    label: "Bag Appeal",
    description: "Scores trim, structure, frost, and color play.",
    prompt:
      "Evaluate the visual appearance (bag appeal) of the buds: trim, structure, trichome density, color, and overall impression.",
  },
];

const FAQ_BLOCKS = [
  {
    id: "faq_full_health",
    title: "Full plant health",
    blurb: "Overall vigor, leaf color, deficiencies, stress.",
    prompt:
      "Provide a concise health assessment: vigor, leaf color, nutrient deficiency signs, stress markers, and any remedial actions.",
  },
  {
    id: "faq_environment",
    title: "Environment & sensors",
    blurb: "Blend sensor data with visible cues.",
    prompt:
      "Use the provided environment readings inside the prompt (humidity, temperature, CO2/PPFD, EC/PH) to contextualize what you see in the image. Flag mismatches between visuals and sensor values.",
  },
  {
    id: "faq_pests",
    title: "Pest sweep",
    blurb: "Check for pests and typical damage patterns.",
    prompt: "Scan for pests or pest damage. Name likely culprits and list next steps to contain them.",
  },
  {
    id: "faq_trichomes",
    title: "Trichome timing",
    blurb: "Harvest window via trichome mix.",
    prompt: "Describe trichome mix (clear/milky/amber) and estimate a harvest window.",
  },
  {
    id: "faq_cleanliness",
    title: "Cleanliness check",
    blurb: "Mold risk, bud handling, wash suggestions.",
    prompt: "Check for mold/rot risk, handling artifacts, and recommend any bud wash or drying adjustments.",
  },
  {
    id: "faq_vpd",
    title: "VPD & climate",
    blurb: "Target humidity/temperature and VPD alignment.",
    prompt:
      "Review vapor pressure deficit alignment using the provided humidity and temperature. Flag if VPD is out of range for the current growth stage and suggest humidity or temperature tweaks to get back on track.",
  },
  {
    id: "faq_ripeness",
    title: "Ripeness check",
    blurb: "Flower maturity and fade cues.",
    prompt:
      "Assess flower ripeness and fade cues beyond trichomes: pistil color, leaf fade patterns, bud density, and any late-stage stress. Provide a harvest readiness note.",
  },
  {
    id: "faq_training",
    title: "Canopy & training",
    blurb: "Node spacing, airflow, light spread.",
    prompt:
      "Evaluate canopy shape, node spacing, airflow lanes, and light distribution. Recommend low-stress training, defoliation, or trellising actions to optimize coverage without stressing the plant.",
  },
  {
    id: "faq_feeding",
    title: "Feeding & EC/PH",
    blurb: "Nutrient strength and pH context.",
    prompt:
      "Use the EC and pH readings (if provided) together with visible leaf cues to decide if feeding strength is appropriate. Flag over/underfeeding signs and suggest small EC or pH adjustments.",
  },
];

const STORAGE_KEYS = {
  THEME: "opencore_theme",
  TEMPLATES: "opencore_custom_templates",
  API_SETTINGS: "opencore_settings",
  ACTIVE_LLM_PROFILE: "opencore_active_llm_profile",
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
  llmProfiles: [],
  activeLlmProfileId: null,
  selectedLlmProfileId: null,
  uploads: [],
  mqttConfig: { broker: "", port: 1883, username: "", password: "", use_tls: false, sensors: [] },
  mqttValues: [],
  builderMode: "blocks",
};

const dom = {};

function on(element, event, handler) {
  if (element) element.addEventListener(event, handler);
}

document.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  initTheme();
  initTemplates();
  initPromptBuilder();
  bindEvents();
  loadModels();
  loadLlmProfiles();
  loadUploads();
  loadMqttConfig();
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
  dom.llmProfileSelect = document.getElementById("llmProfileSelect");
  dom.streamLlmProfile = document.getElementById("streamLlmProfile");
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
  dom.videoInput = document.getElementById("videoInput");
  dom.videoUploadSelect = document.getElementById("videoUploadSelect");
  dom.videoAnalysisMode = document.getElementById("videoAnalysisMode");
  dom.videoLlmProfile = document.getElementById("videoLlmProfile");
  dom.videoPrompt = document.getElementById("videoPrompt");
  dom.analyzeVideoBtn = document.getElementById("analyzeVideoBtn");
  dom.videoStatus = document.getElementById("videoStatus");
  dom.videoOverlayPreview = document.getElementById("videoOverlayPreview");
  dom.libraryInput = document.getElementById("libraryInput");
  dom.libraryUploadBtn = document.getElementById("libraryUploadBtn");
  dom.uploadSort = document.getElementById("uploadSort");
  dom.uploadList = document.getElementById("uploadList");
  dom.sensorChipContainer = document.getElementById("sensorChipContainer");
  dom.sensorStatus = document.getElementById("sensorStatus");
  dom.refreshSensors = document.getElementById("refreshSensors");
  dom.faqBlockList = document.getElementById("faqBlockList");
  dom.builderModeButtons = document.querySelectorAll("[data-builder-mode]");
  dom.builderFreePanel = document.getElementById("builderFreePanel");
  dom.builderQuickQuestion = document.getElementById("builderQuickQuestion");
  dom.addQuickQuestion = document.getElementById("addQuickQuestion");
  dom.clearPromptBtn = document.getElementById("clearPromptBtn");
  dom.builderDropZone = document.getElementById("builderDropZone");
}

function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEYS.THEME);
  state.theme = saved === "light" ? "light" : "dark";
  applyTheme();
  if (dom.themeToggle) {
    dom.themeToggle.addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      applyTheme();
      localStorage.setItem(STORAGE_KEYS.THEME, state.theme);
    });
  }
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
    dom.templateSelect.innerHTML = '<option value="">Select template …</option>';
  PROMPT_TEMPLATES.forEach((tpl) => {
    const option = document.createElement("option");
    option.value = tpl.id;
    option.textContent = `${tpl.label} — ${tpl.description}`;
    dom.templateSelect.appendChild(option);
  });
  if (state.customTemplates.length) {
    const divider = document.createElement("option");
    divider.disabled = true;
      divider.textContent = "──── Custom templates ────";
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
    removeBtn.textContent = "Delete";
    removeBtn.addEventListener("click", () => removeCustomTemplate(tpl.id));
    row.append(info, removeBtn);
    dom.templateList.appendChild(row);
  });
}

function initPromptBuilder() {
  renderFaqBlocks();
  if (dom.builderModeButtons) {
    dom.builderModeButtons.forEach((btn) => {
      btn.addEventListener("click", () => switchBuilderMode(btn.dataset.builderMode));
    });
  }
  if (dom.builderDropZone) {
    dom.builderDropZone.addEventListener("dragover", handleBuilderDragOver);
    dom.builderDropZone.addEventListener("dragleave", handleBuilderDragLeave);
    dom.builderDropZone.addEventListener("drop", handleBuilderDrop);
  }
  on(dom.addQuickQuestion, "click", insertQuickQuestion);
  on(dom.clearPromptBtn, "click", () => {
    if (dom.prompt) dom.prompt.value = "";
  });
  switchBuilderMode(state.builderMode);
}

function renderFaqBlocks() {
  if (!dom.faqBlockList) return;
  dom.faqBlockList.innerHTML = "";
  FAQ_BLOCKS.forEach((block) => {
    const tile = document.createElement("div");
    tile.className = "builder-block";
    tile.draggable = true;
    tile.dataset.text = block.prompt;
    tile.innerHTML = `<h4>${block.title}</h4><p>${block.blurb}</p>`;
    tile.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", tile.dataset.text);
      event.dataTransfer.setData("opencore-kind", "faq");
    });
    tile.addEventListener("click", () => insertPromptText(block.prompt));
    dom.faqBlockList.appendChild(tile);
  });
}

function switchBuilderMode(mode) {
  state.builderMode = mode === "free" ? "free" : "blocks";
  if (dom.builderModeButtons) {
    dom.builderModeButtons.forEach((btn) => {
      const isActive = btn.dataset.builderMode === state.builderMode;
      btn.classList.toggle("active", isActive);
    });
  }
  if (dom.builderFreePanel) {
    dom.builderFreePanel.style.display = state.builderMode === "free" ? "block" : "none";
  }
}

function insertQuickQuestion() {
  if (!dom.builderQuickQuestion) return;
  const text = dom.builderQuickQuestion.value.trim();
  if (!text) {
    showToast("Add a question first.");
    return;
  }
  insertPromptText(text);
  dom.builderQuickQuestion.value = "";
}

function bindEvents() {
  on(dom.templateSelect, "change", handleTemplateSelect);
  on(dom.saveTemplateBtn, "click", () => openModal(dom.templateModal));
  on(dom.cancelTemplate, "click", () => closeModal(dom.templateModal));
  on(dom.confirmTemplate, "click", saveCustomTemplate);
  on(dom.form, "submit", handleAnalyzeSubmit);
  on(dom.imageInput, "change", handleFileChange);
  on(dom.debugToggle, "change", () => {
    state.debugEnabled = dom.debugToggle.checked;
  });
  on(dom.apiSettingsBtn, "click", openApiModal);
  on(dom.closeApiModal, "click", () => closeModal(dom.apiModal));
  on(dom.saveApiSettings, "click", saveApiSettings);
  on(dom.resetApiSettings, "click", resetApiSettings);
  on(dom.jsonFullscreenBtn, "click", () => openJsonFullscreen());
  on(dom.closeJsonFullscreen, "click", () => closeModal(dom.jsonFullscreen));
  on(dom.jsonDownloadBtn, "click", downloadJsonReport);
  on(dom.pdfExportBtn, "click", generatePdfReport);
  on(dom.shareBtn, "click", shareReport);
  on(dom.closeImageModal, "click", () => closeModal(dom.imageModal));
  on(dom.zoomSlider, "input", handleZoomChange);
  on(dom.analysisMode, "change", handleAnalysisModeChange);
  on(dom.llmProfileSelect, "change", handleLlmProfileChange);
  on(dom.streamLlmProfile, "change", handleLlmProfileChange);
  on(dom.streamForm, "submit", handleStreamSubmit);
  on(dom.streamRefresh, "click", loadStreams);
  on(dom.analyzeVideoBtn, "click", handleAnalyzeVideo);
  on(dom.libraryUploadBtn, "click", handleLibraryUpload);
  on(dom.uploadSort, "change", () => loadUploads(dom.uploadSort.value));
  on(dom.refreshSensors, "click", refreshMqttValues);
  if (dom.prompt) {
    dom.prompt.addEventListener("dragover", (event) => event.preventDefault());
    dom.prompt.addEventListener("drop", handlePromptDrop);
  }
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
  if (debug && dom.debugToggle && ["1", "true"].includes(debug.toLowerCase())) {
    dom.debugToggle.checked = true;
    state.debugEnabled = true;
  }
}

function handleAnalysisModeChange() {
  if (!dom.analysisMode) return;
  state.analysisMode = dom.analysisMode.value;
  const isMl = state.analysisMode === "ml";
  if (dom.analysisHint) {
    dom.analysisHint.textContent = isMl
      ? "ML-only does not require a prompt."
      : "Hybrid combines ML + GPT and requires a prompt.";
  }
  if (dom.prompt) {
    dom.prompt.required = !isMl;
    dom.prompt.placeholder = isMl ? "Optional note for the ML log" : "Describe the desired analysis";
  }
  if (isMl) {
    setMlDebugMessage(["ML mode active. Models initialize on the next run."]);
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

async function loadMqttConfig() {
  try {
    const payload = await apiRequest("/api/mqtt/config");
    state.mqttConfig = payload.config || state.mqttConfig;
    renderSensorChips();
  } catch (error) {
    setSensorStatus(`MQTT settings unavailable: ${error.message}`, true);
  }
}

async function refreshMqttValues() {
  try {
    setSensorStatus("Polling broker …");
    const payload = await apiRequest("/api/mqtt/poll", { method: "POST" });
    state.mqttValues = payload.values || [];
    renderSensorChips();
    setSensorStatus(`Received ${payload.count || state.mqttValues.length} readings.`);
  } catch (error) {
    setSensorStatus(error.message || "MQTT poll failed.", true);
  }
}

function setSensorStatus(message, isError = false) {
  if (!dom.sensorStatus) return;
  dom.sensorStatus.textContent = message;
  dom.sensorStatus.style.color = isError ? "#ff9393" : "var(--muted)";
}

function renderSensorChips() {
  if (!dom.sensorChipContainer) return;
  dom.sensorChipContainer.innerHTML = "";
  const valuesById = {};
  state.mqttValues.forEach((entry) => {
    if (entry.id) valuesById[entry.id] = entry;
  });
  if (!state.mqttConfig.sensors || !state.mqttConfig.sensors.length) {
    setSensorStatus("Add MQTT sensors in the config hub.");
    return;
  }
  state.mqttConfig.sensors.forEach((sensor) => {
    const valueEntry = valuesById[sensor.id] || {};
    const token = buildSensorToken(sensor, valueEntry);
    const chip = document.createElement("div");
    chip.className = "sensor-chip";
    chip.draggable = true;
    chip.dataset.text = token;
    chip.innerHTML = `<strong>${sensor.label || sensor.topic}</strong><span>${
      valueEntry.value ?? "No recent value"
    }${sensor.unit ? " " + sensor.unit : ""}</span>`;
    chip.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", chip.dataset.text);
      event.dataTransfer.setData("opencore-kind", "sensor");
    });
    chip.addEventListener("click", () => insertSensorSnippet(chip.dataset.text));
    dom.sensorChipContainer.appendChild(chip);
  });
}

function buildSensorToken(sensor, valueEntry) {
  const label = sensor.label || sensor.topic || sensor.id || "sensor";
  const valueText = valueEntry.value ?? "<value pending>";
  const unitText = sensor.unit ? ` ${sensor.unit}` : "";
  return `【${label}: ${valueText}${unitText}】`;
}

function handlePromptDrop(event) {
  event.preventDefault();
  const text = event.dataTransfer.getData("text/plain");
  const kind = event.dataTransfer.getData("opencore-kind");
  if (!text) return;
  if (kind === "sensor") {
    insertSensorSnippet(text);
  } else {
    insertPromptText(text);
  }
}

function insertSensorSnippet(text) {
  insertPromptText(text, true);
}

function insertPromptText(text, wrap = false) {
  if (!dom.prompt || !text) return;
  const current = dom.prompt.value;
  const separator = current && !current.endsWith("\n") ? "\n" : "";
  const value = wrap ? (text.startsWith("【") ? text : `【${text}】`) : text;
  dom.prompt.value = `${current}${separator}${value}`;
  dom.prompt.focus();
}

function handleBuilderDragOver(event) {
  event.preventDefault();
  if (dom.builderDropZone) dom.builderDropZone.classList.add("active");
}

function handleBuilderDragLeave(event) {
  event.preventDefault();
  if (dom.builderDropZone) dom.builderDropZone.classList.remove("active");
}

function handleBuilderDrop(event) {
  event.preventDefault();
  if (dom.builderDropZone) dom.builderDropZone.classList.remove("active");
  const text = event.dataTransfer.getData("text/plain");
  const kind = event.dataTransfer.getData("opencore-kind");
  if (!text) return;
  if (kind === "sensor") {
    insertSensorSnippet(text);
  } else {
    insertPromptText(text);
  }
  showBuilderDropMessage(text);
}

function showBuilderDropMessage(text) {
  if (!dom.builderDropZone) return;
  dom.builderDropZone.innerHTML = "";
  const label = document.createElement("div");
  label.textContent = "Added to prompt:";
  const code = document.createElement("code");
  code.textContent = text;
  dom.builderDropZone.append(label, code);
  setTimeout(() => {
    if (dom.builderDropZone) {
      dom.builderDropZone.textContent = "Drop FAQ blocks or sensors here to append to the prompt.";
    }
  }, 2500);
}

function saveCustomTemplate() {
  const label = dom.templateName.value.trim();
  const description = dom.templateDescription.value.trim();
  const prompt = dom.prompt.value.trim();
  if (!label || !prompt) {
      showToast("Name and prompt are required.");
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
    showToast("Template saved.");
}

function persistCustomTemplates() {
  localStorage.setItem(STORAGE_KEYS.TEMPLATES, JSON.stringify(state.customTemplates));
}

function removeCustomTemplate(id) {
  state.customTemplates = state.customTemplates.filter((tpl) => tpl.id !== id);
  persistCustomTemplates();
  renderTemplateOptions();
  renderCustomTemplateList();
    showToast("Template removed.");
}

function setupDropzone() {
  if (!dom.dropzone) return;
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
  if (!event?.target) return;
  const files = Array.from(event.target.files || []);
  state.files = files;
  updatePreviewGrid();
}

function updatePreviewGrid() {
  if (!dom.previewContainer) return;
  dom.previewContainer.innerHTML = "";
  if (!state.files.length) {
    if (dom.fileStatus) dom.fileStatus.textContent = "No files selected.";
    return;
  }
  if (dom.fileStatus) dom.fileStatus.textContent = `${state.files.length} file(s) selected.`;
  state.files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const item = document.createElement("div");
      item.className = "preview-item";
        item.innerHTML = `<img src="${event.target.result}" alt="Preview" /><small>${file.name || `Image ${index + 1}`}</small>`;
      item.addEventListener("click", () => openImageModal(event.target.result));
      dom.previewContainer.appendChild(item);
    };
    reader.readAsDataURL(file);
  });
}

function openImageModal(src) {
  if (!dom.modalImage || !dom.imageModal || !dom.zoomSlider) return;
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
    dom.modelSelect.innerHTML = "<option>Loading models …</option>";
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
        dom.modelStatus.textContent = "The selection controls the active Teachable Machine package.";
    })
    .catch((error) => {
        dom.modelStatus.textContent = `Models could not load (${error.message}).`;
    })
    .finally(() => {
      dom.modelSelect.disabled = false;
    });
}

function populateModelSelect(selectEl, models, hasBuiltin, defaultId, optional = false) {
  if (!selectEl) return;
    selectEl.innerHTML = optional ? '<option value="">Use default</option>' : "";
  if (hasBuiltin) {
    const option = document.createElement("option");
    option.value = "builtin";
      option.textContent = "OPENCORE reference (TEACHABLE_MODEL_PATH)";
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
      showToast("Prompt is required.");
    return;
  }
  if (!state.files.length) {
      showToast("Please add at least one image.");
    return;
  }
  if (state.analysisMode === "ml") {
      setMlDebugMessage(["Initializing model …", `Uploads: ${state.files.length}`]);
  }
    dom.runStatus.textContent = "Running analysis …";
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
  if (state.selectedLlmProfileId) {
    formData.append("llm_profile_id", state.selectedLlmProfileId);
  }
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
      dom.runStatus.textContent = `Done (${elapsed} ms)`;
  } catch (error) {
      dom.runStatus.textContent = "Error";
      dom.resultDisplay.innerHTML = `<p class="disclaimer">Error: ${error.message}</p>`;
    dom.resultJson.textContent = JSON.stringify({ status: "error", message: error.message }, null, 2);
    dom.debugPanel.classList.add("active");
      dom.debugPanel.textContent = `Client error: ${error.message}`;
    if (state.analysisMode === "ml") {
        setMlDebugMessage([`Error: ${error.message}`]);
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
      summary: { text: payload.summary?.text || payload.gpt_response || "Report created." },
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
    const tabs = [{ key: "summary", label: "Summary report" }];
  result.items.forEach((item, index) => {
      tabs.push({ key: `item-${index}`, label: item.image_id || `Image ${index + 1}` });
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
    : `<p class="disclaimer">ML-only: no LLM text for this image.</p>`;
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
  const llmProfile = debug?.llm_profile_id || state.selectedLlmProfileId || state.activeLlmProfileId || "Server-Default";
  dom.debugPanel.innerHTML = `
    <strong>Request-ID:</strong> ${debug?.request_id || "n/a"}<br />
    <strong>Modell:</strong> ${debug?.model_name || "OPENCORE"} (${debug?.model_version || ""})<br />
    <strong>LLM-Profil:</strong> ${llmProfile}<br />
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
  if (dom.resultDisplay) {
    dom.resultDisplay.innerHTML = `<p class="disclaimer">No analysis yet.</p>`;
  }
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
    llm_profile_id: state.selectedLlmProfileId || undefined,
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
  if (!dom.streamList && !dom.streamStatus) return;
  try {
    if (dom.streamStatus) dom.streamStatus.textContent = "Loading streams …";
    const payload = await apiRequest("/api/opencore/streams");
    state.streams = payload.streams || [];
    renderStreamList();
    if (dom.streamStatus) {
      dom.streamStatus.textContent = state.streams.length
        ? `${state.streams.length} stream(s) active.`
        : "No streams registered yet.";
    }
  } catch (error) {
    if (dom.streamStatus) {
      dom.streamStatus.textContent = `Streams could not load (${error.message}).`;
    }
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
    const overlay = stream.overlay_available
      ? `<div class="stream-overlay"><img alt="Overlay" src="/api/opencore/streams/${stream.stream_id}/overlay?ts=${Date.now()}" /></div>`
      : "";
    item.innerHTML = `
      <header>
        <strong>${stream.label || stream.stream_id}</strong>
        <small>${stream.analysis_mode?.toUpperCase() || "HYBRID"}</small>
      </header>
      <p class="disclaimer">${summary}</p>
      ${overlay}
      ${errorHtml}
      <small class="disclaimer">LLM-Profil: ${stream.llm_profile_id || state.activeLlmProfileId || "Server-Default"}</small>
      <small class="disclaimer">Letztes Capture: ${stream.last_capture_ts ? new Date(stream.last_capture_ts * 1000).toLocaleTimeString() : "-"}</small>
    `;
    const footer = document.createElement("footer");
    const viewBtn = document.createElement("button");
    viewBtn.textContent = "JSON";
    viewBtn.addEventListener("click", () => {
      if (!stream.last_result) {
        showToast("No result for this stream yet.");
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

async function loadUploads(sort = dom.uploadSort?.value || "recent") {
  // Fetch the upload registry and refresh UI lists.
  if (!dom.uploadList) return;
  try {
    const payload = await apiRequest(`/api/uploads?sort=${sort}`);
    state.uploads = payload.uploads || [];
    renderUploadList();
    renderVideoUploadSelect();
  } catch (error) {
    dom.uploadList.innerHTML = `<p class="disclaimer">Uploads could not be loaded (${error.message}).</p>`;
  }
}

function renderUploadList() {
  // Render the upload table with analyze/rename/delete actions.
  if (!dom.uploadList) return;
  dom.uploadList.innerHTML = "";
  if (!state.uploads.length) {
    dom.uploadList.innerHTML = '<p class="disclaimer">No uploads yet.</p>';
    return;
  }
  state.uploads.forEach((upload) => {
    const row = document.createElement("div");
    row.className = "upload-row";
    row.innerHTML = `
      <div>
        <strong>${upload.name}</strong>
        <small class="disclaimer">${upload.type} · ${upload.created ? new Date(upload.created).toLocaleString() : ""}</small>
      </div>
      <div class="action-row" style="gap:0.5rem">
        <button data-id="${upload.id}" class="pill-button" data-action="analyze">Analyze</button>
        <button data-id="${upload.id}" class="pill-button" data-action="rename">Rename</button>
        <button data-id="${upload.id}" class="pill-button" data-action="delete">Delete</button>
      </div>
    `;
    row.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => handleUploadAction(btn.dataset.action, btn.dataset.id));
    });
    dom.uploadList.appendChild(row);
  });
}

function renderVideoUploadSelect() {
  // Populate the video dropdown with stored uploads.
  if (!dom.videoUploadSelect) return;
  dom.videoUploadSelect.innerHTML = '<option value="">Use uploaded video …</option>';
  state.uploads
    .filter((item) => item.type === "video")
    .forEach((item) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = item.name;
      dom.videoUploadSelect.appendChild(option);
    });
}

async function handleLibraryUpload() {
  // Push one or more media files into the upload registry.
  const files = dom.libraryInput?.files || [];
  if (!files.length) {
    showToast("Select a file to upload.");
    return;
  }
  for (const file of files) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("kind", "auto");
    formData.append("label", file.name);
    try {
      await apiRequest("/api/uploads", { method: "POST", body: formData });
    } catch (error) {
      showToast(`Upload failed: ${error.message}`);
    }
  }
  showToast("Upload(s) stored.");
  loadUploads();
  dom.libraryInput.value = "";
}

async function handleUploadAction(action, uploadId) {
  // Dispatch a row action (analyze, rename, delete) for a stored upload.
  const entry = state.uploads.find((u) => u.id === uploadId);
  if (!entry) return;
  if (action === "delete") {
    await apiRequest(`/api/uploads/${uploadId}`, { method: "DELETE" });
    showToast("Upload deleted.");
    loadUploads();
    return;
  }
  if (action === "rename") {
    const newName = prompt("New name", entry.name || "");
    if (!newName) return;
    const formData = new FormData();
    formData.append("new_name", newName);
    await apiRequest(`/api/uploads/${uploadId}/rename`, { method: "POST", body: formData });
    showToast("Upload renamed.");
    loadUploads();
    return;
  }
  if (action === "analyze") {
    if (entry.type === "video") {
      dom.videoUploadSelect.value = uploadId;
      handleAnalyzeVideo();
    } else {
      analyzeUploadImage(uploadId);
    }
  }
}

async function analyzeUploadImage(uploadId) {
  // Run the standard image analysis against a stored upload id.
  const formData = new FormData();
  formData.append("upload_id", uploadId);
  formData.append("prompt", dom.prompt.value || "");
  formData.append("model_id", dom.modelSelect.value || "");
  formData.append("analysis_mode", dom.analysisMode.value || "hybrid");
  if (dom.llmProfileSelect?.value) {
    formData.append("llm_profile_id", dom.llmProfileSelect.value);
  }
  const start = performance.now();
  try {
    const result = await apiRequest("/analyze", { method: "POST", body: formData });
    handleResult(result, performance.now() - start);
    showToast("Analysis complete.");
  } catch (error) {
    showToast(`Analysis failed: ${error.message}`);
  }
}

async function handleAnalyzeVideo() {
  // Run the video analyzer for an uploaded MP4 or a stored upload id.
  const formData = new FormData();
  const file = dom.videoInput?.files?.[0];
  const selectedUpload = dom.videoUploadSelect?.value;
  if (file) {
    formData.append("file", file);
  } else if (selectedUpload) {
    formData.append("upload_id", selectedUpload);
  } else {
    showToast("Select a video file or library entry.");
    return;
  }
  formData.append("prompt", dom.videoPrompt?.value || "");
  formData.append("model_id", dom.modelSelect.value || "");
  formData.append("analysis_mode", dom.videoAnalysisMode?.value || "hybrid");
  if (dom.videoLlmProfile?.value) {
    formData.append("llm_profile_id", dom.videoLlmProfile.value);
  }
  dom.videoStatus.textContent = "Running …";
  const start = performance.now();
  try {
    const result = await apiRequest("/api/opencore/analyze-video", { method: "POST", body: formData });
    handleResult(result, performance.now() - start);
    if (result.overlay_video && dom.videoOverlayPreview) {
      dom.videoOverlayPreview.src = result.overlay_video.startsWith("http")
        ? result.overlay_video
        : `/` + result.overlay_video.replace(/^\//, "");
      dom.videoOverlayPreview.classList.remove("hidden");
    }
    dom.videoStatus.textContent = "Video processed.";
    loadUploads();
  } catch (error) {
    dom.videoStatus.textContent = `Failed: ${error.message}`;
    showToast(`Video analysis failed: ${error.message}`);
  }
}

async function loadLlmProfiles() {
  if (!dom.llmProfileSelect) return;
  const stored = localStorage.getItem(STORAGE_KEYS.ACTIVE_LLM_PROFILE);
  if (!state.selectedLlmProfileId && stored !== null) {
    state.selectedLlmProfileId = stored || "";
  }
  try {
    const payload = await apiRequest("/api/settings/llm/profiles");
    state.llmProfiles = payload.profiles || [];
    state.activeLlmProfileId = payload.active_profile_id || null;
    if (!state.selectedLlmProfileId) {
      state.selectedLlmProfileId = state.activeLlmProfileId || "";
    }
    renderLlmProfiles();
  } catch (error) {
    showToast(`LLM-Profile konnten nicht geladen werden: ${error.message}`);
  }
}

function renderLlmProfiles() {
  const selects = [dom.llmProfileSelect, dom.streamLlmProfile, dom.videoLlmProfile].filter(Boolean);
  selects.forEach((select) => {
    select.innerHTML = '<option value="">Use active server profile</option>';
    state.llmProfiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      const provider = (profile.config && profile.config.provider) || "openai";
      option.textContent = `${profile.name} (${provider})`;
      select.appendChild(option);
    });
    if (state.selectedLlmProfileId !== null) {
      select.value = state.selectedLlmProfileId || "";
    }
  });
}

function handleLlmProfileChange(event) {
  state.selectedLlmProfileId = event.target.value || "";
  if (state.selectedLlmProfileId) {
    localStorage.setItem(STORAGE_KEYS.ACTIVE_LLM_PROFILE, state.selectedLlmProfileId);
  } else {
    localStorage.removeItem(STORAGE_KEYS.ACTIVE_LLM_PROFILE);
  }
  renderLlmProfiles();
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
    showToast("Analysis triggered.");
    loadStreams();
  } catch (error) {
    showToast(`Trigger fehlgeschlagen: ${error.message}`);
  }
}
