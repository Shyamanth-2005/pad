const form = document.getElementById("intake");
const errorBox = document.getElementById("error");
const sensorForm = document.getElementById("sensor-intake");
const sensorFields = [...sensorForm.querySelectorAll("input")];
const clearButton = document.getElementById("clear");

const placeholder = document.getElementById("placeholder");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const verdictLabel = document.getElementById("verdict-label");
const verdictSub = document.getElementById("verdict-sub");
const riskLabel = document.getElementById("risk-label");
const riskDetail = document.getElementById("risk-detail");
const needle = document.getElementById("needle");
const needleValue = document.getElementById("needle-value");
const confidence = document.getElementById("confidence");
const meterFill = document.getElementById("meter-fill");
const confidenceNote = document.getElementById("confidence-note");
const ratio = document.getElementById("ratio");

// Fields the pipelines impute when absent. Blank means "not known", not zero.
const OPTIONAL = new Set(["HbA1c"]);
const NUMERIC = new Set(["Age", "BMI", "HbA1c", "SBP", "DBP",
                         "Total_Cholesterol", "HDL", "ABI"]);

function updateRatio() {
  const tc = parseFloat(form.Total_Cholesterol.value);
  const hdl = parseFloat(form.HDL.value);
  ratio.textContent = tc > 0 && hdl > 0 ? (tc / hdl).toFixed(2) : "—";
}

function collect() {
  const payload = {};
  for (const [name, raw] of new FormData(form).entries()) {
    const value = raw.trim();
    if (value === "") {
      if (OPTIONAL.has(name)) payload[name] = null;
      continue;
    }
    payload[name] = NUMERIC.has(name) ? Number(value) : value;
  }
  return payload;
}

// The form asks for everyday units; the model takes some on a log or 0-1 scale.
// The model's daily log values are averaged over days, so the log of a weekly
// average is a close approximation, not an exact match.
const fromPercent = (value) => value / 100;
const SENSOR_UNITS = {
  Counts_Per_Day: ["Log_Total_Counts", Math.log1p],
  Sedentary_Percent: ["Sedentary_Share", fromPercent],
  ASTP_Percent: ["ASTP", fromPercent],
  SATP_Percent: ["SATP", fromPercent],
  PM_AM_Ratio: ["Log_PM_AM_Ratio", Math.log],
};

// Every sensor field is a number; the API takes them as one nested object.
function collectSensor() {
  const summaries = {};
  for (const [name, raw] of new FormData(sensorForm).entries()) {
    const [modelName, convert] = SENSOR_UNITS[name] ?? [name, (value) => value];
    summaries[modelName] = convert(Number(raw));
  }
  return summaries;
}

// A blank Sensor tab means no recording, so the record is scored on clinical
// data alone. Once any field is filled all ten become required: the sensor
// model has no imputation, so a partial recording is an error, not a recording.
function syncSensorRequired() {
  const entered = sensorFields.some((input) => input.value.trim() !== "");
  for (const input of sensorFields) input.required = entered;
  return entered;
}

function asPercent(fraction) {
  return `${(fraction * 100).toFixed(1)}%`;
}

function describeProblem(detail) {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "Could not score this record.";
  return detail
    .map((item) => `${item.loc[item.loc.length - 1]}: ${item.msg}`)
    .join("; ");
}

function render(data) {
  placeholder.hidden = true;
  result.hidden = false;

  verdict.classList.toggle("is-detected", data.pad_detected);
  verdict.classList.toggle("is-clear", !data.pad_detected);
  const fused = data.modalities.includes("sensor");
  verdictLabel.textContent = data.verdict;
  const basis = fused ? "Clinical and sensor data" : "Clinical data only";
  const cutoff = asPercent(data.threshold);
  verdictSub.textContent = data.pad_detected
    ? `${basis}. Scored above the ${cutoff} screening threshold.`
    : `${basis}. Scored below the ${cutoff} screening threshold.`;

  riskLabel.textContent = data.risk.label;
  riskLabel.className = `risk-label band-${data.risk.band}`;
  riskDetail.textContent = data.risk.detail;

  needle.hidden = false;
  needle.style.setProperty("--at", Math.min(data.risk.value, 1.6));
  needleValue.textContent = data.risk.value.toFixed(2);

  for (const band of document.querySelectorAll(".scale-band")) {
    band.classList.toggle("is-active", band.dataset.band === data.risk.band);
  }
  for (const row of document.querySelectorAll(".legend li")) {
    row.classList.toggle("is-active", row.dataset.legend === data.risk.band);
  }

  confidence.textContent = asPercent(data.confidence);
  meterFill.style.width = `${data.confidence * 100}%`;
  const side = data.pad_detected ? "above" : "below";
  confidenceNote.textContent = `How far ${side} the threshold this record scored. ` +
    `Combined probability of PAD: ${asPercent(data.probability)}.`;

  if (window.matchMedia("(max-width: 900px)").matches) {
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function clearResult() {
  placeholder.hidden = false;
  result.hidden = true;
}

async function score(payload, button, problemBox) {
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Scoring";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(describeProblem(data.detail));
    render(data);
  } catch (problem) {
    problemBox.textContent = problem.message;
    problemBox.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

form.addEventListener("input", (event) => {
  if (event.target.name === "Total_Cholesterol" || event.target.name === "HDL") {
    updateRatio();
  }
});

sensorForm.addEventListener("input", syncSensorRequired);

clearButton.addEventListener("click", () => {
  form.reset();
  sensorForm.reset();
  syncSensorRequired();
  clearResult();
  errorBox.hidden = true;
  setTimeout(updateRatio);
});

// The one Run button submits the clinical form from any tab. A gap in either
// form sends the user to that tab with the browser pointing at the field.
form.addEventListener("submit", (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  if (!form.checkValidity()) {
    selectTab(document.getElementById("tab-clinical"));
    form.reportValidity();
    return;
  }
  const hasSensor = syncSensorRequired();
  if (!sensorForm.checkValidity()) {
    selectTab(document.getElementById("tab-sensor"));
    sensorForm.reportValidity();
    return;
  }
  const payload = collect();
  if (hasSensor) payload.Accelerometry = collectSensor();
  score(payload, event.submitter, errorBox);
});

// ---------------------------------------------------------------- tabs

const tabs = [...document.querySelectorAll('[role="tab"]')];

function selectTab(tab) {
  for (const other of tabs) {
    const chosen = other === tab;
    other.setAttribute("aria-selected", String(chosen));
    other.tabIndex = chosen ? 0 : -1;
    document.getElementById(other.getAttribute("aria-controls")).hidden = !chosen;
  }
  const panel = document.getElementById(tab.getAttribute("aria-controls"));
  document.querySelector(".workspace").classList.toggle("on-stub", panel.classList.contains("stub"));
  tab.focus();
}

for (const tab of tabs) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, Home: -tabs.length, End: tabs.length };
    if (!(event.key in step)) return;
    event.preventDefault();
    const next = Math.min(Math.max(tabs.indexOf(tab) + step[event.key], 0), tabs.length - 1);
    selectTab(tabs[next]);
  });
}

updateRatio();
