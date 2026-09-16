const form = document.getElementById("intake");
const errorBox = document.getElementById("error");
const runButton = form.querySelector(".run");

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
const OPTIONAL = new Set(["HbA1c", "Smoking", "Diabetes",
                          "Medical_History_Hypertension"]);
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
  verdictLabel.textContent = data.verdict;
  verdictSub.textContent = data.pad_detected
    ? `Scored above the screening threshold of ${data.threshold}`
    : `Scored below the screening threshold of ${data.threshold}`;

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

  const percent = data.confidence * 100;
  confidence.textContent = `${percent.toFixed(1)}%`;
  meterFill.style.width = `${percent}%`;
  const chance = (data.probability * 100).toFixed(1);
  confidenceNote.textContent = data.pad_detected
    ? `How far above the threshold this record scored. Ensemble probability of PAD: ${chance}%.`
    : `How far below the threshold this record scored. Ensemble probability of PAD: ${chance}%.`;

  if (window.matchMedia("(max-width: 900px)").matches) {
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

form.addEventListener("input", (event) => {
  if (event.target.name === "Total_Cholesterol" || event.target.name === "HDL") {
    updateRatio();
  }
});

form.addEventListener("reset", () => {
  placeholder.hidden = false;
  result.hidden = true;
  errorBox.hidden = true;
  setTimeout(updateRatio);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;

  if (!form.reportValidity()) return;

  runButton.disabled = true;
  runButton.textContent = "Scoring";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(describeProblem(data.detail));
    render(data);
  } catch (problem) {
    errorBox.textContent = problem.message;
    errorBox.hidden = false;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run screening";
  }
});

updateRatio();
