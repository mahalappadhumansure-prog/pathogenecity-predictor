
const $ = (id) => document.getElementById(id);
let examples = {};

async function init() {
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) throw new Error("Could not load DNA examples.");

    examples = await response.json();

    const sel = $("example");
    sel.innerHTML = "";

    Object.keys(examples).forEach((name) => {
      sel.add(new Option(name, name));
    });

    sel.onchange = () => fillExample(sel.value);

    if (sel.options.length > 0) {
      sel.selectedIndex = 0;
      fillExample(sel.value);
    }

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.onclick = () => {
        document.querySelectorAll(".tab, .tabpane")
          .forEach((el) => el.classList.remove("active"));

        tab.classList.add("active");
        $(tab.dataset.tab)?.classList.add("active");
      };
    });

    $("predictBtn").onclick = predict;

    try {
      const infoResponse = await fetch("/api/model-info");
      if (infoResponse.ok) {
        const info = await infoResponse.json();
        const m = info.metrics || {};

        $("perfView").innerHTML = `
          <div class="metric"><b>${m.accuracy ?? "—"}</b>Accuracy</div>
          <div class="metric"><b>${m.roc_auc ?? "—"}</b>ROC-AUC</div>
          <div class="metric"><b>${m.n_train ?? "—"}</b>Training samples</div>
          <div class="metric"><b>${m.n_test ?? "—"}</b>Test samples</div>
          <p class="msg">${m.note || m.source || ""}</p>`;

        const importance = Object.entries(
          info.global_importance || {}
        ).map(([feature, value]) => ({ feature, value }));

        renderBars(
          $("impBars"),
          importance.sort((a, b) => b.value - a.value),
          false
        );
      }
    } catch (error) {
      console.error("Model information error:", error);
    }

    await predict();

  } catch (error) {
    console.error("Initialization error:", error);
    $("error").textContent =
      "Could not load the DNA examples. Check the Flask server and browser console.";
    $("error").hidden = false;
  }
}

function fillExample(name) {
  const seq = examples[name];
  if (!seq) return;

  $("seq").value =
    ">" + name.replace(/\s+/g, "_") + "\n" +
    (seq.match(/.{1,60}/g) || []).join("\n");
}

function renderBars(el, items, signed = true) {
  if (!items || items.length === 0) {
    el.innerHTML = "<p>No feature contributions available.</p>";
    return;
  }

  const max = Math.max(
    ...items.map((item) => Math.abs(Number(item.value) || 0)),
    0.01
  );

  el.innerHTML = items.map((item) => {
    const value = Number(item.value) || 0;
    const width = Math.abs(value) / max * 100;
    const cls = signed && value > 0 ? "fill pos" : "fill";
    const label = (signed && value > 0 ? "+" : "") + value.toFixed(3);

    return `
      <div class="hb">
        <span class="name">${item.feature}</span>
        <div class="track">
          <div class="${cls}" style="width:${width}%"></div>
        </div>
        <span>${label}</span>
      </div>`;
  }).join("");
}

async function predict() {
  $("error").hidden = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: $("seq").value })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    render(data);
  } catch (error) {
    console.error("Prediction error:", error);
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("seqBox").textContent =
      "Sequence could not be displayed. Check the prediction error.";
  }
}

function render(data) {
  const variant = data.variant || null;
  const isPathogenic = data.label === "Pathogenic";

  $("result").className =
    "result " + (isPathogenic ? "path" : "ben");

  $("cls").textContent = data.label || "Unknown";
  $("badge").textContent = isPathogenic ? "HIGH RISK" : "LOW RISK";
  $("badge").className =
    "badge " + (isPathogenic ? "path" : "ben");

  const confidence = data.confidence;

  $("conf").textContent =
    confidence == null
      ? "n/a"
      : `${Number(confidence).toFixed(2)} (${Math.round(confidence * 100)}%)`;

  $("confBar").style.width =
    confidence == null ? "0%" : `${confidence * 100}%`;

  $("msg").textContent = data.message || "";

  renderBars($("shapBars"), data.shap || []);

  const rows = [
    ["Gene", data.gene || "HBB"],
    ["Sequence Length", `${data.length || 0} bp`],
    ["Variant Position",
      variant ? `${variant.position} (nt ${variant.nt_position})` : "None"],
    ["Reference Nucleotide",
      variant ? variant.ref_codon : "—"],
    ["Alternate Nucleotide",
      variant ? variant.alt_codon : "—"],
    ["Amino Acid Change",
      variant ? variant.aa_change : "—"],
    ["Conservation Score",
      variant && variant.conservation != null
        ? Number(variant.conservation).toFixed(2)
        : "—"]
  ];

  $("info").innerHTML = rows.map(([key, value]) =>
    `<tr><th>${key}</th><td>${value}</td></tr>`
  ).join("");

  renderSequence(data.sequence || "", variant);
}

function renderSequence(sequence, variant) {
  const box = $("seqBox");

  if (!sequence) {
    box.textContent = "No DNA sequence returned by the server.";
    return;
  }

  const start = variant && Number.isInteger(variant.codon_start)
    ? variant.codon_start
    : -1;

  const end = start >= 0
    ? Math.min(start + 3, sequence.length)
    : -1;

  let output = "";

  for (let row = 0; row < sequence.length; row += 50) {
    output += `<span class="ln">${row + 1}</span>`;

    for (
      let col = row;
      col < Math.min(row + 50, sequence.length);
      col += 10
    ) {
      let chunk = "";

      for (let i = col; i < Math.min(col + 10, sequence.length); i++) {
        const base = sequence[i];

        if (start >= 0 && i >= start && i < end) {
          chunk += `<mark>${base}</mark>`;
        } else {
          chunk += base;
        }
      }

      output += chunk + " ";
    }

    output += "\n";
  }

  box.innerHTML = output;
}

init();
