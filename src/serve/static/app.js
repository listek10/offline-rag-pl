/* app.js — bez frameworka, bez build-stepu.
 *
 * Dwie rzeczy warte uwagi:
 * 1. Parser NDJSON (readNdjson): to cena za POST zamiast SSE. EventSource dałby to za darmo,
 *    ale nie obsługuje POST — pytanie musiałoby jechać w URL, a więc do logów dostępowych.
 * 2. Odpowiedź modelu wstawiamy WYŁĄCZNIE przez textContent / węzły DOM. Wyjście LLM-a to
 *    tekst NIEZAUFANY (w kontekście były znaczniki z PDF-a), więc innerHTML jest tu zakazany.
 */
const $ = (id) => document.getElementById(id);
const STAGES = [
  ["retrieval", "retrieval (hybryda + rerank)"],
  ["gate", "bramka trafności"],
  ["generation", "generacja (LLM)"],
  ["verify", "weryfikacja ugruntowania"],
];
let CFG = null;

/* --- health --------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    CFG = h.config;
    const q = h.qdrant, o = h.ollama;
    $("health").textContent = "";
    addDot("Qdrant", q.ok, q.ok ? `${q.points} chunków` : q.error || "brak");
    addDot("Ollama", o.ok, o.ok ? (o.missing.length ? `brak modeli: ${o.missing.join(", ")}` : "modele gotowe") : o.error || "brak");
    if (!h.warm) addDot("modele", false, "nierozgrzane");
    $("top_k").value = CFG.top_k;
    $("top_k").max = CFG.max_top_k;
    $("verify").checked = CFG.verify_default;
    $("config-note").textContent =
      `próg bramki ${CFG.gate_min} · min. poparte tezy ${CFG.verify_min_supported} · ` +
      `${CFG.llm} + sędzia ${CFG.judge}`;
    const gotowe = q.ok && o.ok;
    $("submit").disabled = !gotowe;
    if (!gotowe) $("submit").textContent = "usługi niegotowe";
  } catch (e) {
    $("health").textContent = "serwer nie odpowiada";
  }
}

function addDot(label, ok, detail) {
  const span = document.createElement("span");
  const dot = document.createElement("span");
  dot.className = "dot " + (ok ? "ok" : "bad");
  span.appendChild(dot);
  span.appendChild(document.createTextNode(`${label}: ${detail}`));
  span.title = detail;
  $("health").appendChild(span);
}

/* --- parser strumienia NDJSON --------------------------------------------------------- */
async function* readNdjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();                     // ostatni fragment może być niepełny — zostaw w buforze
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
  if (buf.trim()) yield JSON.parse(buf);
}

/* --- panel przebiegu ------------------------------------------------------------------ */
function resetStages() {
  const ol = $("stages");
  ol.textContent = "";
  for (const [key, label] of STAGES) {
    const li = document.createElement("li");
    li.id = "stage-" + key;
    li.className = "pending";
    const name = document.createElement("span");
    name.textContent = label;
    const detail = document.createElement("span");
    detail.className = "detail";
    detail.textContent = "—";
    li.append(name, detail);
    ol.appendChild(li);
  }
}

function renderStage(ev) {
  const li = $("stage-" + ev.stage);
  if (!li) return;
  const detail = li.querySelector(".detail");
  li.className = ev.status;
  if (ev.status === "start") { detail.textContent = "w toku…"; li.className = "pending"; return; }
  if (ev.status === "skipped") { detail.textContent = "wyłączone"; return; }
  if (ev.status === "not_run") { detail.textContent = "nie uruchomiono"; return; }
  // status === "done"
  const parts = [];
  if (ev.stage === "gate") {
    const znak = ev.passed ? "≥" : "<";
    parts.push(`${ev.top_score.toFixed(3)} ${znak} ${ev.threshold}`);
    parts.push(ev.passed ? "przeszło" : "ODRZUCONE");
  } else {
    if (typeof ev.ms === "number") parts.push(ev.ms >= 1000 ? `${(ev.ms / 1000).toFixed(1)} s` : `${ev.ms} ms`);
    if (ev.stage === "verify" && ev.verdicts) {
      const ok = ev.verdicts.filter((v) => v.grounded).length;
      parts.push(`${ok}/${ev.verdicts.length} tez poparte`);
    }
  }
  detail.textContent = parts.join(" · ");
}

/* --- źródła i cytaty ------------------------------------------------------------------ */
function renderSources(hits) {
  const box = $("sources");
  box.textContent = "";
  for (const h of hits) {
    const d = document.createElement("details");
    d.id = "src-" + h.n;
    const s = document.createElement("summary");
    s.textContent = `[${h.n}] ${h.source} #${h.chunk_index} · score ${h.score}`;
    const body = document.createElement("div");
    body.className = "chunk";
    body.textContent = h.text;            // NIGDY innerHTML — to treść z PDF-a
    d.append(s, body);
    box.appendChild(d);
  }
}

function highlightSource(n) {
  document.querySelectorAll(".sources details").forEach((d) => d.classList.remove("hl"));
  const target = $("src-" + n);
  if (!target) return;
  target.open = true;
  target.classList.add("hl");
  target.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function renderAnswer(text) {
  const box = $("answer");
  box.textContent = "";
  const re = /\[(\d+)\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    // `n` MUSI być kopiowane do const wewnątrz pętli: `m` jest nadpisywane przy każdym obrocie,
    // więc domknięcie w addEventListener widziałoby ostatnie dopasowanie, nie swoje.
    const n = m[1];
    box.appendChild(document.createTextNode(text.slice(last, m.index)));
    const a = document.createElement("a");
    a.href = "#src-" + n;
    a.className = "cite";
    a.textContent = m[0];
    a.addEventListener("click", (e) => { e.preventDefault(); highlightSource(n); });
    box.appendChild(a);
    last = re.lastIndex;
  }
  box.appendChild(document.createTextNode(text.slice(last)));
}

/* --- odmowy i werdykty ---------------------------------------------------------------- */
/* Trzy odmowy znaczą TRZY RÓŻNE rzeczy i muszą być rozróżnione, bo inaczej użytkownik nie wie,
 * czy szukać dalej: bramka mówi „w korpusie nie ma nic na ten temat", guard #1 „temat jest, ale
 * model nie widzi odpowiedzi", guard #3 „odpowiedź powstała, ale nie trzyma się źródeł". */
const POWODY = {
  gate: () => "Zablokowane przez BRAMKĘ TRAFNOŚCI (guard #2): najlepszy fragment nie przekroczył "
            + `progu ${CFG.gate_min}. Model nie był w ogóle pytany — w korpusie nie ma nic na ten temat.`,
  verify: () => "Zablokowane przez WERYFIKACJĘ UGRUNTOWANIA (guard #3): odpowiedź powstała, ale "
              + "sędzia nie potwierdził wszystkich jej tez wobec źródeł. Poniżej odrzucona treść.",
  self: () => "Model SAM odmówił (guard #1: instrukcja w promptcie) — retrieval znalazł fragmenty "
            + "na temat, ale model nie widzi w nich odpowiedzi.",
};

function renderFinal(ev) {
  const powod = ev.blocked_by || (ev.self_refused ? "self" : null);
  if (powod) {
    $("blocked").textContent = POWODY[powod]();
    $("blocked").hidden = false;
  }
  renderAnswer(ev.answer);
  if (ev.rejected_answer) {
    $("rejected").textContent = ev.rejected_answer;
    $("rejected-box").hidden = false;
  }
  const ul = $("verdicts");
  for (const v of ev.grounding || []) {
    if (v.grounded) continue;                       // pokazujemy to, co PODWAŻONE
    const li = document.createElement("li");
    const claim = document.createElement("div");
    claim.textContent = "niepoparta teza: " + v.claim;
    const reason = document.createElement("div");
    reason.className = "reason";
    reason.textContent = "sędzia: " + v.reason;
    li.append(claim, reason);
    ul.appendChild(li);
  }
}

/* --- pętla zapytania ------------------------------------------------------------------ */
$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("submit");
  btn.disabled = true;                             // ~20 s roboty: blokujemy podwójne wysłanie
  const label = btn.textContent;
  btn.textContent = "pracuję…";
  $("blocked").hidden = true;
  $("rejected-box").hidden = true;
  $("answer").textContent = "";
  $("verdicts").textContent = "";
  $("sources").textContent = "";
  resetStages();

  try {
    const resp = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: $("question").value,
        top_k: Number($("top_k").value),
        gate: $("gate").checked,
        verify: $("verify").checked,
      }),
    });
    if (resp.status === 422) { $("answer").textContent = "Nieprawidłowe wejście (puste pytanie lub top_k poza zakresem)."; return; }
    for await (const ev of readNdjson(resp)) {
      if (ev.event === "stage") {
        renderStage(ev);
        if (ev.stage === "retrieval" && ev.hits) renderSources(ev.hits);
      } else if (ev.event === "final") {
        renderFinal(ev);
      } else if (ev.event === "error") {
        $("blocked").textContent = `Błąd (${ev.where}): ${ev.message}`;
        $("blocked").hidden = false;
      }
    }
  } catch (err) {
    $("blocked").textContent = "Utracono połączenie z serwerem: " + err;
    $("blocked").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

loadHealth();
