/* Weryfikacja logiki app.js na ATRAPIE DOM (node, bez przegladarki).
   Uruchamiany przez tests/test_app_js.py, wiec leci razem z pytest.

   PO CO: app.js to jedyny plik warstwy serwujacej bez testu, a zawiera logike, ktora
   potrafi sie cicho zepsuc. Konkretny bug, ktory ten plik pilnuje: domkniecie w
   addEventListener po zmiennej `m` z petli /\[(\d+)\]/g. Po zakonczeniu petli `m` jest
   null, wiec klikniecie KTOREGOKOLWIEK cytatu rzucaloby TypeError — sprawdzone
   eksperymentalnie na wersji z celowo wstawionym bugiem.

   CZEGO NIE SPRAWDZA (uczciwie): wygladu, CSS ani realnego renderowania w przegladarce.
   To atrapa DOM — sprawdza skladanie wezlow i formatowanie, nie prezentacje. */
const fs = require("fs");
const path = process.argv[2];

const reg = new Map();
function mkEl(tag = "div") {
  const el = {
    tag, className: "", textContent: "", hidden: false, open: false,
    title: "", value: "", max: "", checked: false, children: [], listeners: {},
    href: "",
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { cs.forEach((c) => this.children.push(c)); },
    querySelector(sel) {
      const cls = sel.replace(".", "");
      return this.children.find((c) => (c.className || "").split(" ").includes(cls)) || mkEl();
    },
    classList: { _s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
                 has(c) { return this._s.has(c); } },
    addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); },
    scrollIntoView() {},
  };
  // KLUCZOWE dla wiarygodnosci atrapy: w przegladarce element dostaje id i od tej pory
  // getElementById go widzi. Bez tego resetStages() tworzy <li id="stage-gate">, a
  // renderStage() dostawal od atrapy INNY, pusty element — i "blad" byl w atrapie.
  let _id = "";
  Object.defineProperty(el, "id", {
    get: () => _id,
    set: (v) => { _id = v; reg.set(v, el); },
  });
  return el;
}
global.document = {
  getElementById(id) {
    if (!reg.has(id)) { const e = mkEl(); e.id = id; reg.set(id, e); }
    return reg.get(id);
  },
  createElement: (t) => mkEl(t),
  createTextNode: (t) => ({ __text: true, textContent: t }),
  querySelectorAll: () => [],
};
global.fetch = () => Promise.reject(new Error("bez sieci w tej weryfikacji"));

// app.js na koncu wola loadHealth() i rejestruje submit — atrapa to zniesie.
const src = fs.readFileSync(path, "utf8");
const mod = new Function(src + "\n;return { renderAnswer, renderStage, renderFinal, resetStages, readNdjson, loadHealth };");
const { renderAnswer, renderStage, resetStages, loadHealth } = mod();
resetStages();  // jak w realnej petli zapytania

let bledy = 0;
const ok = (warunek, opis) => {
  console.log((warunek ? "  OK   " : "  BLAD ") + opis);
  if (!warunek) bledy++;
};

/* --- 1. cytaty: struktura wezlow --- */
renderAnswer("Norwegia i Dania [1], a Islandia [2] osobno.");
const box = document.getElementById("answer");
const linki = box.children.filter((c) => c.className === "cite");
ok(linki.length === 2, `2 klikalne cytaty (jest ${linki.length})`);
ok(linki[0].textContent === "[1]" && linki[1].textContent === "[2]", "etykiety [1] i [2]");
ok(linki[0].href === "#src-1" && linki[1].href === "#src-2", "href do wlasciwych zrodel");
const tekst = box.children.map((c) => c.textContent).join("");
ok(tekst === "Norwegia i Dania [1], a Islandia [2] osobno.", "tekst zlozony bez ubytkow");

/* --- 2. TEN BUG: klikniecie [1] musi otworzyc src-1, nie ostatnie zrodlo --- */
document.getElementById("src-1").open = false;
document.getElementById("src-2").open = false;
linki[0].listeners.click[0]({ preventDefault() {} });
ok(document.getElementById("src-1").open === true, "klik [1] otwiera src-1");
ok(document.getElementById("src-2").open === false, "klik [1] NIE otwiera src-2 (domkniecie ok)");

/* --- 3. panel etapow: bramka pokazuje liczby, nie samo 'przeszlo' --- */
renderStage({ event: "stage", stage: "gate", status: "done", passed: false,
              top_score: 0.0123, threshold: 0.35 });
const gate = document.getElementById("stage-gate");
ok(/0\.012 < 0\.35/.test(gate.querySelector(".detail").textContent),
   `bramka pokazuje 'score < prog' (jest: "${gate.querySelector(".detail").textContent}")`);
renderStage({ event: "stage", stage: "verify", status: "not_run" });
ok(document.getElementById("stage-verify").querySelector(".detail").textContent === "nie uruchomiono",
   "not_run -> 'nie uruchomiono'");
renderStage({ event: "stage", stage: "generation", status: "done", ms: 8100 });
ok(document.getElementById("stage-generation").querySelector(".detail").textContent === "8.1 s",
   "8100 ms -> '8.1 s'");

/* --- 4. health: etykieta kropki musi nazywac WLASCIWY silnik (etap 7) --- */
/* Po dodaniu przelacznika backendu panel czytal `h.ollama`. Gdyby ktos przywrocil ten klucz
   albo zapomnial o `config.backend`, kropka klamalaby o tym, co liczy odpowiedz — a tego
   nie widac ani w testach Pythona (nie znaja JS), ani na oko (dot jest zielony tak samo). */
function zdrowieAtrapa(payload) {
  global.fetch = () => Promise.resolve({ json: () => Promise.resolve(payload) });
}
const HEALTH_VLLM = {
  qdrant: { ok: true, points: 1509 },
  llm: { ok: true, models: ["Qwen/Qwen2.5-7B-Instruct-AWQ"], missing: [] },
  judge: { ok: true, models: ["gemma2:9b"], missing: [] },
  config: { top_k: 5, max_top_k: 20, gate_min: 0.35, verify_min_supported: 1.0,
            verify_default: true, backend: "vllm", llm: "Qwen/Qwen2.5-7B-Instruct-AWQ",
            judge: "gemma2:9b", embed: "BAAI/bge-m3", reranker: "BAAI/bge-reranker-v2-m3" },
  warm: true,
};

(async () => {
  zdrowieAtrapa(HEALTH_VLLM);
  await loadHealth();
  // addDot sklada <span><span class=dot></span>"etykieta: detal"</span> — tekst jest w
  // DRUGIM dziecku (wezel tekstowy), wiec sklejamy wszystkie dzieci, nie bierzemy pierwszego.
  const etykiety = document.getElementById("health").children
    .map((c) => c.children.map((x) => x.textContent).join("")).join(" | ");
  ok(/vLLM/.test(etykiety), `backend vllm -> kropka nazywa sie vLLM (jest: "${etykiety}")`);
  ok(!/Ollama/.test(etykiety), "przy backendzie vllm nie klamiemy slowem 'Ollama'");
  ok(document.getElementById("submit").disabled === false,
     "Qdrant + generator OK -> przycisk aktywny");

  // Brak SEDZIEGO nie moze blokowac pytania: mozna pytac z wylaczonym guardem #3.
  zdrowieAtrapa({ ...HEALTH_VLLM, judge: { ok: true, models: [], missing: ["gemma2:9b"] } });
  await loadHealth();
  ok(document.getElementById("submit").disabled === false,
     "brak sedziego -> przycisk NADAL aktywny (guard #3 mozna wylaczyc)");

  // Brak GENERATORA blokuje — bez niego nie ma odpowiedzi wcale.
  zdrowieAtrapa({ ...HEALTH_VLLM, llm: { ok: false, error: "URLError" } });
  await loadHealth();
  ok(document.getElementById("submit").disabled === true,
     "generator lezy -> przycisk zablokowany");

  console.log(bledy === 0 ? "\nWSZYSTKO OK" : `\n${bledy} BLEDOW`);
  process.exit(bledy === 0 ? 0 : 1);
})();
