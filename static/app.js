"use strict";

/* ---- who is this student? (kept in localStorage + a cookie for the server) ---- */
function currentStudent() {
  return localStorage.getItem("gnc_student") || "";
}
function setStudent(name) {
  localStorage.setItem("gnc_student", name);
  document.cookie = "student=" + encodeURIComponent(name) + "; path=/; max-age=31536000; samesite=lax";
}
function ensureStudent(force) {
  let name = currentStudent();
  if (!name || force) {
    name = (window.prompt("Enter your name (used to track your progress):", name || "") || "").trim();
    if (name) setStudent(name);
  }
  const label = document.getElementById("student-label");
  if (label) label.textContent = currentStudent() || "guest";
}

/* ---- admin override (server checks the code; we just carry the answer) ---- */
async function setAdmin(body) {
  const res = await fetch("/admin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function toggleAdmin(isOn) {
  if (isOn) {
    await setAdmin({ off: true });
    location.reload();
    return;
  }
  const code = (window.prompt("Enter the admin code to unlock all modules:") || "").trim();
  if (!code) return;
  let r;
  try {
    r = await setAdmin({ code });
  } catch (e) {
    window.alert("Could not reach the server.");
    return;
  }
  if (r.admin) location.reload();
  else window.alert(r.error || "Incorrect admin code.");
}

/* ---- wipe saved progress (server owns data/progress.json) ---- */
async function resetProgress(isAdmin) {
  const me = currentStudent() || "guest";
  let body = null;
  if (isAdmin) {
    // Admins get the nuclear option first, then fall back to "just me".
    if (window.confirm(`Wipe ALL saved progress for EVERY student?\n\nThis empties progress.json and cannot be undone.\n\nOK = everyone · Cancel = choose "just me" next.`)) {
      body = { all: true };
    }
  }
  if (!body) {
    if (!window.confirm(`Wipe all saved progress for "${me}"?\n\nThis cannot be undone.`)) return;
    body = {};
  }
  try {
    const res = await fetch("/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const r = await res.json();
    if (!r.ok) { window.alert("Could not wipe progress."); return; }
  } catch (e) {
    window.alert("Could not reach the server.");
    return;
  }
  location.href = "/";
}

/* ---- light / dark theme (the <head> script applies it before first paint) ---- */
const editors = [];

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("gnc_theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = theme === "dark" ? "☀️ Light" : "🌙 Dark";
    btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  }
  editors.forEach((cm) => cm.setOption("theme", theme === "dark" ? "material-darker" : "default"));
}

document.addEventListener("DOMContentLoaded", () => {
  ensureStudent(false);
  const changeBtn = document.getElementById("change-student");
  if (changeBtn) changeBtn.addEventListener("click", () => { ensureStudent(true); location.reload(); });

  const adminBtn = document.getElementById("admin-toggle");
  if (adminBtn) adminBtn.addEventListener("click", () => toggleAdmin(adminBtn.dataset.admin === "1"));
  const adminOff = document.getElementById("admin-off");
  if (adminOff) adminOff.addEventListener("click", () => toggleAdmin(true));

  const resetBtn = document.getElementById("reset-progress");
  if (resetBtn) resetBtn.addEventListener("click", () => resetProgress(resetBtn.dataset.admin === "1"));

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) themeBtn.addEventListener("click", () => applyTheme(currentTheme() === "dark" ? "light" : "dark"));

  initExercises();
  initExamples();
  applyTheme(currentTheme());
});

/* ---- figures returned by a run (matplotlib PNGs, as data: URIs) ---- */
function showFigures(container, figures) {
  container.innerHTML = "";
  (figures || []).forEach((src) => {
    const img = document.createElement("img");
    img.className = "cell-figure";
    img.src = src;
    img.alt = "Figure from this cell";
    container.appendChild(img);
  });
}

/* ---- exercise editors ---- */
function initExercises() {
  const moduleEl = document.querySelector(".module");
  if (!moduleEl) return;
  const moduleId = moduleEl.dataset.module;

  document.querySelectorAll(".exercise").forEach((ex) => {
    const textarea = ex.querySelector(".editor");
    const cm = CodeMirror.fromTextArea(textarea, {
      mode: "python",
      theme: currentTheme() === "dark" ? "material-darker" : "default",
      lineNumbers: true,
      indentUnit: 4,
      viewportMargin: Infinity,
    });
    editors.push(cm);

    const exId = ex.dataset.exercise;
    const graded = ex.dataset.graded === "1";
    const outEl = ex.querySelector(".ex-output");
    const figEl = ex.querySelector(".ex-figures");
    const fbEl = ex.querySelector(".feedback");
    const statusEl = ex.querySelector(".ex-status");
    const runBtn = ex.querySelector(".btn.run");
    const checkBtn = ex.querySelector(".btn.check");

    async function post(url) {
      cm.save();
      const body = JSON.stringify({ module: moduleId, exercise: exId, code: cm.getValue() });
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body });
      return res.json();
    }

    function showOutput(text, isError) {
      outEl.hidden = !text;
      outEl.textContent = text || "";
      outEl.classList.toggle("error", !!isError);
    }

    async function runExercise() {
      runBtn.disabled = true; fbEl.hidden = true;
      try {
        const r = await post("/run");
        showFigures(figEl, r.figures);
        // A plot is output too, so a cell that only plots isn't "no output".
        const quiet = (r.figures || []).length ? "" : "(no output)";
        showOutput(r.error ? r.error : (r.stdout || quiet), !!r.error);
      } catch (e) {
        showFigures(figEl, []);
        showOutput("Could not reach the server.", true);
      } finally { runBtn.disabled = false; }
    }
    runBtn.addEventListener("click", runExercise);
    cm.setOption("extraKeys", { "Shift-Enter": runExercise });

    checkBtn.addEventListener("click", async () => {
      if (!currentStudent()) { ensureStudent(true); if (!currentStudent()) return; }
      checkBtn.disabled = true;
      try {
        const r = await post("/grade");
        renderFeedback(r);
      } catch (e) {
        showOutput("Could not reach the server.", true);
      } finally { checkBtn.disabled = false; }
    });

    function renderFeedback(r) {
      showOutput(r.error ? r.error : (r.stdout || ""), !!r.error);
      showFigures(figEl, r.figures);
      fbEl.innerHTML = "";
      fbEl.hidden = false;

      const banner = document.createElement("div");
      banner.className = "banner " + (r.passed ? "pass" : "fail");
      if (!r.graded) {
        banner.textContent = r.passed ? "✓ Ran successfully — marked complete." : "Your code raised an error — see output above.";
      } else {
        banner.textContent = r.passed
          ? "✓ All checks passed — nice work!"
          : `Not quite — ${r.checks.filter((c) => c.ok).length}/${r.checks.length} checks passed.`;
      }
      fbEl.appendChild(banner);

      (r.checks || []).forEach((c) => fbEl.appendChild(checkRow(c)));

      if (r.passed) {
        ex.classList.add("is-done");
        statusEl.textContent = "✓ complete";
        statusEl.className = "ex-status pass";
      } else {
        statusEl.textContent = "";
        statusEl.className = "ex-status fail";
      }
    }
  });
}

function checkRow(c) {
  const row = document.createElement("div");
  row.className = "check-row " + (c.ok ? "ok" : "no");
  const mark = document.createElement("div");
  mark.className = "mark";
  mark.textContent = c.ok ? "✓" : "✗";
  const body = document.createElement("div");

  const label = document.createElement("div");
  label.innerHTML = renderInlineCode(c.label);
  body.appendChild(label);

  if (c.message) {
    const d = document.createElement("div"); d.className = "detail"; d.textContent = c.message;
    body.appendChild(d);
  }
  if (!c.ok && (c.expected != null || c.got != null)) {
    const kv = document.createElement("div"); kv.className = "kv";
    if (c.expected != null) kv.innerHTML += `expected <span class="exp">${escapeHtml(c.expected)}</span>  `;
    if (c.got != null) kv.innerHTML += `got <span class="got">${escapeHtml(c.got)}</span>`;
    body.appendChild(kv);
  }
  row.appendChild(mark); row.appendChild(body);
  return row;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}
function renderInlineCode(s) {
  return escapeHtml(s).replace(/`([^`]+)`/g, "<code>$1</code>");
}

/* ---- worked examples: editable and runnable, but never graded ----
   The server replays the cells above each one, so an example behaves like the
   notebook cell it came from. Edits are local to the page — a reload restores
   the notebook's own code and its saved output. */
function initExamples() {
  const moduleEl = document.querySelector(".module");
  if (!moduleEl) return;
  const moduleId = moduleEl.dataset.module;

  document.querySelectorAll(".example[data-cell]").forEach((ex) => {
    const textarea = ex.querySelector(".example-editor");
    const actions = ex.querySelector(".example-actions");
    const highlight = ex.querySelector(".highlight");
    if (!textarea || !actions) return;

    // Swap the read-only Pygments copy for a live editor.
    highlight.remove();
    textarea.hidden = false;
    actions.hidden = false;

    const cm = CodeMirror.fromTextArea(textarea, {
      mode: "python",
      theme: currentTheme() === "dark" ? "material-darker" : "default",
      lineNumbers: true,
      indentUnit: 4,
      viewportMargin: Infinity,
    });
    editors.push(cm);

    const cell = ex.dataset.cell;
    const runBtn = ex.querySelector(".btn.run");
    const result = ex.querySelector(".cell-result");

    async function runCell() {
      runBtn.disabled = true;
      cm.save();
      try {
        const res = await fetch("/run-cell", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ module: moduleId, cell: Number(cell), code: cm.getValue() }),
        });
        const r = await res.json();
        renderResult(r.error ? r.error : r.stdout, !!r.error, r.figures);
      } catch (e) {
        renderResult("Could not reach the server.", true, []);
      } finally { runBtn.disabled = false; }
    }

    function renderResult(text, isError, figures) {
      result.innerHTML = "";
      const quiet = (figures || []).length ? "" : "(no output)";
      const body = text || quiet;
      if (body) {
        const pre = document.createElement("pre");
        pre.className = "cell-output" + (isError ? " error" : "");
        pre.textContent = body;
        result.appendChild(pre);
      }
      showFigures(result.appendChild(document.createElement("div")), figures);
    }

    runBtn.addEventListener("click", runCell);
    cm.setOption("extraKeys", { "Shift-Enter": runCell });
  });
}
