"use strict";

(function () {
  // Guard against double-initialization if the script is included twice
  if (window.__VISDOM_MAIN_INIT__) return;
  window.__VISDOM_MAIN_INIT__ = true;

  const log = (...args) => console.log("[VISDOM]", ...args);
  const warn = (...args) => console.warn("[VISDOM]", ...args);

  // Global shutdown button handler (delegated)
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest && e.target.closest("#shutdown-btn");
    if (!btn) return;

    // Confirm before killing the server
    if (!window.confirm("Shut down VISDOM now?")) return;

    btn.disabled = true;
    const origText = btn.textContent;
    btn.textContent = "Shutting down…";

    try {
      const r = await fetch("/shutdown", { method: "POST" });
      let msg = "Shutting down…";
      try { msg = (await r.json()).message || msg; } catch(_) {}
      if (typeof window.showToast === "function") window.showToast(msg);
    } catch (err) {
      alert("Could not contact server. It may already be down.");
    } finally {
      // Give the backend a beat to exit, then try to close this window/tab
      setTimeout(() => { window.close(); }, 800);
      // Fallback: restore button if window didn't close (e.g., browser blocks it)
      setTimeout(() => { if (!document.hidden) { btn.disabled = false; btn.textContent = origText; } }, 1500);
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    // ------------------------- Animations: Unified ---------------------------
    (function bootstrapAnimations() {
      const prefersReduced =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      log("main.js loaded • reducedMotion:", prefersReduced);

      const applyPageEnter = () => {
        if (prefersReduced) return;
        const root =
          document.querySelector("body") ||
          document.querySelector("#app") ||
          document.querySelector("main") ||
          document.body;

        if (!root) return;
        root.classList.remove("page-enter");
        // force reflow to restart animation
        // eslint-disable-next-line no-unused-expressions
        void root.offsetWidth;
        root.classList.add("page-enter");
      };

      const supportsIO = "IntersectionObserver" in window;
      const io =
        prefersReduced || !supportsIO
          ? null
          : new IntersectionObserver(
              (entries) => {
                for (const e of entries) {
                  if (e.isIntersecting) {
                    const el = e.target;
                    const delay = el.getAttribute("data-delay");
                    if (delay) el.style.animationDelay = delay;
                    el.classList.add("is-animated");
                    io && io.unobserve(el);
                  }
                }
              },
              { root: null, rootMargin: "0px 0px -10% 0px", threshold: 0.08 }
            );

      const wireObservables = () => {
        const nodes = document.querySelectorAll(
          ".anim, [data-anim], [data-stagger]"
        );
        const count = nodes.length;
        log(`found anim targets: ${count} • IO:${!!io}`);
        nodes.forEach((el) => {
          if (prefersReduced) {
            el.classList.add("is-animated");
          } else if (io) {
            io.observe(el);
          } else {
            el.classList.add("is-animated");
          }
        });
      };

      applyPageEnter();
      wireObservables();
    })();

    // ---------------------------- Common Nodes ------------------------------
    const $ = (id) => document.getElementById(id);

    const toast = $("toast");

    // Crop-Drive controls (if present on page)
    const cropDriveForm = $("crop-drive-form");
    const cropDriveBtn = $("crop-drive-submit");
    const cropStatusSpan = $("crop-status");

    // Split 20→3 controls (if present)
    const splitForm = $("split-drive-form");
    const splitBtn = $("split-drive-submit");
    const splitStatus = $("split-status");

    // OCR controls (if present)
    const ocrForm = $("run-pipeline-form");
    const ocrBtn = $("run-pipeline-submit");
    const ocrStatus = $("run-pipeline-status");

    // Inspection controls (if present)
    const inspectLaserBtn = $("inspect-laser-btn");
    const inspectCriticalBtn = $("inspect-critical-btn");
    const inspectBodyBtn = $("inspect-body-btn");
    const inspectionStatusEl = $("inspection-status");

    // Prepare-defect-folders controls (manual inspection)
    const collectForm = $("collect-defects-form");
    const collectBtn = $("collect-defects-btn");
    const collectStatus = $("collect-status");

    // Master orchestrator controls (home)
    const masterBtn = $("master-btn");
    const masterLight = $("master-light");
    const masterStatus = $("master-status");

    // Level radios
    const laserLevelRadios = document.querySelectorAll(
      "input[name='laser-level']"
    );
    const criticalLevelRadios = document.querySelectorAll(
      "input[name='critical-level']"
    );

    // Advanced selections — Laser
    const laserMiniModelRadios = document.querySelectorAll(
      "input[name='laser-mini-model']"
    );
    const laserModeratePairRadios = document.querySelectorAll(
      "input[name='laser-moderate-pair']"
    );

    // Advanced selections — Critical
    const criticalMiniModelRadios = document.querySelectorAll(
      "input[name='critical-mini-model']"
    );
    const criticalModeratePairRadios = document.querySelectorAll(
      "input[name='critical-moderate-pair']"
    );

    // ------------------------------ Toasts ----------------------------------
    const showToast = (message = "Done.", timeout = 1600) => {
      if (!toast) return;
      toast.textContent = message;
      toast.classList.add("show", "toast-show");
      window.clearTimeout(showToast._t);
      showToast._t = window.setTimeout(() => {
        toast.classList.remove("show", "toast-show");
      }, timeout);
    };
    // Expose for global handlers (e.g., shutdown button above)
    window.showToast = showToast;

    // ---------------------------- Button States -----------------------------
    const setLoadingState = (btn, loading, workingLabel = "Working…") => {
      if (!btn) return;
      if (loading) {
        if (!btn.dataset.label) btn.dataset.label = btn.textContent;
        btn.disabled = true;
        btn.setAttribute("aria-busy", "true");
        btn.textContent = workingLabel;
      } else {
        btn.disabled = false;
        btn.removeAttribute("aria-busy");
        if (btn.dataset.label) btn.textContent = btn.dataset.label;
      }
    };
    // ---------------------- Export Die Reports (2 CSVs) ----------------------
    const exportDieForm = $("export-die-form");
    const exportDieBtn = $("export-die-btn");
    const exportDieStatus = $("export-die-status");

    // One-shot export — no job status polling
    if (exportDieForm && exportDieBtn) {
      exportDieForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        try {
          setLoadingState(exportDieBtn, true, "Building…");
          if (exportDieStatus) exportDieStatus.textContent = "Building reports…";
          const r = await fetch("/export/die_reports", { method: "POST" });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const j = await r.json();
          const msg = j.message || "✅ Export complete.";
          window.showToast && window.showToast(msg);
          if (exportDieStatus) exportDieStatus.textContent = msg;
        } catch (err) {
          const m = err.message || "❌ Export failed.";
          window.showToast && window.showToast(m);
          if (exportDieStatus) exportDieStatus.textContent = m;
        } finally {
          setLoadingState(exportDieBtn, false);
        }
      });
    }


    // After-success: permanently disable if run-once
    const finalizeRunOnce = (btn, statusEl, reason = "Started") => {
      if (!btn) return;
      if (btn.dataset.runonce === "true") {
        btn.disabled = true;
        btn.title = `Idempotent: ${reason}. Clicks are ignored until outputs are cleared.`;
        if (statusEl && !statusEl.textContent)
          statusEl.textContent = `✅ ${reason}.`;
      }
    };

    // ------------------------------ Fetch ----------------------------------
    const postJSON = async (url, payload = {}) => {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      let data = null;
      try {
        data = await res.json();
      } catch (_) {
        // non-JSON; keep data as null
      }

      if (!res.ok) {
        const msg =
          (data && (data.message || data.error)) || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.status = res.status;
        err.data = data;
        throw err;
      }
      return data || {};
    };

    // GET helper (for /manual_data, etc.)
    const getJSON = async (url) => {
      const res = await fetch(url, { method: "GET" });
      let data = null;
      try {
        data = await res.json();
      } catch (_) {}
      if (!res.ok) {
        const msg =
          (data && (data.message || data.error)) || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.status = res.status;
        err.data = data;
        throw err;
      }
      return data || {};
    };

    // -------------------------- Job Status Polling --------------------------
    const pollers = {}; // job -> timeoutId

    const stopPolling = (job) => {
      if (pollers[job]) {
        clearTimeout(pollers[job]);
        delete pollers[job];
      }
    };

    const formatInspectionSummary = (s) => {
      try {
        if (!s || !s.classify) return "";
        const secs =
          s.duration_sec != null ? ` in ${String(s.duration_sec)}s` : "";

        // Accept several shapes from backend
        let comb =
          (s.classify && s.classify.combined_csv) ||
          (s.classify &&
            s.classify.csv_names &&
            s.classify.csv_names.combined) ||
          (s.classify && s.classify.csv_name) ||
          "";

        if (comb) {
          try {
            const bn = comb.split("/").pop();
            comb = bn || comb;
          } catch (_) {}
        }

        const tail = comb ? ` Combined CSV: ${comb}.` : "";
        return `Completed${secs}.${tail}`;
      } catch {
        return "";
      }
    };

    const pollJobOnce = async (job, statusEl, btn, onFinish) => {
      try {
        const res = await fetch(`/job_status?job=${encodeURIComponent(job)}`);
        // If backend doesn’t know this job yet, stop cleanly.
        if (!res.ok) {
          const msg =
            res.status === 400
              ? "⚠️ Backend doesn’t expose this job yet."
              : `⚠️ Job status error (HTTP ${res.status}).`;
          if (statusEl) statusEl.textContent = msg;
          stopPolling(job);
          if (btn) setLoadingState(btn, false);
          return;
        }

        const data = await res.json();
        const st = (data && data.status) || "unknown";

        if (statusEl) {
          if (st === "running") statusEl.textContent = "⏳ Running…";
          else if (st === "done") {
            const extra =
              data && data.summary ? " " + formatInspectionSummary(data.summary) : "";
            statusEl.textContent = `✅ Completed.${extra}`;
          } else if (st === "idle") statusEl.textContent = "🟢 Idle.";
          else if (st === "stale") statusEl.textContent = "🟡 Pending changes.";
          else if (st === "error")
            statusEl.textContent = `❌ ${data?.debug?.reason || "Error."}`;
          else statusEl.textContent = "⚠️ Unknown.";
        }

        if (st === "done") {
          stopPolling(job);
          if (btn) setLoadingState(btn, false);
          window.showToast && window.showToast(`✅ ${job.toUpperCase()} finished.`);
          finalizeRunOnce(btn, statusEl, "Completed");
          if (typeof onFinish === "function") onFinish(data);
        } else if (st === "running") {
          pollers[job] = setTimeout(
            () => pollJobOnce(job, statusEl, btn, onFinish),
            2000
          );
        } else if (st === "error") {
          stopPolling(job);
          if (btn) setLoadingState(btn, false);
          window.showToast && window.showToast(`❌ ${data?.debug?.reason || job + " error."}`);
        } else {
          stopPolling(job);
          if (btn) setLoadingState(btn, false);
        }
      } catch (err) {
        warn("status poll failed:", err);
        pollers[job] = setTimeout(
          () => pollJobOnce(job, statusEl, btn, onFinish),
          4000
        );
      }
    };

    const startPolling = (job, statusEl, btn, onFinish) => {
      stopPolling(job);
      pollJobOnce(job, statusEl, btn, onFinish);
    };

    const initStatus = async (job, statusEl, btn) => {
      try {
        const res = await fetch(`/job_status?job=${encodeURIComponent(job)}`);
        if (!res.ok) {
          // unknown job is fine (e.g., /inspect_body not on backend yet)
          return;
        }
        const data = await res.json();
        const st = (data && data.status) || "unknown";

        if (statusEl) {
          if (st === "running") statusEl.textContent = "⏳ Running…";
          else if (st === "done") statusEl.textContent = "✅ Completed.";
          else if (st === "idle") statusEl.textContent = "🟢 Idle.";
          else if (st === "stale") statusEl.textContent = "🟡 Pending changes.";
          else if (st === "error")
            statusEl.textContent = `❌ ${data?.debug?.reason || "Error."}`;
          else statusEl.textContent = "⚠️ Unknown.";
        }

        if (st === "running") {
          if (btn) setLoadingState(btn, true, "Working…");
          startPolling(job, statusEl, btn);
        }
      } catch (_) {
        // ignore init errors
      }
    };

    // -------------------------- Verify form wiring --------------------------
    (function wireVerifyForm() {
      const form = document.getElementById("verify-form");
      if (!form) return;

      form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const corrections = {};

        // Pattern A: table inputs named actual_code[<image>]
        const tableInputs = form.querySelectorAll("input[name^='actual_code[']");
        tableInputs.forEach((inp) => {
          const m = inp.name.match(/^actual_code\[(.+?)\]$/);
          if (!m) return;
          const image = (m[1] || "").trim();
          const val = (inp.value || "").trim();
          if (image && val) corrections[image] = val;
        });

        // Pattern B: single viewer input + hidden image name
        const viewerInput = form.querySelector("#manual-input");
        const viewerImage = form.querySelector("#image-name");
        if (viewerInput && viewerImage) {
          const image = (viewerImage.value || "").trim();
          const val = (viewerInput.value || "").trim();
          if (image && val) corrections[image] = val;
        }

        if (!Object.keys(corrections).length) {
          window.showToast && window.showToast("⚠️ Nothing to save.");
          return;
        }

        try {
          const data = await postJSON("/save_verifications", { corrections });
          window.showToast && window.showToast(data.message || "✅ Saved corrections.");

          if (typeof window.VERIFY_ADVANCE_NEXT === "function") {
            window.VERIFY_ADVANCE_NEXT();
          }

          if (viewerInput) viewerInput.value = "";
        } catch (err) {
          warn(err);
          window.showToast && window.showToast(err.message || "❌ Save failed.");
        }
      });

      form.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          const tgt = e.target;
          if (tgt && tgt.tagName === "INPUT") {
            e.preventDefault();
            form.requestSubmit ? form.requestSubmit() : form.submit();
          }
        } else if (e.ctrlKey && e.key === "ArrowRight") {
          e.preventDefault();
          if (typeof window.VERIFY_GO_NEXT === "function")
            window.VERIFY_GO_NEXT();
        } else if (e.ctrlKey && e.key === "ArrowLeft") {
          e.preventDefault();
          if (typeof window.VERIFY_GO_PREV === "function")
            window.VERIFY_GO_PREV();
        }
      });
    })();

    // -------------------------- Form Wiring Helper --------------------------
    const wireRunButton = (
      form,
      btn,
      statusEl,
      url,
      jobKey,
      workingLabel = "Starting…",
      onFinish /* optional */
    ) => {
      if (!form || !btn) return;

      // initial status on page load
      initStatus(jobKey, statusEl, btn);

      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (statusEl) statusEl.textContent = workingLabel;
        setLoadingState(btn, true, workingLabel);

        try {
          const data = await postJSON(url, {});
          // Prefer explicit counts when backend provides them
          const newCount =
            data.new_count ??
            data.new_items ??
            data.count ??
            (data.summary &&
              data.summary.classify &&
              data.summary.classify.count);

          let msg = data.message || "✅ Started.";
          if (typeof newCount === "number") {
            msg += ` (${newCount} new)`;
          }
          if (statusEl) statusEl.textContent = msg;
          window.showToast && window.showToast(msg);

          const already = /already|up[-\s]?to[-\s]?date|nothing\s+to\s+do|no\s+new|skipp?ed/i.test(
            String(data.message || "")
          )
            ? "No new images"
            : "Started";
          finalizeRunOnce(btn, statusEl, already);

          startPolling(jobKey, statusEl, btn, onFinish);

          if (data.preview_new) log("New items:", data.preview_new);
          if (data.output_folders) log("Outputs:", data.output_folders);
        } catch (err) {
          warn(err);
          let msg = err.message || "❌ Failed to start.";
          if (err.status === 429) {
            msg = "⏳ A job is already running. Please wait.";
            startPolling(jobKey, statusEl, btn, onFinish);
          }
          if (statusEl) statusEl.textContent = msg;
          window.showToast && window.showToast(msg);
          if (!(btn.dataset.runonce === "true" && btn.disabled)) {
            setLoadingState(btn, false);
          }
        }
      });
    };

    // ------------------------------ Wiring ----------------------------------
    wireRunButton(
      cropDriveForm,
      cropDriveBtn,
      cropStatusSpan,
      "/crop_drive",
      "crop",
      "Starting crop…"
    );
    wireRunButton(
      splitForm,
      splitBtn,
      splitStatus,
      "/split_drive",
      "split",
      "Starting split…"
    );
    wireRunButton(
      ocrForm,
      ocrBtn,
      ocrStatus,
      "/run_pipeline",
      "ocr",
      "Starting OCR…"
    );

    // Prepare Defect Folders (Manual Inspection pre-step) — idempotent, shows new counts
    wireRunButton(
      collectForm,
      collectBtn,
      collectStatus,
      "/prepare_defect_folders",
      "prepare_defects",
      "Collecting defects…",
      () => {
        localStorage.setItem("mi_use_prepared_folders", "1");
        window.showToast && window.showToast("Prepared defect folders ready. Viewer will use them.");
      }
    );

    // =========================================================================
    // Inspection: Levels + Advanced selections
    // =========================================================================

    // Defaults per spec:
    //   Mini     → MaxViT
    //   Moderate → CoAtNet + MaxViT
    //   Max      → SwinV2 + CoAtNet + MaxViT
    const DEFAULTS = {
      mini: ["maxvit"],
      moderate: ["coat", "maxvit"],
      max: ["swinv2", "coat", "maxvit"],
    };

    // Compose-normalize model arrays
    const uniq = (arr) => Array.from(new Set(arr));
    const sortModels = (arr) =>
      uniq(arr).sort((a, b) => {
        const order = { maxvit: 0, coat: 1, swinv2: 2 };
        return (order[a] ?? 9) - (order[b] ?? 9);
      });

    // Helpers to read advanced selections (if present), else defaults
    const readMiniChoice = (scope /* 'laser' | 'critical' */) => {
      const radios =
        scope === "laser" ? laserMiniModelRadios : criticalMiniModelRadios;
      let val = "";
      radios.forEach((r) => {
        if (r.checked) val = (r.value || "").toLowerCase();
      });
      if (!val) return DEFAULTS.mini.slice();
      return [val];
    };

    const readModerateChoice = (scope) => {
      const radios =
        scope === "laser" ? laserModeratePairRadios : criticalModeratePairRadios;
      let val = "";
      radios.forEach((r) => {
        if (r.checked) val = (r.value || "").toLowerCase();
      });
      if (!val) return DEFAULTS.moderate.slice();
      const parts = val.split("+").map((s) => s.trim());
      return sortModels(parts);
    };

    const modelsFor = (scope, level /* 'mini'|'moderate'|'max' */) => {
      if (level === "mini") return readMiniChoice(scope);
      if (level === "moderate") return readModerateChoice(scope);
      return DEFAULTS.max.slice();
    };

    const getLevel = (nodeList, fallback /* 'max' */) => {
      let v = "";
      nodeList.forEach((r) => {
        if (r.checked) v = (r.value || "").toLowerCase();
      });
      return v || fallback;
    };

    // Set default Advanced choices if user hasn't chosen yet
    (function initAdvancedDefaults() {
      // Laser Mini default = maxvit
      if (laserMiniModelRadios.length) {
        const anyChecked = Array.from(laserMiniModelRadios).some((r) => r.checked);
        if (!anyChecked) {
          const def = Array.from(laserMiniModelRadios).find(
            (r) => (r.value || "").toLowerCase() === "maxvit"
          );
          if (def) def.checked = true;
        }
      }
      // Laser Moderate default = coat+maxvit
      if (laserModeratePairRadios.length) {
        const anyChecked = Array.from(laserModeratePairRadios).some(
          (r) => r.checked
        );
        if (!anyChecked) {
          const def = Array.from(laserModeratePairRadios).find((r) => {
            const v = (r.value || "").toLowerCase();
            return v === "coat+maxvit" || v === "maxvit+coat";
          });
          if (def) def.checked = true;
        }
      }
      // Critical Mini default = maxvit
      if (criticalMiniModelRadios.length) {
        const anyChecked = Array.from(criticalMiniModelRadios).some(
          (r) => r.checked
        );
        if (!anyChecked) {
          const def = Array.from(criticalMiniModelRadios).find(
            (r) => (r.value || "").toLowerCase() === "maxvit"
          );
          if (def) def.checked = true;
        }
      }
      // Critical Moderate default = coat+maxvit
      if (criticalModeratePairRadios.length) {
        const anyChecked = Array.from(criticalModeratePairRadios).some(
          (r) => r.checked
        );
        if (!anyChecked) {
          const def = Array.from(criticalModeratePairRadios).find((r) => {
            const v = (r.value || "").toLowerCase();
            return v === "coat+maxvit" || v === "maxvit+coat";
          });
          if (def) def.checked = true;
        }
      }
    })();

    // --------------------------- Inspection Actions --------------------------
    const setInspectionMsg = (text) => {
      if (inspectionStatusEl) inspectionStatusEl.textContent = text;
    };

    const sendInspection = async (scope /* 'laser' | 'critical' */) => {
      const isLaser = scope === "laser";
      const btn = isLaser ? inspectLaserBtn : inspectCriticalBtn;

      // Guard: ignore if already busy/disabled (idempotent click)
      if (!btn || btn.disabled || btn.getAttribute("aria-busy") === "true") {
        return;
      }

      const level =
        isLaser && laserLevelRadios.length
          ? getLevel(laserLevelRadios, "max")
          : !isLaser && criticalLevelRadios.length
          ? getLevel(criticalLevelRadios, "max")
          : "max";

      // Build selection payload per level:
      //   level='mini'     → { selection: { mini: '<model>' } }
      //   level='moderate' → { selection: { moderate: ['m1','m2'] } }
      //   level='max'      → no selection (always all three)
      let selection = null;
      if (level === "mini") {
        const models = modelsFor(scope, "mini");
        selection = { mini: models[0] }; // exactly one
      } else if (level === "moderate") {
        const models = modelsFor(scope, "moderate");
        selection = { moderate: models.slice(0, 2) }; // exactly two
      }

      const payload = selection ? { level, selection } : { level };
      const url = isLaser ? "/inspect_laser" : "/inspect_critical";
      const job = isLaser ? "inspect_laser" : "inspect_critical";

      setInspectionMsg("Starting…");
      setLoadingState(btn, true, "Starting…");

      try {
        const data = await postJSON(url, payload);
        let msg = data.message || "✅ Started.";
        const newCount =
          data.new_count ??
          data.new_items ??
          data.count ??
          (data.summary &&
            data.summary.classify &&
            data.summary.classify.count);
        if (typeof newCount === "number") msg += ` (${newCount} new)`;
        setInspectionMsg(msg);
        window.showToast && window.showToast(msg);

        // If server indicates nothing to do / already done, treat as idempotent and lock button.
        if (
          /already|up[-\s]?to[-\s]?date|nothing\s+to\s+do|no\s+new|skipp?ed/i.test(
            String(data.message || "")
          )
        ) {
          setLoadingState(btn, false);
          finalizeRunOnce(btn, inspectionStatusEl, "No new images");
          return;
        }

        // Begin polling for completion and summary
        startPolling(job, inspectionStatusEl, btn, (fin) => {
          const extra =
            fin && fin.summary ? " " + formatInspectionSummary(fin.summary) : "";
          if (extra) setInspectionMsg(`✅ Completed.${extra}`);

          // If zero items were processed, permanently mark as idempotent to avoid re-runs.
          try {
            const count =
              (fin &&
                fin.summary &&
                fin.summary.classify &&
                fin.summary.classify.count) ||
              0;
            if (Number(count) === 0) {
              finalizeRunOnce(btn, inspectionStatusEl, "No new images");
            }
          } catch (_) {
            /* noop */
          }
        });
      } catch (err) {
        warn(err);
        let msg = err.message || "❌ Failed to start.";
        if (err.status === 429) {
          msg = "⏳ An inspection is already running. Please wait.";
          // Still start polling to reflect the existing run
          startPolling(job, inspectionStatusEl, btn);
        } else {
          // restore button only if not a concurrency case
          setLoadingState(btn, false);
        }
        setInspectionMsg(msg);
        window.showToast && window.showToast(msg);
      }
    };

    // --- Body inspection (no levels) ---
    const sendBodyInspection = async () => {
      const btn = inspectBodyBtn;
      if (!btn || btn.disabled || btn.getAttribute("aria-busy") === "true")
        return;

      const job = "inspect_body";
      setInspectionMsg("Starting body inspection…");
      setLoadingState(btn, true, "Starting…");

      try {
        const data = await postJSON("/inspect_body", {});
        let msg = data.message || "✅ Started.";
        const newCount =
          data.new_count ??
          data.new_items ??
          data.count ??
          (data.summary &&
            data.summary.classify &&
            data.summary.classify.count);
        if (typeof newCount === "number") msg += ` (${newCount} new)`;
        setInspectionMsg(msg);
        window.showToast && window.showToast(msg);

        // If backend says up-to-date, mark idempotent
        if (
          /already|up[-\s]?to[-\s]?date|nothing\s+to\s+do|no\s+new|skipp?ed/i.test(
            String(data.message || "")
          )
        ) {
          setLoadingState(btn, false);
          finalizeRunOnce(btn, inspectionStatusEl, "No new images");
          return;
        }

        // Begin polling for completion
        startPolling(job, inspectionStatusEl, btn, (fin) => {
          const extra =
            fin && fin.summary ? " " + formatInspectionSummary(fin.summary) : "";
          if (extra) setInspectionMsg(`✅ Completed.${extra}`);
          try {
            const count =
              (fin &&
                fin.summary &&
                fin.summary.classify &&
                fin.summary.classify.count) ||
              0;
            if (Number(count) === 0) {
              finalizeRunOnce(btn, inspectionStatusEl, "No new images");
            }
          } catch (_) {}
        });
      } catch (err) {
        // Graceful fallback if backend isn’t ready for body yet
        warn(err);
        let msg = err.message || "❌ Failed to start.";
        if (err.status === 404 || err.status === 400 || err.status === 501) {
          msg = "⚠️ Body backend not available yet.";
        } else if (err.status === 429) {
          msg = "⏳ An inspection is already running. Please wait.";
          startPolling(job, inspectionStatusEl, btn);
        } else {
          setLoadingState(btn, false);
        }
        setInspectionMsg(msg);
        window.showToast && window.showToast(msg);
      }
    };

    if (inspectLaserBtn) {
      inspectLaserBtn.addEventListener("click", (e) => {
        e.preventDefault();
        sendInspection("laser");
      });
    }
    if (inspectCriticalBtn) {
      inspectCriticalBtn.addEventListener("click", (e) => {
        e.preventDefault();
        sendInspection("critical");
      });
    }
    if (inspectBodyBtn) {
      // Attempt to read initial status if backend already supports it.
      initStatus("inspect_body", inspectionStatusEl, inspectBodyBtn);
      inspectBodyBtn.addEventListener("click", (e) => {
        e.preventDefault();
        sendBodyInspection();
      });
    }

    // =========================================================================
    // Manual Inspection v2 (viewer): one-by-one viewer
    // =========================================================================
    (function wireManualInspectionViewer() {
      // Detect presence of the viewer UI
      const imgEl = document.getElementById("img");
      const nextBtn = document.getElementById("next");
      const prevBtn = document.getElementById("prev");
      const markDefBaseBtn = document.getElementById("mark-def-base");
      const markGoodBtn = document.getElementById("mark-good");
      const onlyBadEl = document.getElementById("only-bad");
      const regionEl = document.getElementById("region-filter");
      const hideDecidedEl = document.getElementById("hide-decided");
      const confirmModal = document.getElementById("confirm-modal");

      const viewerPresent = !!(
        imgEl &&
        nextBtn &&
        prevBtn &&
        markDefBaseBtn &&
        markGoodBtn &&
        onlyBadEl &&
        regionEl &&
        hideDecidedEl
      );
      if (!viewerPresent) return; // No viewer present

      if (window.__MI_VIEWER_BOUND__) return;
      window.__MI_VIEWER_BOUND__ = true;

      // Base-key per naming rule: {name}_laser_die_{1-20}_{laser|critical|body}.(png|jpg)
      const BASE_RE =
        /^(?<pfx>.+?_laser)(?:_die_\d{1,2})?_(?:laser|critical|body)\.[A-Za-z0-9]+$/;
      const baseKeyOf = (name) =>
        name?.match(BASE_RE)?.groups?.pfx || name || "";

      const sevClass = (n) => `sev-${Math.max(0, Math.min(3, n || 0))}`;
      const voteClass = (v) => {
        const s = (v || "").toLowerCase();
        if (s === "defective") return "bad";
        if (s === "good") return "good";
        return "na";
      };

      // Normalizes any returned path/url to something fetchable by the browser.
      const normalizeImageURL = (u) => {
        if (!u) return "";
        if (
          /^https?:\/\//i.test(u) ||
          u.startsWith("/proxy") ||
          u.startsWith("/image/") ||
          u.startsWith("/local/")
        ) {
          return u;
        }
        // Treat as a relative path under IMAGES_DIR; use backend /local/<rel> route
        return "/local/" + String(u).replace(/^\/+/, "");
      };

      // Prefer proxy URL from backend (Drive-safe), then other hints, include local keys
      const pickImgUrl = (it) =>
        normalizeImageURL(
          it.proxy_url ||
            it.drive_url ||
            it.img_url ||
            it.image_url ||
            it.img_path ||
            ""
        );

      // Normalize backend model vote keys to lowercase expected by viewer
      function normalizeModelKeys(m) {
        if (!m) return { maxvit: "N/A", coat: "N/A", swinv2: "N/A" };
        return {
          maxvit: m.maxvit ?? m.MaxViT ?? "N/A",
          coat: m.coat ?? m.CoaT ?? "N/A",
          swinv2: m.swinv2 ?? m.SwinV2 ?? "N/A",
        };
      }

      const countDef = (m) =>
        ["maxvit", "coat", "swinv2"].reduce(
          (k, key) =>
            k + ((m?.[key] || "").toLowerCase() === "defective" ? 1 : 0),
          0
        );

      // State
      let all = []; // full dataset from backend
      let filtered = []; // after filters
      let idx = 0; // pointer

      // UNDO STACK (last 50 actions)
      const actionStack = [];
      const pushAction = (action) => {
        actionStack.push(action);
        if (actionStack.length > 50) actionStack.shift(); // keep last 50
        setUndoEnabled(true);
      };
      const popAction = () => {
        const a = actionStack.pop();
        setUndoEnabled(actionStack.length > 0);
        return a;
      };

      // DOM references
      const titleEl = document.getElementById("title");
      const baseEl = document.getElementById("base");
      const regionOutEl = document.getElementById("region");
      const sevDot = document.getElementById("sevdot");
      const sevText = document.getElementById("sevtext");
      const sevChip = document.getElementById("sevchip");
      const vMax = document.getElementById("v-maxvit");
      const vCoaT = document.getElementById("v-coat");
      const vSwin = document.getElementById("v-swinv2");
      const posEl = document.getElementById("pos");
      const crumbEl = document.getElementById("crumb");
      const counterEl = document.getElementById("counter");
      const undoBtn = document.getElementById("undo");
      const reloadBtn = document.getElementById("reload");
      const dontAskAgainEl = document.getElementById("dont-ask-again");
      const confirmCancel = document.getElementById("confirm-cancel");
      const confirmOk = document.getElementById("confirm-ok");

      function setUndoEnabled(enabled) {
        if (undoBtn) undoBtn.disabled = !enabled;
      }

      function current() {
        return filtered[idx];
      }

      // Backend I/O
      async function loadManualData(onlyBad = true) {
        // Prefer prepared folders if user ran "Prepare Defect Folders"
        const usePrepared =
          localStorage.getItem("mi_use_prepared_folders") === "1";
        const qs = `only_bad=${onlyBad ? 1 : 0}${
          usePrepared ? "&use_prepared_folders=1" : ""
        }`;
        try {
          const data = await getJSON(`/manual_data?${qs}`);
          const items = data?.items || [];

          // Normalize per-item fields for the viewer:
          items.forEach((it) => {
            if (!it.base_key && it.image) it.base_key = baseKeyOf(it.image);
            it.models = normalizeModelKeys(it.models);
            // Make sure image url is usable:
            if (it.image_url) it.image_url = normalizeImageURL(it.image_url);
            if (it.img_url) it.img_url = normalizeImageURL(it.img_url);
            if (it.img_path) it.img_path = normalizeImageURL(it.img_path);
          });

          return items;
        } catch (e) {
          warn("manual_data failed", e);
          return [];
        }
      }

      // IMPORTANT: Backend expects (base_key, region, decision, last_image)
      async function saveDecision(image, region, decision) {
        const base_key = baseKeyOf(image || "");
        const payload = {
          base_key,
          region,
          decision: (decision || "").toUpperCase(),
          last_image: image || "",
        };
        try {
          await postJSON("/manual/decision", payload);
        } catch (e) {
          warn("saveDecision failed", e);
        }
      }

      // Render helpers
      function renderEmpty() {
        if (imgEl) imgEl.src = "";
        if (titleEl) titleEl.textContent = "No items";
        if (baseEl) baseEl.textContent = "—";
        if (regionOutEl) regionOutEl.textContent = "—";
        if (sevText) sevText.textContent = "0/3 Defective";
        if (sevDot) sevDot.className = "dot sev-0";
        [vMax, vCoaT, vSwin].forEach((e) => {
          if (e) {
            e.className = "na";
            e.textContent = "N/A";
          }
        });
        if (posEl) posEl.textContent = "0/0";
        if (crumbEl) crumbEl.textContent = "—";
        if (counterEl) counterEl.textContent = "(0 items)";
      }

      function render() {
        if (!filtered.length) {
          renderEmpty();
          return;
        }
        if (idx < 0) idx = 0;
        if (idx >= filtered.length) idx = filtered.length - 1;

        const it = filtered[idx];
        const url = pickImgUrl(it);
        if (imgEl) {
          imgEl.src = url;
          imgEl.alt = it.image || "image";
        }
        if (titleEl)
          titleEl.textContent = `${(it.region || "").toUpperCase()} · ${
            it.image || ""
          }`;
        if (baseEl) baseEl.textContent = it.base_key || baseKeyOf(it.image || "");
        if (regionOutEl) regionOutEl.textContent = it.region || "—";

        const sev = countDef(it.models);
        if (sevDot) sevDot.className = "dot " + sevClass(sev);
        if (sevText) sevText.textContent = `${sev}/3 Defective`;
        if (sevChip) sevChip.title = `${sev} model(s) marked Defective`;

        if (vMax) {
          vMax.className = voteClass(it.models?.maxvit);
          vMax.textContent = it.models?.maxvit || "N/A";
        }
        if (vCoaT) {
          vCoaT.className = voteClass(it.models?.coat);
          vCoaT.textContent = it.models?.coat || "N/A";
        }
        if (vSwin) {
          vSwin.className = voteClass(it.models?.swinv2);
          vSwin.textContent = it.models?.swinv2 || "N/A";
        }

        if (posEl) posEl.textContent = `${idx + 1}/${filtered.length}`;
        if (crumbEl) crumbEl.textContent = it.image || "—";
        if (counterEl)
          counterEl.textContent = `(${filtered.length} item${
            filtered.length > 1 ? "s" : ""
          })`;
      }

      function applyFilters() {
        const onlyBad = !!onlyBadEl?.checked;
        const region = regionEl?.value || "all";
        const hideDecided = !!hideDecidedEl?.checked;

        filtered = all.filter((it) => {
          if (region !== "all" && it.region !== region) return false;
          if (onlyBad && countDef(it.models) < 1) return false;
          if (hideDecided && it.decision) return false;
          return true;
        });

        idx = Math.min(idx, Math.max(0, filtered.length - 1));
        render();
      }

      function move(delta) {
        if (!filtered.length) return;
        idx += delta;
        if (idx < 0) idx = 0;
        if (idx >= filtered.length) idx = filtered.length - 1;
        render();
      }

      function allByBase(base) {
        return all.filter(
          (z) => (z.base_key || baseKeyOf(z.image || "")) === base
        );
      }

      function jumpToNextSibling(base, fromIndex) {
        if (!filtered.length) return false;
        const start = Math.max(0, fromIndex + 1);
        for (let i = start; i < filtered.length; i++) {
          const b =
            filtered[i].base_key || baseKeyOf(filtered[i].image || "");
          if (b === base) {
            idx = i;
            render();
            return true;
          }
        }
        return false;
      }

      async function markDefectiveBase(it) {
        if (!it) return;
        const base = it.base_key || baseKeyOf(it.image || "");
        const affectAll = allByBase(base);
        if (!affectAll.length) {
          window.showToast && window.showToast("Nothing to mark.");
          return;
        }

        // Snapshot for UNDO (deep-ish copy of affected items)
        const snapshot = affectAll.map((x) => ({
          image: x.image,
          region: x.region,
          prev: x.decision || "",
          models: x.models,
          base_key: x.base_key,
        }));

        // Optimistic UI + persist (mark every item DEFECTIVE)
        for (const row of affectAll) {
          row.decision = "DEFECTIVE";
          await saveDecision(row.image, row.region, "DEFECTIVE");
        }

        // Remove this base entirely from the working queues
        const beforeLen = all.length;
        all = all.filter(
          (z) => (z.base_key || baseKeyOf(z.image || "")) !== base
        );
        const removedCount = beforeLen - all.length;

        // Push UNDO action
        pushAction({
          type: "batch-def",
          base,
          removed: snapshot,
          undo: async () => {
            for (const row of snapshot) {
              all.push({
                image: row.image,
                region: row.region,
                decision: row.prev || undefined,
                models: row.models,
                base_key: row.base_key,
              });
              await saveDecision(row.image, row.region, row.prev || "");
            }
            applyFilters();
            const firstIdx = filtered.findIndex(
              (z) => (z.base_key || baseKeyOf(z.image || "")) === base
            );
            if (firstIdx >= 0) {
              idx = firstIdx;
              render();
            }
            window.showToast && window.showToast("Undone.");
          },
        });

        // Refresh view and advance
        applyFilters();
        window.showToast && window.showToast(`Marked base as Defective (${removedCount} removed)`);
      }

      function mustConfirmGood() {
        return localStorage.getItem("mi_skip_confirm_good") !== "1";
      }

      function confirmGood(it) {
        if (!it) return;
        if (!confirmModal) return doMarkGood(it);
        if (!mustConfirmGood()) return doMarkGood(it);
        if (dontAskAgainEl) dontAskAgainEl.checked = false;
        confirmModal.showModal();
        confirmModal._pendingItem = it;
      }

      async function doMarkGood(it) {
        if (!it) return;
        const prev = it.decision || "";
        const base = it.base_key || baseKeyOf(it.image || "");
        it.decision = "GOOD";
        await saveDecision(it.image, it.region, "GOOD");

        // Push UNDO action
        pushAction({
          type: "single-good",
          image: it.image,
          region: it.region,
          prev,
          undo: async () => {
            await saveDecision(it.image, it.region, prev || "");
            const ref = all.find(
              (z) => z.image === it.image && z.region === it.region
            );
            if (ref) {
              if (prev) ref.decision = prev;
              else delete ref.decision;
            }
            applyFilters();
            window.showToast && window.showToast("Undone.");
          },
        });

        // Advance to next image of SAME base if present; else next item
        if (!jumpToNextSibling(base, idx)) {
          move(1);
        } else {
          render();
        }
        window.showToast && window.showToast("Marked Good.");
      }

      async function undo() {
        const act = popAction();
        if (!act) return;
        await act.undo?.();
      }

      async function reload() {
        const onlyBad = !!onlyBadEl?.checked;
        all = await loadManualData(onlyBad);
        applyFilters();
        window.showToast && window.showToast("Reloaded.");
      }

      // Wire controls
      prevBtn?.addEventListener("click", () => move(-1));
      nextBtn?.addEventListener("click", () => move(1));
      reloadBtn?.addEventListener("click", reload);
      undoBtn?.addEventListener("click", undo);
      markDefBaseBtn?.addEventListener("click", () =>
        markDefectiveBase(current())
      );
      markGoodBtn?.addEventListener("click", () => confirmGood(current()));

      onlyBadEl?.addEventListener("change", reload); // re-fetch to expand dataset if unchecked
      regionEl?.addEventListener("change", applyFilters);
      hideDecidedEl?.addEventListener("change", applyFilters);

      // Confirm dialog events + backend toggle (optional; no UI change)
      confirmCancel?.addEventListener("click", () => confirmModal.close());
      confirmOk?.addEventListener("click", async () => {
        if (dontAskAgainEl?.checked) {
          localStorage.setItem("mi_skip_confirm_good", "1");
          try {
            await postJSON("/manual/confirm_toggle", { disable: true });
          } catch (_) {}
        }
        const it = confirmModal._pendingItem;
        confirmModal._pendingItem = null;
        confirmModal.close();
        if (it) doMarkGood(it);
      });

      // Keyboard shortcuts
      window.addEventListener("keydown", (e) => {
        if (e.defaultPrevented) return;
        const tag = (e.target && (e.target.tagName || "")).toLowerCase();
        if (tag === "input" || tag === "textarea") return;

        switch (e.key) {
          case "ArrowLeft":
            e.preventDefault();
            move(-1);
            break;
          case "ArrowRight":
            e.preventDefault();
            move(1);
            break;
          case "d":
          case "D":
            e.preventDefault();
            markDefectiveBase(current());
            break;
          case "g":
          case "G":
            e.preventDefault();
            confirmGood(current());
            break;
          case "u":
          case "U":
            e.preventDefault();
            undo();
            break;
          case "r":
          case "R":
            e.preventDefault();
            reload();
            break;
          case "h":
          case "H":
            e.preventDefault();
            if (hideDecidedEl) {
              hideDecidedEl.checked = !hideDecidedEl.checked;
              applyFilters();
            }
            break;
          default:
            break;
        }
      });

      // Boot
      (async function init() {
        setUndoEnabled(false);
        const onlyBadDefault = true; // "show only bad" by default
        all = await loadManualData(onlyBadDefault);
        applyFilters();
      })();
    })();

    // ====================== Master automation (managed / /master API) ======================
    (function wireMaster() {
      if (!masterBtn || !masterLight || !masterStatus) return;

      const setLight = (color) => {
        masterLight.classList.remove("red", "orange", "yellow", "green");
        masterLight.classList.add(color);
      };

      const updateUI = (s) => {
        const st = (s && s.status) || "idle";
        const phase = (s && s.phase) || "idle";
        const msg = (s && s.message) || "";

        masterStatus.textContent =
          msg ||
          (st === "running"
            ? `Running… (${phase})`
            : st === "done"
            ? "Completed."
            : st === "error"
            ? `Error: ${s.error || "unknown"}`
            : "Idle.");
        if (st === "running") {
          setLight("yellow");
          setLoadingState(masterBtn, true, "Running…");
        } else if (st === "done") {
          setLight("green");
          setLoadingState(masterBtn, false);
        } else if (st === "error") {
          setLight("red");
          setLoadingState(masterBtn, false);
        } else {
          setLight("green");
          setLoadingState(masterBtn, false);
        }
      };

      let masterTimer = null;
      const stop = () => {
        if (masterTimer) {
          clearTimeout(masterTimer);
          masterTimer = null;
        }
      };
      const poll = async () => {
        try {
          const r = await fetch("/master/status");
          if (!r.ok) {
            stop();
            return;
          }
          const data = await r.json();
          updateUI(data);
          if (data.status === "running") {
            masterTimer = setTimeout(poll, 1500);
          } else {
            stop();
          }
        } catch (_) {
          // keep polling slowly on transient failures
          masterTimer = setTimeout(poll, 3000);
        }
      };

      // On load, show current state if any
      poll();

      masterBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        setLoadingState(masterBtn, true, "Starting…");
        setLight("yellow");
        masterStatus.textContent = "Starting…";
        try {
          const res = await fetch("/master/start", { method: "POST" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          updateUI(data);
          poll();
        } catch (err) {
          setLoadingState(masterBtn, false);
          setLight("red");
          masterStatus.textContent = `Failed to start: ${err.message || err}`;
        }
      });
    })();

    window.addEventListener("beforeunload", () => {
      Object.keys(pollers).forEach((k) => stopPolling(k));
    });
  });
})();
