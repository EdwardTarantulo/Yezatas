// Polls /api/status while a sign-in, a catalog pull, a push, a suggestion, a translation, subtitles, a transcript, chapters, a packaging check, variants, an outlier explanation, a strategy report, a copilot answer, a bulk edit agent turn, a link check, an advertiser-friendly check, a comments fetch or a comment sort are running, then reloads the page when a job finishes.
// Copy buttons copy the text of the element named by data-copy-target. Also stops double submits, shows character counters and wires the theme toggle. No external code.
(function () {
  "use strict";

  // Theme toggle in the top bar (theme.js already set the theme). Hidden without JS; pressed means dark.
  var themeButton = document.querySelector("[data-theme-toggle]");
  if (themeButton && window.yezatasTheme) {
    var showTheme = function () {
      themeButton.setAttribute("aria-pressed", window.yezatasTheme.current() === "dark" ? "true" : "false");
    };
    themeButton.addEventListener("click", function () {
      window.yezatasTheme.set(window.yezatasTheme.current() === "dark" ? "light" : "dark");
    });
    document.addEventListener("yezatas-theme", showTheme);
    showTheme();
    themeButton.hidden = false;
  }

  document.querySelectorAll("form").forEach(function (form) {
    form.addEventListener("submit", function () {
      form.querySelectorAll("button[type=submit]").forEach(function (button) {
        // Disable after the browser has collected the form, so the POST still goes out.
        setTimeout(function () { button.disabled = true; }, 0);
      });
    });
  });

  // Slide guides: one screen at a time, Back / Next and dots. Without JS every slide is shown.
  document.querySelectorAll("[data-slider]").forEach(function (slider) {
    var slides = Array.prototype.slice.call(slider.querySelectorAll("[data-slide]"));
    var dots = slider.querySelector("[data-dots]");
    var prev = slider.querySelector(".slider-nav [data-slide-prev]");
    var next = slider.querySelector(".slider-nav [data-slide-next]");
    var current = 0;
    slides.forEach(function (_, i) {
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "dot";
      dot.setAttribute("aria-label", "Slide " + (i + 1) + " of " + slides.length);
      dot.addEventListener("click", function () { show(i); });
      dots.appendChild(dot);
    });
    function show(i) {
      current = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach(function (s, j) { s.hidden = j !== current; });
      Array.prototype.forEach.call(dots.children, function (d, j) {
        d.classList.toggle("active", j === current);
        if (j === current) d.setAttribute("aria-current", "step"); else d.removeAttribute("aria-current");
      });
      prev.disabled = current === 0;
      next.disabled = current === slides.length - 1;
    }
    slider.addEventListener("click", function (e) {
      var el = e.target.closest("button");
      if (!el || !slider.contains(el)) return;
      if (el.hasAttribute("data-slide-next")) show(current + 1);
      else if (el.hasAttribute("data-slide-prev")) show(current - 1);
      else if (el.hasAttribute("data-slide-goto")) show(Number(el.getAttribute("data-slide-goto")));
    });
    slider.classList.add("js");
    show(0);
  });

  // Edit form: a live count next to the title and description fields (the server still validates).
  document.querySelectorAll("[data-count-for]").forEach(function (field) {
    var out = document.getElementById(field.getAttribute("data-count-for"));
    if (!out) return;
    var limit = Number(field.getAttribute("data-count-limit"));
    var unit = field.getAttribute("data-count-unit");
    var encoder = window.TextEncoder ? new TextEncoder() : null;
    function update() {
      var n = unit === "bytes" && encoder ? encoder.encode(field.value).length : field.value.length;
      out.textContent = n + " of " + limit + " " + unit + (n > limit ? ": too long for YouTube" : "");
      out.classList.toggle("over", n > limit);
    }
    field.addEventListener("input", update);
    update();
  });

  // Translations card: Select all / none / not translated / out of date (each label carries data-state).
  document.querySelectorAll("[data-select-for]").forEach(function (bar) {
    var box = document.getElementById(bar.getAttribute("data-select-for"));
    if (!box) return;
    bar.hidden = false;
    bar.addEventListener("click", function (e) {
      var button = e.target.closest("[data-select]");
      if (!button) return;
      var how = button.getAttribute("data-select");
      box.querySelectorAll("label[data-state]").forEach(function (label) {
        var input = label.querySelector("input[type=checkbox]");
        var state = label.getAttribute("data-state");
        if (!input || state === "own") return;
        input.checked = how === "all" || (how === "new" && (state === "" || state === "failed"))
          || (how === "stale" && state === "stale");
      });
    });
  });

  // The video page's Analytics card reads itself on opening when the server marks its form as due (numbers
  // missing or old). At most once per video every 12 hours in this tab, so a failed read is not retried on every visit.
  var autoForm = document.querySelector("form[data-auto-refresh]");
  if (autoForm) {
    var autoKey = "yz-auto-read:" + autoForm.getAttribute("action");
    var tried = 0;
    try { tried = Number(window.sessionStorage.getItem(autoKey) || 0); } catch (e) { tried = 0; }
    if (!tried || Date.now() - tried > 12 * 3600 * 1000) {
      try { window.sessionStorage.setItem(autoKey, String(Date.now())); } catch (e) { /* storage off */ }
      keepScroll();
      autoForm.submit();
    }
  }

  // Coming back with the Back button must not show disabled buttons or stale state.
  window.addEventListener("pageshow", function (e) { if (e.persisted) window.location.reload(); });

  // Copy buttons: copy the text of the element named by data-copy-target.
  document.querySelectorAll("[data-copy-target]").forEach(function (button) {
    button.addEventListener("click", function () {
      var el = document.getElementById(button.getAttribute("data-copy-target"));
      if (!el) return;
      var text = el.textContent || "";
      function fallback() {
        var area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        try { document.execCommand("copy"); } catch (e) { /* nothing more to try */ }
        area.remove();
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(fallback);
      } else {
        fallback();
      }
    });
  });

  // Example chips (Bulk edit): put the example into the textarea named by data-fill-target.
  document.querySelectorAll("[data-fill-text]").forEach(function (chip) {
    chip.addEventListener("click", function () {
      var field = document.getElementById(chip.getAttribute("data-fill-target"));
      if (!field) return;
      field.value = chip.getAttribute("data-fill-text");
      field.dispatchEvent(new Event("input", { bubbles: true }));
      field.focus();
    });
  });

  // Pop-ups: a button with data-dialog-open opens the native dialog it names. The browser traps focus,
  // closes on Escape and puts focus back on the button. A click outside the dialog closes it too.
  document.querySelectorAll("[data-dialog-open]").forEach(function (button) {
    var dialog = document.getElementById(button.getAttribute("data-dialog-open"));
    if (!dialog || !dialog.showModal) return;
    button.hidden = false;
    button.addEventListener("click", function () { dialog.showModal(); dialog.scrollTop = 0; });
    dialog.querySelectorAll("[data-dialog-close]").forEach(function (close) {
      close.addEventListener("click", function () { dialog.close(); });
    });
    dialog.addEventListener("click", function (e) {
      var box = dialog.getBoundingClientRect();
      if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) dialog.close();
    });
  });

  if (document.body.dataset.poll !== "1") return;

  var csrf = document.body.dataset.csrf || "";
  var seenRunning = false;

  // Bulk edit agent: add the log lines written since the page was drawn (text only, never HTML), so the
  // panel keeps its size and nothing below it moves.
  var agentLog = document.getElementById("ag-log");
  var agentLines = agentLog ? Number(agentLog.getAttribute("data-count") || 0) : 0;
  function followAgentLog() {
    if (!agentLog) return;
    fetch("/api/bulk-agent?after=" + agentLines, { credentials: "same-origin", headers: { "X-CSRF-Token": csrf } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("status " + r.status)); })
      .then(function (d) {
        var now = document.getElementById("ag-now");
        var atEnd = agentLog.scrollTop + agentLog.clientHeight >= agentLog.scrollHeight - 24;
        (d.log || []).forEach(function (line) {
          var item = document.createElement("li");
          item.className = "ag-line";
          item.textContent = line;
          agentLog.insertBefore(item, now);
        });
        agentLines = d.total;
        if (atEnd) agentLog.scrollTop = agentLog.scrollHeight;
        var staged = document.getElementById("ag-staged");
        if (staged) staged.textContent = d.staged + (d.staged === 1 ? " video" : " videos") + " staged";
        if (d.stopping && now && !now.querySelector(".ag-stopping")) {
          var note = document.createElement("span");
          note.className = "muted ag-stopping";
          note.textContent = " Stopping after this step.";
          now.appendChild(note);
        }
      })
      .catch(function () { /* the next poll tries again */ });
  }
  if (agentLog) agentLog.scrollTop = agentLog.scrollHeight;

  function show(id, text) {
    var el = document.getElementById(id);
    if (el && text) el.textContent = text;
  }

  function poll() {
    fetch("/api/status", { credentials: "same-origin", headers: { "X-CSRF-Token": csrf } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("status " + r.status)); })
      .then(function (s) {
        var jobs = ["signin", "pull", "write", "push", "suggest", "translate", "subtitles", "transcribe", "chapters", "packaging", "variants", "outliers", "strategy", "competitors", "competitors_explain", "searchgaps_signin", "searchgaps", "searchgaps_rank", "channel_stats", "video_stats", "copilot", "bulk_agent", "links", "adcheck", "category", "pinned", "shorts", "comments", "assist", "channel_comments", "channel_sort", "thumb"];
        var running = jobs.some(function (name) { return s[name] && s[name].status === "running"; });
        if (running) {
          seenRunning = true;
          jobs.forEach(function (name) {
            show(name + "-msg", s[name] && s[name].status === "running" ? s[name].message : "");
          });
          followAgentLog();
          setTimeout(poll, 1500);
        } else {
          keepScroll();
          // Load the page again without the #anchor, so the browser does not jump to it; a replace, not a
          // reload, so browsers that cross-fade page loads (app.css) do so here too.
          window.location.replace(window.location.pathname + window.location.search);
        }
      })
      .catch(function () {
        // Server gone or a hiccup: try again a little later, but do not loop fast.
        setTimeout(poll, seenRunning ? 3000 : 5000);
      });
  }

  setTimeout(poll, 1000);
})();

// A button press or a finished job reloads the page (often with a #card anchor). Remember where the reader was so
// the page comes back at the same place instead of jumping to the card's top.
function keepScroll(form) {
  // A [data-keep-place] element stays at the same height on screen, even when text above it comes or goes.
  var mark = form && form.querySelector ? form.querySelector("[data-keep-place]") : null;
  if (!mark) {
    Array.prototype.some.call(document.querySelectorAll("[data-keep-place]"), function (el) {
      var r = el.getBoundingClientRect();
      if (r.height && r.top >= 0 && r.top < window.innerHeight) { mark = el; return true; }
      return false;
    });
  }
  var place = [window.scrollY, Date.now()];
  if (mark && mark.id) place.push(mark.id, mark.getBoundingClientRect().top);
  try { window.sessionStorage.setItem("yz-scroll:" + window.location.pathname, JSON.stringify(place)); } catch (e) { /* storage off */ }
}
document.addEventListener("submit", function (e) {
  var form = e.target;
  if ((form.getAttribute("method") || "").toLowerCase() !== "post" || form.target) return;
  keepScroll(form);
  // Ask the server to come back without the #anchor (see stay_in_place in app.py).
  var url = new URL(form.action, window.location.href);
  url.searchParams.set("stay", "1");
  form.action = url.toString();
});

// Video page: the tools and the checklist are folded <details data-keep>. The page reloads when a job
// finishes, so remember which ones the creator opened (per page, this tab only). Also open the one a link points
// into, and any that shows a running job or an error.
(function () {
  "use strict";
  var folds = Array.prototype.slice.call(document.querySelectorAll("details[data-keep]"));
  if (!folds.length) return;
  var key = "yz-open:" + window.location.pathname;
  var saved = [];
  try { saved = JSON.parse(window.sessionStorage.getItem(key) || "[]"); } catch (e) { saved = []; }
  function remember() {
    var open = folds.filter(function (d) { return d.open; }).map(function (d) { return d.id; });
    try { window.sessionStorage.setItem(key, JSON.stringify(open)); } catch (e) { /* storage off: nothing to keep */ }
  }
  function openTarget() {
    var id = decodeURIComponent(window.location.hash.slice(1));
    var target = id && document.getElementById(id);
    if (!target) return;
    var fold = target.closest("details[data-keep]");
    if (fold) { fold.open = true; target.scrollIntoView({ behavior: "smooth", block: "start" }); }
  }
  folds.forEach(function (d) {
    if (saved.indexOf(d.id) >= 0 || d.querySelector(".waiting, .flash.error")) d.open = true;
    d.addEventListener("toggle", remember);
  });
  var back = null;
  try {
    back = JSON.parse(window.sessionStorage.getItem("yz-scroll:" + window.location.pathname) || "null");
    window.sessionStorage.removeItem("yz-scroll:" + window.location.pathname);
  } catch (e) { back = null; }
  var params = new URLSearchParams(window.location.search);
  var go = params.get("go");
  if (go) {
    // Put the card back in the address bar as #anchor, without scrolling to it.
    params.delete("go");
    var q = params.toString();
    history.replaceState(null, "", window.location.pathname + (q ? "?" + q : "") + "#" + encodeURIComponent(go));
  }
  if (back && Date.now() - back[1] < 120000) {
    // Back from a reload this page caused: come back at exactly the same place (the result appears where
    // the button was), and keep the row open.
    var card = go && document.getElementById(go);
    var fold = card && card.closest("details[data-keep]");
    if (fold) fold.open = true;
    var mark = back[2] && document.getElementById(back[2]);
    var top = mark ? mark.getBoundingClientRect().top + window.scrollY - back[3] : back[0];
    window.scrollTo({ top: top, behavior: "instant" });
    document.documentElement.classList.remove("yz-restoring");
  } else {
    openTarget();
  }
  window.addEventListener("hashchange", openTarget);
})();

// Whatever happened above, never leave the page hidden (theme.js hides it while the place is restored).
document.documentElement.classList.remove("yz-restoring");

// Phone menu: the sidebar is one scrolling row there. Bring the current page's chip into view (centred) without
// scrolling the page itself.
(function () {
  "use strict";
  var bar = document.querySelector(".sidebar");
  var active = bar && bar.querySelector(".nav-item.active");
  if (!active || !window.matchMedia("(max-width: 899px)").matches) return;
  bar.scrollLeft = Math.max(0, active.offsetLeft - (bar.clientWidth - active.offsetWidth) / 2);
})();

// Jobs on every page: the top bar shows what is running (its menu links back to the video), and a job that
// finishes while you are elsewhere in Yezatas shows a card with a link to its result, plus a system notification
// when this tab is in the background and notifications are allowed.
(function () {
  "use strict";
  var box = document.getElementById("jobs");
  var toasts = document.getElementById("job-toasts");
  if (!box || !toasts || !window.fetch) return;
  var chip = box.querySelector(".jobs-chip");
  var chipText = box.querySelector(".jobs-chip-text");
  var menu = document.getElementById("jobs-menu");
  var KEY = "yz-jobs-seen";  // the jobs this tab saw running, so each finish is announced once (this tab only)
  var seen = {};
  try { seen = JSON.parse(window.sessionStorage.getItem(KEY) || "{}") || {}; } catch (e) { seen = {}; }
  function save() { try { window.sessionStorage.setItem(KEY, JSON.stringify(seen)); } catch (e) { /* storage off */ } }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }
  function ago(s) { return s < 60 ? "just now" : Math.round(s / 60) + " min ago"; }
  function row(j) {
    var running = j.status === "running";
    var a = el("a", "jobs-row" + (running ? " now" : ""));
    a.href = j.href;
    a.appendChild(running ? el("span", "spinner") : el("span", "jobs-dot " + (j.status === "done" ? "ok" : "bad")));
    var t = el("span", "jobs-row-text");
    t.appendChild(el("strong", "", running && j.message ? j.message : j.label));
    t.appendChild(el("span", "muted", (j.title || j.label) + (running ? " · started " + ago(j.seconds) : "")));
    a.appendChild(t);
    a.appendChild(el("span", "jobs-open", "Open"));
    return a;
  }
  function toast(j) {
    var ok = j.status === "done";
    var card = el("div", "job-toast " + (ok ? "ok" : "bad"));
    card.setAttribute("role", ok ? "status" : "alert");
    card.appendChild(el("span", "jobs-dot " + (ok ? "ok" : "bad")));
    var t = el("span", "jobs-row-text");
    t.appendChild(el("strong", "", j.label));
    t.appendChild(el("span", "muted", j.title || j.message || ""));
    card.appendChild(t);
    var open = el("a", "button " + (ok ? "primary" : "secondary"), j.video_id ? "Open video" : "Open");
    open.href = j.href;
    card.appendChild(open);
    var x = el("button", "job-toast-x", "×");
    x.type = "button";
    x.setAttribute("aria-label", "Close");
    x.addEventListener("click", function () { card.remove(); });
    card.appendChild(x);
    toasts.appendChild(card);
    setTimeout(function () { card.classList.add("in"); }, 20);
    setTimeout(function () { card.classList.remove("in"); setTimeout(function () { card.remove(); }, 300); }, 12000);
    if (document.hidden && window.Notification && Notification.permission === "granted") {
      try {
        var n = new Notification(j.label, { body: j.title || j.message || "", tag: "yz-" + j.job });
        n.onclick = function () { window.focus(); window.location.href = j.href; };
      } catch (e) { /* notifications refused here */ }
    }
  }

  var timer = 0;
  function poll() {
    fetch("/api/activity", { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("status " + r.status)); })
      .then(function (data) {
        var jobs = data.jobs || [];
        var running = jobs.filter(function (j) { return j.status === "running"; });
        var here = window.location.pathname;
        jobs.forEach(function (j) {
          var id = j.job + ":" + j.video_id;
          if (j.status === "running") { seen[id] = 1; return; }
          if (!seen[id]) return;
          delete seen[id];
          // The job's own page reloads itself and shows the result there; anywhere else, announce it.
          if (j.href.split("#")[0] !== here || document.hidden) toast(j);
        });
        save();
        box.hidden = jobs.length === 0;
        box.classList.toggle("running", running.length > 0);
        chipText.textContent = running.length
          ? (running.length === 1 ? running[0].label : running.length + " jobs running")
          : (jobs.length === 1 ? "1 finished" : jobs.length + " finished");
        menu.textContent = "";
        if (running.length) menu.appendChild(el("p", "jobs-h", "Running now"));
        running.forEach(function (j) { menu.appendChild(row(j)); });
        var finished = jobs.filter(function (j) { return j.status !== "running"; });
        if (finished.length) menu.appendChild(el("p", "jobs-h", "Finished"));
        finished.forEach(function (j) { menu.appendChild(row(j)); });
        clearTimeout(timer);
        timer = setTimeout(poll, running.length ? 2000 : 10000);
      })
      .catch(function () { clearTimeout(timer); timer = setTimeout(poll, 10000); });
  }
  chip.addEventListener("click", function () {
    var open = menu.hidden;
    menu.hidden = !open;
    chip.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) toasts.textContent = "";  // the menu lists the same finish; close its card
  });
  document.addEventListener("click", function (e) {
    if (!box.contains(e.target)) { menu.hidden = true; chip.setAttribute("aria-expanded", "false"); }
  });
  // Starting a job is a click: the moment a page may ask, once, to show notifications.
  document.addEventListener("submit", function () {
    if (window.Notification && Notification.permission === "default") {
      try { Notification.requestPermission(); } catch (e) { /* older browsers */ }
    }
  });
  poll();
})();
