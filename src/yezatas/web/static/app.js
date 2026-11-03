// Polls /api/status while a sign-in, a catalog pull, a push, a suggestion, a translation, subtitles, a transcript, chapters, a packaging check, variants, an outlier explanation, a strategy report, a copilot answer, a link check, an advertiser-friendly check, a comments fetch or a comment sort are running, then reloads the page when a job finishes.
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

  if (document.body.dataset.poll !== "1") return;

  var csrf = document.body.dataset.csrf || "";
  var seenRunning = false;

  function show(id, text) {
    var el = document.getElementById(id);
    if (el && text) el.textContent = text;
  }

  function poll() {
    fetch("/api/status", { credentials: "same-origin", headers: { "X-CSRF-Token": csrf } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error("status " + r.status)); })
      .then(function (s) {
        var jobs = ["signin", "pull", "write", "push", "suggest", "translate", "subtitles", "transcribe", "chapters", "packaging", "variants", "outliers", "strategy", "competitors", "competitors_explain", "searchgaps_signin", "searchgaps", "searchgaps_rank", "channel_stats", "video_stats", "copilot", "links", "adcheck", "category", "pinned", "shorts", "comments", "assist", "channel_comments", "channel_sort", "thumb"];
        var running = jobs.some(function (name) { return s[name] && s[name].status === "running"; });
        if (running) {
          seenRunning = true;
          jobs.forEach(function (name) {
            show(name + "-msg", s[name] && s[name].status === "running" ? s[name].message : "");
          });
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
function keepScroll() {
  try { window.sessionStorage.setItem("yz-scroll:" + window.location.pathname, JSON.stringify([window.scrollY, Date.now()])); } catch (e) { /* storage off */ }
}
document.addEventListener("submit", function (e) {
  var form = e.target;
  if ((form.getAttribute("method") || "").toLowerCase() !== "post" || form.target) return;
  keepScroll();
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
    // Back from a reload this page caused: come back at the same place, then glide to the card only if it is
    // out of view.
    window.scrollTo({ top: back[0], behavior: "instant" });
    document.documentElement.classList.remove("yz-restoring");
    var card = go && document.getElementById(go);
    if (card) {
      var fold = card.closest("details[data-keep]");
      if (fold) fold.open = true;
      var box = card.getBoundingClientRect();
      if (box.top > window.innerHeight || box.bottom < 0) {
        setTimeout(function () { card.scrollIntoView({ behavior: "smooth", block: "start" }); }, 150);
      }
    }
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
