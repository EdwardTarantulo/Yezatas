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
        var jobs = ["signin", "pull", "write", "push", "suggest", "translate", "subtitles", "transcribe", "chapters", "packaging", "variants", "outliers", "strategy", "copilot", "links", "adcheck", "shorts", "comments", "assist"];
        var running = jobs.some(function (name) { return s[name] && s[name].status === "running"; });
        if (running) {
          seenRunning = true;
          jobs.forEach(function (name) {
            show(name + "-msg", s[name] && s[name].status === "running" ? s[name].message : "");
          });
          setTimeout(poll, 1500);
        } else {
          window.location.reload();
        }
      })
      .catch(function () {
        // Server gone or a hiccup: try again a little later, but do not loop fast.
        setTimeout(poll, seenRunning ? 3000 : 5000);
      });
  }

  setTimeout(poll, 1000);
})();
