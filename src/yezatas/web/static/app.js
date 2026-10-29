// Polls /api/status while a sign-in or a catalog pull is running, then reloads the page
// when a job finishes. Also stops double submits. No external code.
(function () {
  "use strict";

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

  // Coming back with the Back button must not show disabled buttons or stale state.
  window.addEventListener("pageshow", function (e) { if (e.persisted) window.location.reload(); });

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
        var running = s.signin.status === "running" || s.pull.status === "running";
        if (running) {
          seenRunning = true;
          show("signin-msg", s.signin.status === "running" ? s.signin.message : "");
          show("pull-msg", s.pull.status === "running" ? s.pull.message : "");
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
