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
