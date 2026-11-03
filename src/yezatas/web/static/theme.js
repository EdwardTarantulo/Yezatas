// Sets the colour theme on <html> before the page is drawn: the creator's choice from the top bar toggle, else the
// system setting. Loaded in <head> without defer, so a dark page never flashes light. The toggle is wired in app.js.
(function () {
  "use strict";
  var KEY = "yezatas-theme";
  var root = document.documentElement;
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  var choice = null;  // kept in memory too, for browsers that refuse localStorage
  try {
    var saved = window.localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") choice = saved;
  } catch (e) { /* storage blocked: follow the system */ }

  function current() { return choice || (media && media.matches ? "dark" : "light"); }
  function apply() {
    root.setAttribute("data-theme", current());
    document.dispatchEvent(new Event("yezatas-theme"));
  }
  function set(theme) {
    choice = theme;
    try { window.localStorage.setItem(KEY, theme); } catch (e) { /* the choice lasts for this page only */ }
    apply();
  }

  apply();
  // Coming back from a button press (app.js keeps the place): where the first paint cannot be held until app.js
  // has scrolled, keep the page invisible for that moment instead of showing its top.
  try {
    var link = document.createElement("link");
    if (!(link.relList && link.relList.supports && link.relList.supports("expect"))
        && window.sessionStorage.getItem("yz-scroll:" + window.location.pathname)) {
      root.classList.add("yz-restoring");
    }
  } catch (e) { /* storage blocked: nothing kept */ }
  if (media && media.addEventListener) media.addEventListener("change", apply);
  window.yezatasTheme = { current: current, set: set };
})();
