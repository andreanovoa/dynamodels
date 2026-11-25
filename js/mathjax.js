// MathJax config for pymdownx.arithmatex (generic mode); `\bm` is the tutorials' bold-vector macro.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: { bm: ["\\boldsymbol{#1}", 1] }
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" }
};
