// Open the links that leave the documentation (references, R packages, other
// projects' documentation) in a new tab; links within it stay in the same tab.
document.addEventListener("DOMContentLoaded", () => {
  for (const link of document.querySelectorAll("a[href]")) {
    if (link.host && link.host !== window.location.host) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  }
});
