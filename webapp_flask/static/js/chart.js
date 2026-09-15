/* Fetches the band-coverage Vega-Lite spec for the given text and renders
   it into #band-chart with vega-embed. Called once on page load with the
   currently-selected text (the text-switcher dropdown is a normal <select
   onchange=submit>, so a full page reload already re-invokes this with
   the new selection -- no separate AJAX wiring needed yet). */
function renderBandChart(textName) {
    const el = document.getElementById("band-chart");
    if (!el) return;
    fetch(`/chart/band-coverage.json?text=${encodeURIComponent(textName)}`)
        .then((r) => r.json())
        .then((spec) => vegaEmbed("#band-chart", spec, { actions: false }))
        .catch(() => {
            el.textContent = "Couldn't load the chart.";
        });
}
