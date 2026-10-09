---
description: Build charts, dashboards, and visual reports as HTML files that Humphrey shows inline.
---
Use this skill when the user wants a chart, a dashboard, or any visual or interactive output.

Write one self-contained `.html` file to the working directory (e.g. `sales-chart.html`). Humphrey renders it inline in the conversation, inside a sandboxed frame about 760px wide that grows to fit the page's height. You don't need to paste it into chat; say what you made in one line.

## Rules

- Self-contained: inline CSS and JS. For charts use Chart.js from `<script src="/vendor/chart.umd.min.js"></script>` (no other CDNs; the frame may be offline).
- No page chrome: transparent background, no outer card or border, 16px body padding. Humphrey's card frames it.
- Titles in HTML, not on the canvas: an `h1` at 15px weight 500, then a one-line caption in the muted colour saying what the reader should notice.
- Charts: wrap each canvas in a `div` with an explicit height (240-360px) and set `maintainAspectRatio: false`. Faint gridlines on the value axis only, no axis borders, legend only when there is more than one series, tooltips on.
- Numbers use `font-variant-numeric: tabular-nums`. Format them for people (12.4k, 31%, £1.2m).
- Follow Humphrey's theme: dark by default, light when `<html data-theme="light">`. The attribute changes live, so read colours from the CSS variables below and redraw on change (see the template).
- Weights 400 and 500 only. Containers use the fold: `border-radius: 3px 12px 12px 12px`.

## Palette

Series colours, in order: `--s1` tan, `--s2` sage, `--s3` coral, `--s4` mist, `--s5` deep tan, `--s6` sand. For one series, use `--s1`. Use `--s2` for good/up and `--s3` for bad/down only when that meaning applies.

## Template

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monthly revenue</title>
<style>
  :root {
    --text: #e9e9ed; --muted: #9397ab; --grid: rgba(233, 233, 237, 0.08); --surface: #232532;
    --s1: #c8a87a; --s2: #7fae8f; --s3: #d08f76; --s4: #b2b6ca; --s5: #a88a5f; --s6: #e6d4b6;
  }
  :root[data-theme="light"] {
    --text: #23242c; --muted: #6e7079; --grid: rgba(20, 21, 28, 0.08); --surface: #fbfaf7;
    --s1: #856b46; --s2: #4f8461; --s3: #b0654a; --s4: #6e7079; --s5: #a88a5f; --s6: #c8a87a;
  }
  body { margin: 0; padding: 16px; background: transparent; color: var(--text);
         font: 14px/1.5 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  h1 { margin: 0; font-size: 15px; font-weight: 500; letter-spacing: -0.01em; }
  .caption { margin: 2px 0 16px; color: var(--muted); font-size: 13px; }
  .chart { position: relative; height: 300px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .kpi { padding: 12px 14px; border-radius: 3px 12px 12px 12px; background: var(--surface); }
  .kpi b { display: block; font-size: 22px; font-weight: 500; font-variant-numeric: tabular-nums; }
  .kpi span { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
  <h1>Monthly revenue</h1>
  <p class="caption">Revenue grew in five of six months; March was the outlier.</p>
  <div class="chart"><canvas id="revenue"></canvas></div>
  <script src="/vendor/chart.umd.min.js"></script>
  <script>
    const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    let chart;
    function draw() {
      chart?.destroy();
      Chart.defaults.font.family = "Inter, system-ui, sans-serif";
      Chart.defaults.color = css("--muted");
      chart = new Chart(document.getElementById("revenue"), {
        type: "bar",
        data: {
          labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
          datasets: [{ label: "Revenue", data: [12, 19, 8, 21, 24, 27],
                       backgroundColor: css("--s1"), borderRadius: 4, maxBarThickness: 36 }],
        },
        options: {
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, border: { display: false } },
            y: { grid: { color: css("--grid") }, border: { display: false }, ticks: { callback: (v) => `£${v}k` } },
          },
        },
      });
    }
    draw();
    new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  </script>
</body>
</html>
```

For a written report rather than a visual, write Markdown (`.md`) instead: it is also shown inline, with headings, tables, and Mermaid diagrams.
