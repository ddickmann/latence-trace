# Heatmap — structured payload + copy-paste HTML fragment

The heatmap is a **convenience surface** for IDE plugins, enterprise
dashboards, and email alerts. It does not change how groundedness is
measured — it only re-projects the already-computed scores, file
attribution, and session signals into a render-ready structure.

Two flavours ship on the same endpoint:

- **`heatmap`** — structured JSON any UI can bind to. Ships by default
  on every scored response.
- **`heatmap_html`** — a single self-contained `<div>` with inline CSS.
  Drop into any page, email template, or Markdown preview. Opt-in via
  `heatmap_format: "html"`.

The endpoint only emits the HTML fragment when explicitly requested
(the data path is cheap; HTML is a few extra KB that most clients
don't need).

## Request flag

```jsonc
POST /groundedness
{
  "scoring_mode": "rag",            // or "code"
  "query_text": "…",
  "response_text": "…",
  "support_units": [ … ],

  // "none" | "data" | "html"  -- default "data"
  "heatmap_format": "html"
}
```

| `heatmap_format` | `heatmap` | `heatmap_html` |
| --- | --- | --- |
| `"none"`  | omitted | omitted |
| `"data"`  | emitted | omitted |
| `"html"`  | emitted | emitted |

## Response shape

```jsonc
{
  "heatmap": {
    "summary": {
      "headline_score": 0.71,
      "risk_band": "yellow",
      "recommendation": "continue",
      "groundedness_pct": 0.71,
      "dead_weight_pct": 0.34,
      "reason_code_histogram": {
        "never_won_argmax": 2,
        "dominated_by_single_file": 1
      }
    },
    "tokens": [
      { "index": 0, "token": "Saturn's",  "score": 0.92, "band": "green" },
      { "index": 1, "token": "rings",     "score": 0.88, "band": "green" },
      { "index": 2, "token": "are",       "score": 0.42, "band": "amber" }
    ],
    "files": [
      {
        "path": "docs/saturn_rings.md",
        "owner_share": 0.72,
        "band": "green",
        "reason_codes": [],
        "dead_weight": false
      },
      {
        "path": "docs/bamboo_growth.md",
        "owner_share": 0.0,
        "band": "red",
        "reason_codes": ["never_won_argmax"],
        "dead_weight": true
      }
    ],
    "thresholds": {
      "token_green_min": 0.60,
      "token_amber_min": 0.35,
      "file_green_min":  0.20,
      "file_amber_min":  0.05
    }
  },

  // Only present when heatmap_format == "html".
  "heatmap_html": "<div class=\"lt-heatmap\">…</div>"
}
```

### Bands

The same three-band vocabulary is used for tokens and files, with
different cut-offs that reflect their different scales:

| Band     | Token (`heatmap_score`)       | File (`owner_share`)          |
| -------- | ----------------------------- | ----------------------------- |
| `green`  | `≥ token_green_min` (0.60)    | `≥ file_green_min` (0.20)     |
| `amber`  | `≥ token_amber_min` (0.35)    | `≥ file_amber_min` (0.05)     |
| `red`    | below `amber_min`             | below `amber_min`             |

Exact cut-offs are echoed on `heatmap.thresholds` so any UI can
reproduce the bucketing server-side or explain it in tool-tips.

## Copy-paste HTML fragment

When `heatmap_format == "html"` the response carries
`heatmap_html` — a single `<div class="lt-heatmap">` with an inline
`<style>` block. It is safe to drop into:

- a React dangerouslySetInnerHTML block,
- a Markdown preview pane,
- a transactional email template that supports inline styles.

The fragment never pulls external stylesheets, fonts, or scripts. It
uses system fonts and renders at ≤ 720 px wide. CSS class names are
scoped under `.lt-heatmap` so collisions with the host page are
unlikely. The three band colours are defined as regular class rules,
not as CSS custom properties (so email clients that do not support
`var()` still render the right colours):

```css
.lt-heatmap .lt-band-green { background: #d1fae5; color: #065f46; }
.lt-heatmap .lt-band-amber { background: #fef3c7; color: #92400e; }
.lt-heatmap .lt-band-red   { background: #fee2e2; color: #991b1b; }
```

### Minimal data-only template

If you prefer to render the data payload yourself:

```html
<style>
  .heatmap-tok { padding: 1px 4px; margin: 1px; border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .heatmap-tok.green { background: #d1fae5; color: #065f46; }
  .heatmap-tok.amber { background: #fef3c7; color: #92400e; }
  .heatmap-tok.red   { background: #fee2e2; color: #991b1b; }
</style>

<div class="heatmap">
  {{#each heatmap.tokens}}
    <span class="heatmap-tok {{band}}" title="score={{score}}">{{token}}</span>
  {{/each}}
</div>
```

## Rollup-level heatmap

The rollup endpoint also accepts `heatmap_format: "html"` and returns
a session-level heatmap card summarising the whole conversation. See
[`docs/rollup.md`](rollup.md).

## Token list cap

The `tokens` list is capped at **512 entries** (and `files` at **20**)
to keep wire size bounded for very long responses. Scoring itself is
unaffected — only the convenience projection is truncated.
