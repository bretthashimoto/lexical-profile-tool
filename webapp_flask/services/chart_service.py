"""Builds the band-coverage chart (bar + cumulative line + 95%/98%
threshold rules) as an Altair chart, serialized to a Vega-Lite spec dict
for the client to render with vega-embed.

Ported from the layered chart in webapp/app.py's "Band coverage" section
-- same marks/encodings/colors, built from a list of dicts instead of a
pandas DataFrame (Altair 5 doesn't need pandas for this).
"""

from __future__ import annotations

import altair as alt

from .render_helpers import BAND_RAMP


def build_band_chart(result) -> dict:
    bands = sorted(result.band_token_counts.keys())
    chart_data = [
        {
            "band": result.band_label(b),
            "band_num": b,
            "pct_tokens": result.band_token_pct.get(b, 0.0),
            "cumulative_pct": result.cumulative_token_pct.get(b, 0.0),
        }
        for b in bands
    ]

    bar = alt.Chart(alt.Data(values=chart_data)).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
    ).encode(
        x=alt.X("band:N", sort=None, title="Frequency band", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("pct_tokens:Q", title="% of tokens", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("band_num:Q", scale=alt.Scale(range=BAND_RAMP), legend=None),
        tooltip=[
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("pct_tokens:Q", title="% tokens", format=".2f"),
            alt.Tooltip("cumulative_pct:Q", title="Cumulative %", format=".2f"),
        ],
    )
    line = alt.Chart(alt.Data(values=chart_data)).mark_line(color="#eb6834", point=True).encode(
        x=alt.X("band:N", sort=None),
        y=alt.Y("cumulative_pct:Q", scale=alt.Scale(domain=[0, 100])),
    )
    thresholds = alt.Chart(alt.Data(values=[{"y": 95}, {"y": 98}])).mark_rule(
        strokeDash=[4, 4], color="#898781",
    ).encode(y="y:Q")

    return (bar + line + thresholds).properties(height=380).to_dict()
