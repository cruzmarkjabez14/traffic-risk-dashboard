from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Flag to indicate whether a real PDF generator is available
try:  # pragma: no cover
    from fpdf import FPDF  # type: ignore
    PDF_ENABLED = True
except Exception:  # pragma: no cover
    FPDF = None  # type: ignore
    PDF_ENABLED = False


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def make_zip_bytes(paths: Iterable[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


def build_pdf_summary(history_df: pd.DataFrame, thresholds: dict, metrics: Optional[dict]) -> bytes:
    # Fallback to TXT bytes if PDF is not available
    if not PDF_ENABLED:
        lines = [
            "Child Marriage Prediction Summary\n",
            f"Generated: {datetime.utcnow().isoformat()}Z\n",
            "\nThresholds:\n",
            json.dumps(thresholds, indent=2),
            "\n\nMetrics:\n",
            json.dumps(metrics or {}, indent=2),
            "\n\nRecent What-if entries (head):\n",
            history_df.head(20).to_csv(index=False),
        ]
        return "".join(lines).encode("utf-8")

    # Helper to coerce text to Latin-1 safe strings for classic FPDF
    def _safe(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        replacements = {
            "\u2014": "-",  # em dash
            "\u2013": "-",  # en dash
            "\u2012": "-",
            "\u2010": "-",
            "\u2011": "-",  # non-breaking hyphen
            "\u2212": "-",  # minus sign
            "\u2022": "*",  # bullet
            "\u2265": ">=",  # ≥
            "\u2264": "<=",  # ≤
            "\u00A0": " ",   # nbsp
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        # Final guarantee: replace non-Latin-1 with '?'
        return s.encode("latin1", errors="replace").decode("latin1")

    # Landscape A4, light modern styling
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_margins(10, 10, 10)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, _safe("Child Marriage Prediction - Summary"), ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, _safe(f"Generated: {datetime.utcnow().isoformat()}Z"), ln=True)

    # -------- Thresholds table --------
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 8, _safe("Thresholds"), ln=True)
    pdf.set_font("Arial", style="B", size=9)
    pdf.set_fill_color(238, 238, 238)
    # Header (wider for landscape)
    colw = [50, 35, 35]
    pdf.cell(colw[0], 7, _safe("Band"), border=1, fill=True)
    pdf.cell(colw[1], 7, _safe("Min"), border=1, fill=True)
    pdf.cell(colw[2], 7, _safe("Max"), border=1, ln=1, fill=True)
    pdf.set_font("Arial", size=9)
    bands = (thresholds or {}).get("bands", {})
    for band in ["high", "medium", "low"]:
        row = bands.get(band, {})
        pdf.cell(colw[0], 7, _safe(band.title()), border=1)
        pdf.cell(colw[1], 7, _safe(str(row.get("min", ""))), border=1)
        pdf.cell(colw[2], 7, _safe(str(row.get("max", ""))), border=1, ln=1)
    pdf.ln(2)

    # -------- Metrics table --------
    if metrics:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 8, _safe("Metrics"), ln=True)
        pdf.set_font("Arial", style="B", size=9)
        headers = ["Split", "AUC", "PR-AUC", "Brier", "Thr", "TN", "FP", "FN", "TP"]
        widths = [28, 20, 24, 22, 16, 18, 18, 18, 18]
        pdf.set_fill_color(238, 238, 238)
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, _safe(h), border=1, fill=True)
        pdf.ln(7)
        pdf.set_font("Arial", size=9)
        def _row(split: str, d: dict):
            conf = (d or {}).get("confusion", {})
            vals = [
                split,
                f"{(d or {}).get('auc', float('nan')):.3f}",
                f"{(d or {}).get('pr_auc', float('nan')):.3f}",
                f"{(d or {}).get('brier', float('nan')):.3f}",
                f"{(d or {}).get('threshold', float('nan')):.2f}",
                str(conf.get('tn', '')),
                str(conf.get('fp', '')),
                str(conf.get('fn', '')),
                str(conf.get('tp', '')),
            ]
            for v, w in zip(vals, widths):
                pdf.cell(w, 7, _safe(v), border=1)
            pdf.ln(7)
        if "valid" in metrics:
            _row("valid", metrics.get("valid", {}))
        if "test" in metrics:
            _row("test", metrics.get("test", {}))
        pdf.ln(2)

    # -------- What-if entries table --------
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 8, _safe("Recent What-if entries"), ln=True)
    pdf.set_font("Arial", style="B", size=8)
    cols = ["timestamp", "v012", "v106_label", "v190_label", "v025_label", "v024_label", "pred_prob", "band"]
    labels = ["Time", "Age", "Education", "Wealth", "Residence", "Region", "p", "Band"]
    widths2 = [56, 12, 36, 28, 24, 56, 18, 18]
    pdf.set_fill_color(238, 238, 238)
    for h, w in zip(labels, widths2):
        pdf.cell(w, 6, _safe(h), border=1, fill=True)
    pdf.ln(6)
    pdf.set_font("Arial", size=8)
    rows = history_df.tail(30)
    for _, r in rows.iterrows():
        vals = [
            r.get("timestamp", ""),
            str(r.get("v012", "")),
            r.get("v106_label", ""),
            r.get("v190_label", ""),
            r.get("v025_label", ""),
            r.get("v024_label", ""),
            f"{float(r.get('pred_prob', 0)):.3f}",
            r.get("band", ""),
        ]
        for v, w in zip(vals, widths2):
            pdf.cell(w, 6, _safe(v), border=1)
        pdf.ln(6)

    out = pdf.output(dest="S").encode("latin1", errors="replace")
    return out


def summary_filename() -> str:
    """Return a suggested filename extension based on availability of PDF."""
    return "summary.pdf" if PDF_ENABLED else "summary.txt"


