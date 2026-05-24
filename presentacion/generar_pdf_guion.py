"""Generate the exposition script PDF from guion_exposicion.md."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "guion_exposicion.md"
OUTPUT = ROOT / "guion_exposicion.pdf"


def clean_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u03b2": "beta",
        "\u03b1": "alpha",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u2192": "->",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ScriptPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Guion de exposicion - Proyecto ML Oro/Plata", 0, 1, "R")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Pagina {self.page_no()}", 0, 0, "C")


def add_markdown_line(pdf: ScriptPDF, raw_line: str) -> None:
    pdf.set_x(pdf.l_margin)
    line = clean_text(raw_line.rstrip())
    if not line:
        pdf.ln(3)
        return

    if line.startswith("# "):
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(25, 40, 55)
        pdf.multi_cell(0, 8, line[2:])
        pdf.ln(3)
        return

    if line.startswith("## "):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(35, 95, 120)
        pdf.multi_cell(0, 7, line[3:])
        pdf.ln(2)
        return

    if line.startswith("- "):
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 5.5, "  - " + line[2:])
        return

    if line.startswith("> "):
        pdf.set_font("Helvetica", "I", 10.5)
        pdf.set_text_color(65, 65, 65)
        pdf.multi_cell(0, 5.8, line[2:])
        return

    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5.5, line)


def main() -> None:
    pdf = ScriptPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        add_markdown_line(pdf, line)

    pdf.output(str(OUTPUT))


if __name__ == "__main__":
    main()
