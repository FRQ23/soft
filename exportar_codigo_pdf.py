"""
exportar_codigo_pdf.py — Exporta todos los .py del proyecto a un solo PDF
=========================================================================
Uso:  python3 exportar_codigo_pdf.py
Salida: codigo_fuente.pdf
"""

import os
import glob
from fpdf import FPDF


class PDFCodigo(FPDF):
    def header(self):
        self.set_font("CourierUni", "B", 10)
        if hasattr(self, "_current_file"):
            self.cell(0, 8, self._current_file, border=0, new_x="LMARGIN", new_y="NEXT")
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("CourierUni", "", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")


def exportar_a_pdf(output="codigo_fuente.pdf"):
    base = os.path.dirname(os.path.abspath(__file__))

    # Buscar todos los .py recursivamente, excluyendo __pycache__ y este script
    archivos = sorted(glob.glob(os.path.join(base, "**", "*.py"), recursive=True))
    archivos = [
        f for f in archivos
        if "__pycache__" not in f
        and os.path.basename(f) != "exportar_codigo_pdf.py"
    ]

    if not archivos:
        print("No se encontraron archivos .py")
        return

    pdf = PDFCodigo(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)

    # Fuente Unicode monoespaciada (Courier New en macOS)
    font_path = "/System/Library/Fonts/Supplemental/Courier New.ttf"
    pdf.add_font("CourierUni", "", font_path)
    pdf.add_font("CourierUni", "B", "/System/Library/Fonts/Supplemental/Courier New Bold.ttf")

    for filepath in archivos:
        rel = os.path.relpath(filepath, base)
        print(f"  Agregando: {rel}")

        pdf._current_file = rel
        pdf.add_page()
        pdf.set_font("CourierUni", size=7.5)

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for num, line in enumerate(f, 1):
                line = line.rstrip("\n\r")
                # Prefijo con número de línea
                texto = f"{num:4d} | {line}"
                pdf.cell(0, 3.5, texto, new_x="LMARGIN", new_y="NEXT")

    out_path = os.path.join(base, output)
    pdf.output(out_path)
    print(f"\n  PDF generado: {output}  ({len(archivos)} archivos)")


if __name__ == "__main__":
    print("Exportando archivos .py a PDF …\n")
    exportar_a_pdf()
