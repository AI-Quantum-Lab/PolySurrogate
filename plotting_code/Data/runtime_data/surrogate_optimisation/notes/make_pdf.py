from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_margins(25, 20, 25)

# Title
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "L1 Regularisation Values for Inverse-Design", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "Surrogate Optimisation", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 6, "Extracted: 2026-06-20", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(6)

# Section: Extracted values
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Extracted lambda_L1 Values", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 6,
    "The table below lists the L1 regularisation strength extracted from every "
    "JSON file in runtime_data/surrogate_optimisation/. "
    "No files were missing and no lambda_l1 key was absent from any file.",
    new_x="LMARGIN", new_y="NEXT"
)
pdf.ln(4)

# Table
col_w = [38, 38, 58, 40]
headers = ["Target state", "Node number", "File", "lambda_L1"]

# Header row
pdf.set_fill_color(220, 220, 220)
pdf.set_font("Helvetica", "B", 10)
for h, w in zip(headers, col_w):
    pdf.cell(w, 8, h, border=1, fill=True, align="C")
pdf.ln()

rows = [
    ("GHZ", "4", "GHZ/4n.json", "1e-4"),
    ("GHZ", "6", "GHZ/6n.json", "1e-5"),
    ("GHZ", "8", "GHZ/8n.json", "1e-6"),
    ("W",   "4", "W/4n.json",   "1e-4"),
    ("W",   "6", "W/6n.json",   "1e-5"),
    ("W",   "8", "W/8n.json",   "1e-6"),
    ("LC",  "4", "LC/4n.json",  "1e-4"),
    ("LC",  "6", "LC/6n.json",  "1e-5"),
    ("LC",  "8", "LC/8n.json",  "1e-6"),
]

pdf.set_font("Helvetica", "", 10)
fill = False
for i, (state, n, fname, val) in enumerate(rows):
    bg = (245, 245, 245) if fill else (255, 255, 255)
    pdf.set_fill_color(*bg)
    pdf.cell(col_w[0], 7, state, border=1, fill=True, align="C")
    pdf.cell(col_w[1], 7, n,     border=1, fill=True, align="C")
    pdf.cell(col_w[2], 7, fname, border=1, fill=True, align="L")
    pdf.cell(col_w[3], 7, val,   border=1, fill=True, align="C")
    pdf.ln()
    # alternate shading per group of 3
    if (i + 1) % 3 == 0:
        fill = not fill

pdf.ln(6)

# Section: Pattern
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Pattern", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 6,
    "The regularisation strength is identical across all three target states "
    "(GHZ, W, LC) and decreases by one order of magnitude with each two-node "
    "increase in system size:",
    new_x="LMARGIN", new_y="NEXT"
)
pdf.ln(2)
bullets = [
    "4-node:  lambda = 1e-4",
    "6-node:  lambda = 1e-5",
    "8-node:  lambda = 1e-6",
]
for b in bullets:
    pdf.cell(8, 6, "", new_x="RIGHT", new_y="TOP")
    pdf.cell(4, 6, chr(149), new_x="RIGHT", new_y="TOP")
    pdf.cell(0, 6, b, new_x="LMARGIN", new_y="NEXT")

pdf.ln(6)

# Section: Suggested Methods sentence
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Suggested Methods Sentence", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "I", 10)
pdf.set_fill_color(245, 248, 255)
pdf.multi_cell(0, 7,
    '"The L1 regularisation strength was set to lambda = 1e-4, 1e-5, and 1e-6 '
    'for the 4-, 6-, and 8-node systems, respectively, applied uniformly across '
    'all target states (GHZ, W, and LC)."',
    border=1, fill=True, new_x="LMARGIN", new_y="NEXT"
)

out = "/home/bo48god/quick_tests/Plotting_codes/Data/runtime_data/surrogate_optimisation/lambda_l1_summary.pdf"
pdf.output(out)
print(f"PDF saved to: {out}")
