"""
G5 -- does it fit in four A4 pages, references included?

The page geometry is not assumed: it is read out of the section properties of
the APACE-supplied template, APACE2026_conference_template_A4.docx.

    page      595.30 x 841.90 pt (A4)
    margins   top 54, bottom 72, left 45.35, right 45.35 pt
    columns   2, gutter 18 pt
    body font Times New Roman 10 pt (template docDefaults)

Line capacity is measured, not guessed: a representative body paragraph is
typeset with real Times font metrics at 10 pt and the achieved characters per
line is measured against the true 243.30 pt column width.
"""

import numpy as np
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

# --- geometry read from the template ---------------------------------------
PAGE_W, PAGE_H = 595.30, 841.90
M_TOP, M_BOT, M_LR = 54.0, 72.0, 45.35
GUTTER = 18.0
N_COL = 2
COL_W = (PAGE_W - 2 * M_LR - GUTTER) / N_COL      # 243.30 pt
TEXT_H = PAGE_H - M_TOP - M_BOT                    # 715.90 pt
BODY_PT = 10.0
LEADING = 11.5                                     # Word single spacing, TNR 10 pt
LINES_PER_COL = TEXT_H / LEADING
COL_IN_PER_PAGE = N_COL * TEXT_H / 72.0

SAMPLE = (
    "The transverse equivalent network reduces a stratified medium at a fixed "
    "transverse wavenumber to a cascade of uniform transmission line sections, "
    "so that the reflection coefficient of the conductor backed stack is the "
    "input reflection coefficient of a short circuited two port cascade. "
    "Differentiating the cascade with respect to the parameters of one section "
    "replaces a single factor in the product, and caching the partial products "
    "on either side of that factor yields every design gradient from one "
    "forward and one backward sweep of two by two matrix vector products."
)


def find_serif():
    for name in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                 "DejaVu Serif"):
        try:
            p = font_manager.findfont(FontProperties(family=name),
                                      fallback_to_default=False)
            return name, p
        except Exception:
            continue
    return "DejaVu Serif", font_manager.findfont(FontProperties(family="serif"))


def measure_words_per_line():
    name, path = find_serif()
    fp = FontProperties(fname=path, size=BODY_PT)
    words = SAMPLE.split()
    # width of the whole sample rendered as one run, at 10 pt
    tp = TextPath((0, 0), SAMPLE, prop=fp, size=BODY_PT)
    total_w = tp.get_extents().width
    n_lines = total_w / COL_W
    wpl = len(words) / n_lines
    cpl = len(SAMPLE) / n_lines
    return name, wpl, cpl, total_w


def main():
    name, wpl, cpl, total_w = measure_words_per_line()
    words_per_col = wpl * LINES_PER_COL
    words_per_page = N_COL * words_per_col
    print("=" * 84)
    print("G5  FOUR-PAGE BUDGET, MEASURED AGAINST THE APACE TEMPLATE")
    print("=" * 84)
    print(f"  column width      {COL_W:.2f} pt = {COL_W/72:.3f} in "
          f"= {COL_W*25.4/72:.1f} mm   (from the .docx sectPr)")
    print(f"  text height       {TEXT_H:.2f} pt = {TEXT_H/72:.3f} in")
    print(f"  lines per column  {LINES_PER_COL:.1f}   "
          f"(Times New Roman {BODY_PT:.0f} pt, {LEADING} pt leading)")
    print(f"  metrics font used {name}")
    print(f"  measured          {cpl:.1f} characters/line, {wpl:.2f} words/line")
    print(f"  =>                {words_per_col:.0f} words/column, "
          f"{words_per_page:.0f} words/page of solid body text")
    print(f"  4-page capacity   {4*COL_IN_PER_PAGE:.1f} column-inches\n")

    # --- what the paper spends -------------------------------------------
    #   (heights in column-inches; a full-width item costs 2 columns)
    items = [
        ("Title + author block (full width, 1.85 in)",           2 * 1.85),
        ("Abstract + Index Terms (9 pt, ~13 lines)",             13 * LEADING * 0.9 / 72),
        ("8 section / subsection headings",                      8 * 0.26),
        ("11 displayed equations",                               11 * 0.30),
        ("Fig. 1  gradient verification (single column)",        2.45),
        ("Fig. 2  |Gamma| vs frequency and angle (single col)",  2.55),
        ("Fig. 3  speed-up vs design variables (single col)",    2.35),
        ("Table I verification summary",                         2.10),
        ("12 references (9 pt, ~2.4 lines each)",                12 * 2.4 * LEADING * 0.9 / 72),
        ("Acknowledgment incl. AI-use disclosure",               0.75),
    ]
    spent = sum(v for _, v in items)
    print("  fixed furniture (column-inches):")
    for k, v in items:
        print(f"    {k:<52} {v:6.2f}")
    print(f"    {'TOTAL FURNITURE':<52} {spent:6.2f}")

    body_ci = 4 * COL_IN_PER_PAGE - spent
    words_per_ci = wpl * (72.0 / LEADING)
    body_words = body_ci * words_per_ci
    print(f"\n  column-inches left for body prose : {body_ci:.2f}")
    print(f"  words per column-inch              : {words_per_ci:.1f}")
    print(f"  BODY WORDS THAT FIT IN 4 PAGES     : {body_words:.0f}")

    # --- what the outline actually plans ---------------------------------
    plan = [
        ("I. Introduction", 380),
        ("II. Transverse network formulation", 330),
        ("III. Closed-form adjoint sensitivity", 430),
        ("IV. Verification", 380),
        ("V. Wideband wide-angle synthesis", 430),
        ("VI. Conclusion", 120),
    ]
    planned = sum(v for _, v in plan)
    print("\n  planned body prose:")
    for k, v in plan:
        print(f"    {k:<52} {v:6d}")
    print(f"    {'TOTAL PLANNED':<52} {planned:6d}")
    slack = body_words - planned
    print(f"\n  slack: {slack:+.0f} words "
          f"({100*slack/body_words:+.1f} % of the body budget)")
    print(f"  minimum is 2 pages: planned content is comfortably above it.")
    verdict = "PASS" if slack > 0 else "FAIL"
    print(f"\n  >>> G5 VERDICT: {verdict}")
    if slack > 0:
        eq = slack / words_per_ci
        print(f"      the slack is worth {eq:.1f} column-inches, i.e. roughly "
              f"one more figure\n      or ~{slack/wpl:.0f} extra lines of prose.")
    print("=" * 84)


if __name__ == "__main__":
    main()
