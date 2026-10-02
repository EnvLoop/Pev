# LaTeX version of the technical report

`main.tex` is the conference-format version of [`../report.md`](../report.md); the compiled PDF is
[`../pev.pdf`](../pev.pdf). The text follows `report.md`; every result table is generated
from the JSON result files, never typed by hand.

```bash
make            # tables (checked against report.md) -> tectonic -> ../pev.pdf -> log check
make figures    # redraw figures/*.pdf (and ../figures/*.png,svg) with ../make_figures.py (needs uv)
make ENGINE=latexmk   # use a TeX Live installation instead of tectonic
```

| File | What |
|---|---|
| `main.tex`, `refs.bib` | Paper source and bibliography |
| `make_tables.py` | Writes `tables/*.tex` from `../data/{dev-rounds,hidden}.json`, `../../TEST_RESULTS.json` and `../../reference-dev.json`; `--check` requires every number to equal the corresponding table in `report.md` |
| `figures/*.pdf` | Vector figures drawn by `../make_figures.py` |
| `finalize_pdf.py` | Copies `main.pdf` to `../pev.pdf` without creation/modification dates |
| `neurips_2026.sty` | NeurIPS 2026 style file, used with `\usepackage[preprint]{neurips_2026}` (non-anonymous preprint mode) |

**Style file.** `neurips_2026.sty` is copied unmodified from the official NeurIPS 2026 formatting kit,
<https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip> (file SHA-256
`c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`; header: "last revision: January 2026", authors Roman
Garnett and the authors of `nips15submit_e.sty`). The file and the kit carry no licence statement; it is distributed by
NeurIPS for preparing papers and is not covered by this report's CC-BY-4.0 licence ([`../LICENSE.md`](../LICENSE.md)).
