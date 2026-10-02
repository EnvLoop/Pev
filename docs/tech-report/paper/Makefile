# Build the LaTeX version of the tech report.
#   make            tables (checked against report.md) + PDF, copied to ../pev.pdf
#   make figures    redraw the vector figures (runs ../make_figures.py with uv; also refreshes ../figures/*.png,svg)
#   make check      fail on undefined references/citations or overfull boxes in the last build log
# Engine: tectonic (default; fetches the LaTeX packages it needs). With TeX Live: make ENGINE=latexmk
ENGINE ?= tectonic
PDF    := ../pev.pdf
ROOT   := ../../..

.PHONY: all tables figures check clean

all: $(PDF) check

tables:
	python3 make_tables.py --check

figures:
	cd $(ROOT) && uv run docs/tech-report/make_figures.py

main.pdf: main.tex refs.bib neurips_2026.sty $(wildcard figures/*.pdf) tables
ifeq ($(ENGINE),tectonic)
	tectonic --keep-logs --keep-intermediates main.tex
else
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
endif

$(PDF): main.pdf
	uv run finalize_pdf.py main.pdf $(PDF)   # drops the PDF creation/modification dates

check:
	@! grep -nE "undefined|Overfull|Citation .* undefined|Reference .* undefined|There were undefined" main.log \
	  || (echo "main.log has warnings above" && false)
	@echo "main.log: no undefined references or citations, no overfull boxes"

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk main.synctex.gz
