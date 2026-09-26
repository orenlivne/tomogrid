# Rebuild the paper from the measured data in ../experiments/results.
PAPER = tomogrid_built

all: $(PAPER).pdf

figs/scaling.pdf figs/knobs.pdf: make_figures.py ../experiments/results/refinement.csv
	python3 make_figures.py

$(PAPER).tex: tomogrid.tex make_tables.py ../experiments/results/refinement.csv
	python3 make_tables.py

$(PAPER).pdf: $(PAPER).tex refs.bib figs/scaling.pdf figs/knobs.pdf
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

data:
	cd .. && python3 -m experiments.scaling && python3 -m experiments.battery

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out

.PHONY: all data clean
