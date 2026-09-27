# Rebuild the paper from the measured data in ../experiments/results.
PAPER = tomogrid

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

data:
	cd .. && python3 -m bdlines.experiments 1 && python3 -m experiments.spect_bd

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out

.PHONY: all data clean
