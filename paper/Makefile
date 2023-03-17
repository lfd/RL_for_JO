# Latex Makefile using latexmk

JOB = qrl_jo
JOB_SUPPLEMENTARY = supplementary
INPUT = main.tex
INPUT_SUPPLEMENTARY = supplementary.tex
OUTPUT = build

.PHONY: all clean

all: gen paper supplementary

paper: $(INPUT)
	mkdir -p $(OUTPUT)
	# -lualatex                 use lualatex for processing files to pdf and turn dvi/ps modes off
	latexmk -lualatex -interaction=nonstopmode -jobname=$(OUTPUT)/$(JOB) $^

supplementary: $(INPUT_SUPPLEMENTARY)
	mkdir -p $(OUTPUT)
	# -lualatex                 use lualatex for processing files to pdf and turn dvi/ps modes off
	latexmk -lualatex -interaction=nonstopmode -jobname=$(OUTPUT)/$(JOB_SUPPLEMENTARY) $^

gen:
	bash gen_img.sh

clean:
	rm -rf $(OUTPUT)
