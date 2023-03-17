#!/bin/bash

# main paper
JOB=paper
INPUT=main.tex
OUTPUT=build

# supplementary material
SUP_JOB=supplementary
SUP_INPUT=supplementary.tex

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

cd paper/

echo "started generating paper..."
mkdir -p $OUTPUT
latexmk -lualatex -interaction=nonstopmode -jobname=$OUTPUT/$JOB $INPUT
biber $OUTPUT/$JOB
latexmk -lualatex -interaction=nonstopmode -jobname=$OUTPUT/$JOB $INPUT
echo "paper generation done."

echo "started generating supplementary material..."
latexmk -lualatex -interaction=nonstopmode -jobname=$OUTPUT/$SUP_JOB $SUP_INPUT
echo "supplementary material generation done."

cd ..

