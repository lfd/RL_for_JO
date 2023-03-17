#!/bin/bash

PLOT_SCRIPT=plots/plot.r

# in case the script is not started from within qce24_repro directory
if [ ! "${PWD}" = "/home/repro/qce24_repro" ]; then
    cd /home/repro/qce24_repro/
fi

echo "started generating plots..."
Rscript $PLOT_SCRIPT
cd ..
cp -r plots/img-tikz paper/
cd paper/
source gen_img.sh

echo "plot generation done."

cd ..
