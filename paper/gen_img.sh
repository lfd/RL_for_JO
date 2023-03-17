#! /bin/bash

for d in img-gen build; do
    if [ ! -d ${d} ]; then
	mkdir $d
    fi
done
    
(cd plots && R CMD BATCH plot.r /dev/stdout)


for file in img-tikz/*.tex; do
    file_base=`basename $file`;
    
    ## NOTE: We use lualatex on purpose here because it lifts any memory
    ## limitations that classical TeX implementations have, which is important
    ## for complicated TikZ pictures.
    cat img.tex | sed -e "s/FILE/$file_base/" | \
	lualatex -output-directory build/ -jobname `basename $file_base .tex` && \
	mv build/`basename $file_base .tex`.pdf img-gen/
done
