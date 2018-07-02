#!/bin/bash

for filename in *.htsf; do
    prefix="$(cut -d'.' -f1 <<<"$filename")"
    hfst-fst2fst -t -b -i $filename -o $(printf "$prefix.ofst")
done
