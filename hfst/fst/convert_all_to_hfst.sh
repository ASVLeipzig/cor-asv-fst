#!/bin/bash

for filename in *.fsm; do
    prefix="$(cut -d'.' -f1 <<<"$filename")"
    hfst-fst2fst -t $filename -o $(printf "$prefix.hfst")
done
