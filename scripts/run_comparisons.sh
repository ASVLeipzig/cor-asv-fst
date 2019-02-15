#!/bin/bash

NUM_PROCESSES=8
INPUT_SUF=Fraktur4.txt
GT_SUF=gt.txt
TESTDATA_DIR=/disk/data/ocr-d/dta19-reduced/testdata/

# take the first argument as prefix, if it is supplied; default is 'cor'
if [ ! -z "$1" ]; then
  PREFIX=$1
else
  PREFIX='cor'
fi

experiment() {
  MODEL_NAME=$1
  REJECTION_WEIGHT=$2
  COMPOSITION_DEPTH=$3
  OUTPUT_SUF="$PREFIX-$MODEL_NAME-J$REJECTION_WEIGHT-W$COMPOSITION_DEPTH.txt"
  TIME=$(/usr/bin/time -f '%e' nice python3 process_test_data.py \
    -I $INPUT_SUF -O $OUTPUT_SUF -P bracket -W $COMPOSITION_DEPTH \
    -J $REJECTION_WEIGHT -Q $NUM_PROCESSES $TESTDATA_DIR \
    2>&1 1>/dev/null | tail -n 1)
  RESULT=$(python3 evaluate_correction.py \
    -I $INPUT_SUF -O $OUTPUT_SUF -G $GT_SUF -M Levenshtein $TESTDATA_DIR \
    | tail -n 1 | grep -o '[0-9.]\+$')
  echo -e "$OUTPUT_SUF\t$RESULT\t$TIME"
}

RESULTS_FILE=RESULTS-$(date +%Y-%m-%d-%H-%M).txt
cat /dev/null > $RESULTS_FILE

echo "# TESTDATA_DIR=$TESTDATA_DIR" >> $RESULTS_FILE
echo "# INPUT_SUF=$INPUT_SUF" >> $RESULTS_FILE
echo "# GT_SUF=$GT_SUF" >> $RESULTS_FILE
echo "# NUM_PROCESSES=$NUM_PROCESSES" >> $RESULTS_FILE
echo "# GIT_BRANCH=$(git branch | grep '^*' | sed 's/^* //')" >> $RESULTS_FILE
echo "# format: output_suf <tab> error_rate <tab> processing_time" >> $RESULTS_FILE

# training lexicon, old error model
unlink fst/lexicon.hfst
unlink fst/error.hfst
ln -s lexicon_transducer_dta.hfst fst/lexicon.hfst
ln -s max_error_3_context_23_dta.hfst fst/error.hfst
experiment old 1.5 1 >> $RESULTS_FILE

# training lexicon, ST error model
unlink fst/lexicon.hfst
unlink fst/error.hfst
ln -s lexicon_transducer_dta.hfst fst/lexicon.hfst
ln -s error_st.fst fst/error.hfst
experiment st 3 1 >> $RESULTS_FILE
