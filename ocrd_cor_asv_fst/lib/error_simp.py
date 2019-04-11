"""
Create error correcting string transducers trained 
from paired OCR / ground truth text data.
"""

import difflib
import math
from nltk import ngrams
import pynini

from .helper import escape_for_pynini

# gap/epsilon needs to be a character so we can easily make a transducer from it,
#             but it must not ever occur in input
GAP_ELEMENT = u' ' # (nbsp) # '\0' # (nul breaks things in libhfst)

def get_confusion_dicts(gt_dict, raw_dict, max_n):
    """
    Take two dictionaries, mapping: id -> line for GT and OCR, 
    and align corresponding lines. 
    If they are different, split into n-grams for n={1,2,3},
    and count n-gram occurances.
    Return these counts as a list of dicts:
    [ignored, 1-grams, 2-grams, 3-grams]
    """

    corresponding_list = []     # list of tuples (gt_line, raw_line)
    difference_list = []        # list of tuples (gt_line, raw_line) with gt_line != raw_line

    # divide sentences in those containing OCR errors and the rest
    for key in gt_dict.keys():
        raw_line = raw_dict.get(key, None)
        gt_line = gt_dict[key]
        if raw_line != None:
            corresponding = (gt_line, raw_line)
            corresponding_list.append(corresponding)
            if raw_line != gt_line:
                difference_list.append(corresponding)

    # each dict in this list contains counts of character ngram confusions up to
    # the position in the list (thus, up to 3grams are considered);
    # position 0 is ignored:
    # [ignored, 1grams, 2grams, 3grams]
    confusion_dicts = [{}, {}, {}, {}]
    
    matcher = difflib.SequenceMatcher(isjunk=None, autojunk=False) # disable "junk" detection heuristics (mainly for source code)
    
    for gt_line, raw_line in corresponding_list:

        if not gt_line or not raw_line:
            continue
        if GAP_ELEMENT in gt_line or GAP_ELEMENT in raw_line:
            raise Exception('gap element must not occur in text', GAP_ELEMENT, raw_line, gt_line)
        
        # alignment of lines

        matcher.set_seqs(raw_line, gt_line)
        if matcher.quick_ratio() < 0.1 and len(gt_line) > 5:
            continue
        else:
            alignment = []
            for op, raw_begin, raw_end, gt_begin, gt_end in matcher.get_opcodes():
                if op == 'equal':
                    alignment.extend(zip(raw_line[raw_begin:raw_end], gt_line[gt_begin:gt_end]))
                elif op == 'replace': # not really substitution:
                    delta = raw_end-raw_begin-gt_end+gt_begin
                    if delta > 0: # replace+delete
                        alignment.extend(zip(raw_line[raw_begin:raw_end-delta], gt_line[gt_begin:gt_end]))
                        alignment.extend(zip(raw_line[raw_end-delta:raw_end], [GAP_ELEMENT]*(delta)))
                    if delta <= 0: # replace+insert
                        alignment.extend(zip(raw_line[raw_begin:raw_end], gt_line[gt_begin:gt_end+delta]))
                        alignment.extend(zip([GAP_ELEMENT]*(-delta), gt_line[gt_end+delta:gt_end]))
                elif op == 'insert':
                    alignment.extend(zip([GAP_ELEMENT]*(gt_end-gt_begin), gt_line[gt_begin:gt_end]))
                elif op == 'delete':
                    alignment.extend(zip(raw_line[raw_begin:raw_end], [GAP_ELEMENT]*(raw_end-raw_begin)))
                else:
                    raise Exception("difflib returned invalid opcode", op, "in", gt_line)
            assert raw_end == len(raw_line)
            assert gt_end == len(gt_line)
            
            if alignment:
                
                raw_aligned = ''.join(map(lambda x: x[0], alignment))
                gt_aligned = ''.join(map(lambda x: x[1], alignment))
                
                for n in range(1, max_n+1): # the ngrams which are considered
                    
                    raw_ngrams = ngrams(raw_aligned, n)
                    gt_ngrams = ngrams(gt_aligned, n)
                    
                    for raw_ngram, gt_ngram in zip(raw_ngrams, gt_ngrams):
                        #print(raw_ngram, gt_ngram)
                        raw_string = ''.join(raw_ngram)
                        gt_string = ''.join(gt_ngram)
                        
                        confusion_dicts[n][raw_string] = confusion_dicts[n].setdefault(raw_string, {})
                        confusion_dicts[n][raw_string][gt_string] = confusion_dicts[n][raw_string].setdefault(gt_string, 0) + 1
    
    return confusion_dicts


def preprocess_confusion_dict(confusion_dict):
    """
    Convert confusion dictionary (for one n),
    mapping: input_ngram, output_ngram -> count,
    to a list with relative frequencies 
    (in relation to the total number of that input ngram, 
    not of all input ngrams).
    Return a list of tuples:
    input_ngram, output_ngram, frequency.
    """
    
    #Convert list of form ((input_string, output_string),
    #count) into list of form (((input_string, output_string),
    #relative_frequency), excluding infrequent errors,
    #maybe smoothing (not implemented)."""

    frequency_list = []

    raw_items = confusion_dict.items()

    # count number of all occurrences
    total_freq = sum([sum(freq
                          for gt_ngram, freq in gt_dict.items())
                      for raw_ngram, gt_dict in raw_items])
    print('total edit count:', total_freq)

    # count number of ε-substitutions
    epsilon_freq = sum([gap_freq
                        for gap_ngram, gap_freq in confusion_dict.setdefault(GAP_ELEMENT, {}).items()])
    #epsilon_freq = sum([gap_freq for gap_ngram, gap_freq in confusion_dict[GAP_ELEMENT].items()])
    print('insertion count:', epsilon_freq)

    # set ε-to-ε transitions to number of all occurrences minus ε-substitutions
    # (because it models an occurrence of an ε that is not confused with an
    # existing character; this is needed for correctly calculating the
    # frequencies of ε to something transitions;
    # in the resulting (not complete) error transducer, these ε to ε transitions
    # are not preserved, but only transitions changing the input)
    if epsilon_freq != 0:
        confusion_dict[GAP_ELEMENT][GAP_ELEMENT] = total_freq - epsilon_freq

    for raw_ngram, gt_dict in raw_items:
        substitutions = gt_dict.items()
        total_freq = sum([freq for gt_ngram, freq in substitutions])

        for gt_ngram, freq in substitutions:
            frequency_list.append((raw_ngram, gt_ngram, freq / total_freq))

    #print(sorted(frequency_list, key=lambda x: x[2]))
    return frequency_list


def write_frequency_list(frequency_list, filename):
    """Write human-readable (as string) frequency_list to filename (tab-separated)."""

    with open(filename, mode='w', encoding='utf-8') as f:
        for raw_gram, gt_gram, freq in frequency_list:
            f.write(raw_gram.replace(GAP_ELEMENT, u'□') + u'\t' +
                    gt_gram.replace(GAP_ELEMENT, u'□') + u'\t' +
                    str(freq) + u'\n')
    return


def read_frequency_list(filename):
    """Read frequency_list from filename."""

    freq_list = []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            instr, outstr, freq = line.strip('\n').split('\t')
            freq_list.append((instr.replace(u'□', GAP_ELEMENT),
                              outstr.replace(u'□', GAP_ELEMENT),
                              float(freq)))
    return freq_list


def transducer_from_list(confusion_list, weight_threshold=7.0, identity_transitions=False):
    """
    Convert a list of tuples: input_gram, output_gram, relative_frequency,
    into a weighted transducer performing the given transductions with 
    the encountered probabilities. 
    If identity_transitions is True, then keep non-edit transitions.
    If weight_threshold is given, then prune away those transitions with 
    a weight higher than that.
    """

    mappings = []
    for in_str, out_str, relfreq in confusion_list:
        ilabel = escape_for_pynini(in_str).replace(GAP_ELEMENT, '')
        olabel = escape_for_pynini(out_str).replace(GAP_ELEMENT, '')
        weight = -math.log(relfreq)
        if (identity_transitions or ilabel != olabel) \
                and (ilabel or olabel) \
                and weight <= weight_threshold:
            mappings.append((ilabel, olabel, str(weight)))
    return pynini.string_map(mappings)


def is_punctuation_edit(raw_char, gt_char):
    """
    Return True iff an edit of raw_char to gt_char could be
    a punctuation edit.
    Punctuation characters only matter if on the GT side, i.e.
    allow edits from punctuation to alphanumeric characters,
    (because those often occur inside words), but forbid 
    edits from alphanumeric to punctuation characters,
    as well as punctuation-only edits (because those likely
    cannot be corrected with a lexicon).
    Whether or not that edit is part of a punctuation edit
    still depends on the context, though.
    """

    # no edit
    if raw_char == gt_char:
        return False

    # segmentation error
    if raw_char in [GAP_ELEMENT, " "] and gt_char in [GAP_ELEMENT, " "]:
        return False

    # edit to an alphanumeric character
    if gt_char == "\u0364" or gt_char != GAP_ELEMENT and gt_char.isalnum():
        return False

    # alphanumeric to epsilon or space
    if gt_char in [GAP_ELEMENT, " "] and (raw_char == "\u0364" or raw_char != GAP_ELEMENT and raw_char.isalnum()):
        return False

    # all other edits modify output punctuation
    return True


def no_punctuation_edits(confusion):
    """
    Take one confusion entry and return True iff 
    none of the n-gram positions contain edits that 
    would convert some character into punctuation.
    """
    
    for in_char, out_char in zip(confusion[0], confusion[1]):
        if is_punctuation_edit(in_char, out_char):
            return False
    return True


def compile_single_error_transducer(confusion_dict, preserve_punct=False):
    confusion_list = preprocess_confusion_dict(confusion_dict)
    if preserve_punct:
        confusion_list = list(filter(no_punctuation_edits, confusion_list))
    # create (non-complete) error_transducer and optimize it
    tr = transducer_from_list(confusion_list)
    tr.optimize()
    return tr


def combine_error_transducers(transducers, max_context, max_errors):

    def _universal_acceptor(symbol_table):
        fst = pynini.epsilon_machine()
        fst.set_input_symbols(symbol_table)
        fst.set_output_symbols(symbol_table)
        for x, y in symbol_table:
            if x > 0:
                fst.add_arc(0, pynini.Arc(x, x, 0, 0))
        return fst

    contexts = []
    for n in range(1,max_context+1):
        for m in range(1,n+1):
            contexts.append(list(range(m,n+1)))
    
    # FIXME refactor the merging of symbol tables into a separate function
    symtab = pynini.SymbolTable()
    for t in transducers:
        symtab = pynini.merge_symbol_table(symtab, t.input_symbols())
        symtab = pynini.merge_symbol_table(symtab, t.output_symbols())
    for t in transducers:
        t.relabel_tables(new_isymbols=symtab, new_osymbols=symtab)
    
    acceptor = _universal_acceptor(symtab)
    combined_transducers_dicts = []
    for context in contexts:
        print('Context: ', context)
        one_error = pynini.Fst()
        for n in context:
            one_error.union(transducers[n-1])
        
        for num_errors in range(1, max_errors+1):
            print('Number of errors:', num_errors)
            result_tr = acceptor.copy()
            result_tr.concat(one_error)
            result_tr.closure(0, num_errors)
            result_tr.concat(acceptor)
            result_tr.arcsort()
            combined_transducers_dicts.append({
                'max_error' : num_errors,
                'context' : ''.join(map(str, context)),
                'transducer' : result_tr })
    return combined_transducers_dicts

