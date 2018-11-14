# use like this:
# $ hfst-fst2txt file.hfst | python att2dot.py | dot -Tpng -ofile.png
# FIXME: preferably wrap libhfst's HfstPrintDot by extending the SWIG bindings

import sys
import math

print('digraph G { rankdir="LR"')
print('node [fontname="Tahoma",shape=circle,fontsize=14,fixedsize=true,fillcolor="grey",style=filled]')
print('edge [fontname="FreeMono",fontsize=16]')
escape = str.maketrans({'"': r'\"', '\\': r'\\'})
for i, line in enumerate(sys.stdin):
    if i >= 1000:
        print('cutting off input at line %d' % i, file=sys.stderr)
        break
    row = line.strip().split('\t')
    if len(row) >= 4:
        source_state, target_state, in_str, out_str, weight = row
        print('%s [label="%s"];' % (source_state, source_state))
        if float(weight) > 10:
            continue # prune away from graph (would be too small to be legible anyway)
        #weight = float(weight)+1 # weight seems to have no effect on dot/graphviz
        penwidth = 4*math.exp(-float(weight))
        if in_str == out_str:
            color = 'black'
        else:
            color = 'red'
        if source_state.startswith('@N.'):
            minlen = 0.5
        else:
            minlen = 1.0
        in_str = in_str.translate(escape)
        out_str = out_str.translate(escape)
        print('%s -> %s [label="%s:%s",penwidth=%f,color=%s,minlen=%f];' % (source_state, target_state, in_str, out_str, penwidth, color, minlen))
    elif len(row) == 2: # Final state
        print('%s [label="%s",shape=doublecircle];' % (row[0], row[0]))
print('}')
