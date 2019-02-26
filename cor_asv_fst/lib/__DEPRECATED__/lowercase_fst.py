import hfst

symbol_table_file = 'gesamt_dta_spaces.syms'
output_file = 'lowercase.hfst'


lowercase_fst = hfst.HfstBasicTransducer()
lowercase_fst.set_final_weight(0, 0.0)

tr = hfst.HfstBasicTransition(0, ' ', ' ', 0.0)
lowercase_fst.add_transition(0, tr)

with open(symbol_table_file) as f:

    for line in f:
        splitted = line.split('\t')

        lowercase = splitted[0]
        uppercase = lowercase.upper()

        tr = hfst.HfstBasicTransition(0, lowercase, lowercase, 0.0)
        lowercase_fst.add_transition(0, tr)

        if lowercase != uppercase:
            tr = hfst.HfstBasicTransition(0, uppercase, lowercase, 0.0)
            lowercase_fst.add_transition(0, tr)


lowercase_fst = hfst.HfstTransducer(lowercase_fst)

out = hfst.HfstOutputStream(filename=output_file, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
out.write(lowercase_fst)
out.flush()
out.close()
