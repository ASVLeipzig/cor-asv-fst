import hfst

import error_transducer as et

file_name = 'lang_mod_theta_0_000001.mod'

lm_fst = et.load_transducer(file_name)

lm_fst.substitute((' ', ' '), (hfst.EPSILON, hfst.EPSILON))
lm_fst.substitute(('_', '_'), (' ', ' '))


out = hfst.HfstOutputStream(filename=file_name + '.modified.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
out.write(lm_fst)
out.flush()
out.close()
