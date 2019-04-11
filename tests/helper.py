from ocrd_cor_asv_fst.lib.helper import transducer_from_dict

import pynini
import unittest


class HelperTest(unittest.TestCase):

    def test_transducer_from_dict(self):
        testdata_in = { 'abc' : 3.5, 'düh' : 5.7, 'a\u0364hi' : 9.2 }
        tr = transducer_from_dict(testdata_in)
        self.assertIsInstance(tr, pynini.Fst)
        testdata_out = { str_in : float(weight) \
                         for str_in, str_out, weight in tr.paths().items() }
        for key in set(testdata_in.keys()) | set(testdata_out.keys()):
            self.assertAlmostEqual(
                testdata_in[key], testdata_out[key], places=5)

