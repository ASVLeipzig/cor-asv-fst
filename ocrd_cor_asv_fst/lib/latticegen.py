import pynini

from ..lib.sliding_window import lexicon_to_window_fst, process_string


class FSTLatticeGenerator:
    # TODO docstrings

    def __init__(self, lexicon_file, error_model_file = None, **kwargs):
        # load all transducers and build a model out of them
        self.lexicon_fst = pynini.Fst.read(lexicon_file)
        self.window_fst = lexicon_to_window_fst(
            self.lexicon_fst,
            kwargs['words_per_window'])
        self.window_fst.arcsort()
        self.error_fst = pynini.Fst.read(error_model_file) \
                         if error_model_file \
                         else None
        self.rejection_weight = kwargs['rejection_weight']
        self.beam_width = kwargs['beam_width']

    def lattice_from_string(self, string):
        lattice = process_string(
            string,
            (self.error_fst, self.window_fst),
            beam_width       = self.beam_width,
            rejection_weight = self.rejection_weight)
        return lattice

