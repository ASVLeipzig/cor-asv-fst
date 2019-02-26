from __future__ import absolute_import
import itertools

from ocrd import Processor, MIMETYPE_PAGE
from ocrd.validator.page_validator import PageValidator, ConsistencyError
from ocrd.utils import \
    getLogger, concat_padded, xywh_from_points, points_from_xywh
from ocrd.model.ocrd_page import \
    from_file, to_xml, GlyphType, CoordsType, TextEquivType
from ocrd.model.ocrd_page_generateds import \
    MetadataItemType, LabelsType, LabelType

import networkx as nx
import hfst

from ocrd_keraslm.lib import Rater
from .config import OCRD_TOOL
from ..lib.sliding_window_no_flags import Corrector # ???

LOG = getLogger('processor.FSTCorrection')
MAX_WINDOW_SIZE = 2

class FSTCorrection(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-cor-asv-fst-process']
        kwargs['version'] = OCRD_TOOL['version']
        super(FSTCorrection, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace:
            # no parameter/workspace for --dump-json or --version (no
            # processing)
            return
        
        # we need a language model for decoding hypotheses graphs:
        self.rater = Rater(logger=LOG)
        self.rater.load_config(self.parameter['keraslm_file'])
        # overrides for incremental mode necessary before compilation:
        self.rater.stateful = False         # no implicit state transfer
        self.rater.incremental = True       # but explicit state transfer
        self.rater.configure()
        self.rater.load_weights(self.parameter['keraslm_file'])
        
        # initialisation for FST models goes here...
        self.corrector = Corrector(
            self.parameter['errorfst_file'],
            self.parameter['lexiconfst_file']) # ???
    
    def process(self):
        """
        Transduce textual annotation into hypotheses graphs for each line, then
        decode their best paths incrementally, producing output files with
        FST+LM scores.
        
        Read and parse PAGE input files one by one, splitting its lines into
        windows of 
        """
         # word or glyph input (output will always be on word level)
        level = self.parameter['textequiv_level']
        beam_width = self.parameter['beam_width']
        lm_weight = self.parameter['lm_weight']
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file)
            pcgts = from_file(self.workspace.download_file(input_file))
            LOG.info("Scoring text in page '%s' at the %s level", pcgts.get_pcGtsId(), level)
            
            # annotate processing metadata:
            metadata = pcgts.get_Metadata() # ensured by from_file()
            metadata.add_MetadataItem(
                MetadataItemType(type_="processingStep",
                                 name=OCRD_TOOL['tools']['ocrd-cor-asv-fst-process']['steps'][0],
                                 value='ocrd-cor-asv-fst-process',
                                 Labels=[LabelsType(externalRef="parameters",
                                                    Label=[LabelType(type_=name,
                                                                     value=self.parameter[name])
                                                           for name in self.parameter.keys()])]))
            
            # context preprocessing:
            # todo: as soon as we have true MODS meta-data in METS (dmdSec/mdWrap/xmlData/mods),
            #       get global context variables from there (e.g. originInfo/dateIssued/@text for year)
            ident = self.workspace.mets.unique_identifier # at least try to get purl
            context = [0]
            if ident:
                name = ident.split('/')[-1]
                year = name.split('_')[-1]
                if year.isnumeric():
                    year = ceil(int(year)/10)
                    context = [year]
                    # todo: author etc
        
        # for each line, make a graph of window results (keeping WordType element references)
        # beam-decode the line graph, starting with previous line's traceback;
        # at the end of it, go back in the best hypothesis to the end of the previous line
        # (by stopping at nodes from the previous line's traceback), 
        # identify this Node instance as the decisively best for the previous line, 
        # and filter all current hypotheses such that they must contain it in their path,
        # and cut them so that they do not reach beyond it;
        # get the path leading up to it, and
        # get the remaining beam as traceback for the next line;
        # with these results for the line, modify the previous TextLineType to include
        # only the WordType/TextEquivType elements and scores of the best path,
        # then repeat with the new traceback and next line
        lines = _page_get_line_acceptors(level, pcgts)
        
        prev_traceback = None # todo: pass from previous page
        prev_line = None # todo: pass from previous page
        for line, tokens in lines:
            # split line into windows of 1 to MAX_WINDOW_SIZE words:
            # todo: read graph directly from OCR's CTC decoder, 'spliting' sub-graphs at whitespace candidates
            windows = {}
            for i in range(len(tokens)):
                for j in range(1, min(MAX_WINDOW_SIZE + 1, len(tokens) - i + 1)):
                    windows[(i, j)] = (_merge_elements_func(tokens[i:i+j]), # a WordType reference
                                       reduce(_concat_fst_func, tokens[i:i+j], hfst.empty_fst())) # an FST
            
            # process windows in parallel:
            with mp.Pool() as pool:
                # interface to lib.Corrector.process_window:
                #   takes a 2-tuple of a 2-tuple (i,j) and a 2-tuple (reference, input FSA),
                #   reproduces the input structure, only replacing input with output FSA
                #   (this is necessary to get multiprocessing encapsulate the data)
                result = pool.starmap_async(self.corrector.process_window, windows.items(), error_callback=LOG.error) # ???
                result.wait()
                if result.successful():
                    windows = result.get()
                else:
                    LOG.critical('No window results')
                    windows = []
                    # exit or raise?

            # combine windows into line graph (with WordType references):
            graph = nx.DiGraph()
            for i in range(len(tokens) + 1):
                graph.add_node(i)
            line_start_node = 0
            line_end_node = len(tokens)
            for (i, j), (ref, fst) in windows:
                start_node = i
                end_node = i + j
                # get best paths instead of individual character transitions, 
                # so alternatives can be rated in parallel below
                # (this assumes that result is acyclic, and n_best() has been run already):
                paths = fst.extract_shortest_paths().values() # ignore input paths from now on
                paths = itertools.chain(*paths) # flatten
                graph.add_edge(start_node, end_node, element=ref, 
                               alternatives=map(lambda path:
                                                TextEquivType(Unicode=path[0], 
                                                              conf=pow(2, -path[1])), paths))
            
            # find best path for previous line, advance traceback/beam for current line
            path, entropy, traceback = self.rater.rate_best(graph, line_start_node, line_end_node,
                                                            start_traceback=prev_traceback,
                                                            context=context,
                                                            lm_weight=lm_weight,
                                                            max_length=MAX_LENGTH,
                                                            beam_width=beam_width,
                                                            beam_clustering_dist=BEAM_CLUSTERING_DIST if BEAM_CLUSTERING_ENABLE else 0)

            # apply best path to line in PAGE
            if prev_line:
                _line_update_from_path(prev_line, path, entropy)
            prev_line = line
            prev_traceback = traceback
        
        # apply best path to last line in PAGE:
        # todo: only to last line in document (when passing traceback between pages)
        if prev_line:
            path, entropy, _ = self.rater.next_path(prev_traceback, [])
            _line_update_from_path(prev_line, path, entropy)

        # ensure parent textequivs are up to date:
        page_update_higher_textequiv_levels('word', pcgts)
            
        # write back result
        file_id = concat_padded(self.output_file_grp, n)
        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            basename=file_id + '.xml', # with suffix or bare?
            mimetype=MIMETYPE_PAGE,
            content=to_xml(pcgts),
        )
        
def _page_get_line_acceptors(level, pcgts):
    results = []
    regions = pcgts.get_Page().get_TextRegion()
    if not regions:
        LOG.warning("Page contains no text regions")
    first_region = True
    for region in regions:
        lines = region.get_TextLine()
        if not lines:
            LOG.warning("Region '%s' contains no text lines", region.id)
        first_line = True
        for line in lines:
            tokens = []
            words = line.get_Word()
            if not words:
                LOG.warning("Line '%s' contains no word", line.id)
            first_word = True
            for word in words:
                token = '' # todo: FSA (for glyph alternatives and conf/weights)
                if not first_word or not first_line or not first_region:
                    token += '\n' if first_word else ' '
                if level == 'word':
                    LOG.debug("Getting text in word '%s'", word.id)
                    textequivs = word.get_TextEquiv()
                    if textequivs:
                        token += textequivs[0].Unicode
                    else:
                        LOG.warning("Word '%s' contains no text results", word.id)
                else:
                    glyphs = word.get_Glyph()
                    if not glyphs:
                        LOG.warning("Word '%s' contains no glyphs", word.id)
                    for glyph in glyphs:
                        LOG.debug("Getting text in glyph '%s'", glyph.id)
                        textequivs = glyph.get_TextEquiv()
                        if textequivs:
                            token += textequivs[0].Unicode
                        else:
                            LOG.warning("Glyph '%s' contains no text results", glyph.id)
                tokens.append((word, hfst.fst(token)))
                first_word = False
            results.append((line, tokens))
            first_line = False
        first_region = False
    return results

def _line_update_from_path(line, path, entropy):
    line.set_Word(None)
    strlen = 0
    for word, textequiv, score in path:
        if word: # not just space
            word.set_TextEquiv([textequiv]) # delete others
            strlen += len(textequiv.Unicode)
            textequiv.set_conf(score)
            prev_line.add_Word(word)
        else:
            strlen += 1
    ent = entropy/strlen
    avg = pow(2.0, -ent)
    ppl = pow(2.0, ent) # character level
    ppll = pow(2.0, ent * strlen/len(path)) # textequiv level (including spaces/newlines)
    LOG.info("line '%s' avg: %.3f, char ppl: %.3f, word ppl: %.3f", line.id, avg, ppl, ppll)

def _concat_fst_func(fst, tok):
    fst.concatenate(tok[1])
    return fst

def _merge_elements_func(toks):
    elements = map(lambda tok: tok[0], toks)
    merged = WordType()
    merged.set_id(','.join(map(lambda elem: elem.id, elements)))
    points = ' '.join(map(lambda elem: elem.get_Coords().points, elements))
    merged.set_Coords(points=points_from_xywh(xywh_from_points(points)))
    # make other attributes and TextStyle a majority vote, 
    # but no Glyph (too fine-grained) or TextEquiv (overwritten from best path anyway)
    languages = map(lambda elem: elem.get_language(), elements)
    merged.set_language(max(set(languages), key=languages.count))
    # todo: other attributes...
    styles = map(lambda elem: elem.get_TextStyle(), elements)
    if any(styles):
        # todo: make a majority vote on each attribute here
        merged.set_TextStyle(elements[0].get_TextStyle())

