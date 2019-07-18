from math import ceil
import networkx as nx
import os.path

from ocrd import Processor
from ocrd_utils import concat_padded, getLogger, MIMETYPE_PAGE
from ocrd_models.ocrd_page import \
    MetadataItemType, LabelType, LabelsType, TextEquivType, WordType, parse, \
    to_xml
from ocrd_keraslm.lib import Rater

from .config import OCRD_TOOL
from ..lib.latticegen import create_window, process_window
from ..scripts.process import prepare_model


LOG = getLogger('processor.FSTCorrection')

# enable pruning partial paths by history clustering
BEAM_CLUSTERING_ENABLE = True

# maximum distance between state vectors to form a cluster
BEAM_CLUSTERING_DIST = 5


class FSTCorrection(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-cor-asv-fst-process']
        kwargs['version'] = OCRD_TOOL['version']
        super(FSTCorrection, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace:
            raise RuntimeError('no workspace specified!')

        # initialize the decoder
        LOG.info("Loading the correction models")
        self.model = prepare_model(
            self.parameter['lexicon_file'],
            self.parameter['error_model_file'],
            words_per_window = self.parameter['words_per_window'])

        # initialize the language model
        self.rater = Rater(logger=LOG)
        self.rater.load_config(self.parameter['keraslm_file'])
        # overrides for incremental mode necessary before compilation:
        self.rater.stateful = False         # no implicit state transfer
        self.rater.incremental = True       # but explicit state transfer
        self.rater.configure()
        self.rater.load_weights(self.parameter['keraslm_file'])

    def process(self):
        for (n, input_file) in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file)
            local_input_file = self.workspace.download_file(input_file)
            pcgts = parse(local_input_file.url, silence=True)
            LOG.info("Scoring text in page '%s' at the %s level",
                     pcgts.get_pcGtsId(), self.parameter['textequiv_level'])
            self._process_page(pcgts)

            # write back result
            file_id = concat_padded(self.output_file_grp, n)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                local_filename=os.path.join(self.output_file_grp, file_id),
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts),
            )

    def _process_page(self, pcgts):
        self._add_my_metadata_to_page(pcgts)
        prev_traceback = None   # TODO: pass from previous page
        prev_line = None        # TODO: pass from previous page
        for n_line in _page_get_lines(pcgts):
            # decoding: line -> windows -> lattice
            windows = self._line_to_windows(n_line)
            self._process_windows(windows)
            graph = self._combine_windows_to_line_graph(windows)
            
            # find best path for previous line, advance traceback/beam for
            # current line
            line_start_node = 0
            line_end_node = max(i+j for i, j in windows)
            context = self._get_context_from_identifier(\
                self.workspace.mets.unique_identifier)
            path, entropy, traceback = self.rater.rate_best(
                graph, line_start_node, line_end_node,
                start_traceback = prev_traceback,
                context = context,
                lm_weight = self.parameter['lm_weight'],
                beam_width = self.parameter['lm_beam_width'],
                beam_clustering_dist = \
                    BEAM_CLUSTERING_DIST if BEAM_CLUSTERING_ENABLE else 0)

            # apply best path to line in PAGE
            if prev_line:
                _line_update_from_path(prev_line, path, entropy)
            prev_line = n_line
            prev_traceback = traceback
        
        # apply best path to last line in PAGE
        # TODO only to last line in document (when passing traceback between
        #      pages)
        if prev_line:
            path, entropy, _ = self.rater.next_path(
                prev_traceback[0],
                ([], prev_traceback[1]))
            _line_update_from_path(prev_line, path, entropy)

        # ensure parent textequivs are up to date:
        page_update_higher_textequiv_levels('word', pcgts)

    def _add_my_metadata_to_page(self, pcgts):
        metadata = pcgts.get_Metadata()
        metadata.add_MetadataItem(
            MetadataItemType(
                type_='processingStep',
                name=OCRD_TOOL['tools']['ocrd-cor-asv-fst-process']['steps'][0],
                value='ocrd-cor-asv-fst-process',
                Labels=[
                    LabelsType(externalRef='parameters',
                               Label=[LabelType(type_=name,
                                                value=self.parameter[name])
                                      for name in self.parameter.keys()])]))

    def _line_to_tokens(self, n_line):
        result = []
        n_words = n_line.get_Word()
        if not n_words:
            LOG.warning("Line '%s' contains no word", n_line.id)
        for n_word in n_words:
            n_textequivs = n_word.get_TextEquiv()
            if n_textequivs and n_textequivs[0].Unicode:
                result.append(n_textequivs[0].Unicode)
            else:
                LOG.warning("Word '%s' contains no text results", n_word.id)
        return result

    def _line_to_windows(self, n_line):
        # TODO currently: gets the text from the Word elements and returns
        # lines as lists of words;
        # needed: also get glyph alternatives
        n_words = n_line.get_Word()
        tokens = self._line_to_tokens(n_line)
        return { (i, j) : (self._merge_word_nodes(n_words[i:i+j]),
                           create_window(tokens[i:i+j]),
                           tokens[i:i+j]) \
                 for i in range(len(tokens)) \
                 for j in range(1, min(self.parameter['max_window_size']+1,
                                       len(tokens)-i+1)) }

    def _merge_word_nodes(self, nodes):
        if not nodes:
            LOG.error('nothing to merge')
            return None
        merged = WordType()
        merged.set_id(','.join([n.id for n in nodes]))
        points = ' '.join([n.get_Coords().points for n in nodes])
        if points:
            merged.set_Coords(nodes[0].get_Coords())    # TODO merge
        # make other attributes and TextStyle a majority vote, but no Glyph
        # (too fine-grained) or TextEquiv (overwritten from best path anyway)
        languages = list(map(lambda elem: elem.get_language(), nodes))
        if languages:
            merged.set_language(max(set(languages), key=languages.count))
        # TODO other attributes...
        styles = map(lambda elem: elem.get_TextStyle(), nodes)
        if any(styles):
            # TODO make a majority vote on each attribute here
            merged.set_TextStyle(nodes[0].get_TextStyle())
        return merged

    def _process_windows(self, windows):
        for (i, j), (ref, fst, tokens) in windows.items():
            LOG.debug('Processing window ({}, {})'.format(i, j))
            fst = process_window(
                ' '.join(tokens), fst, self.model,
                beam_width=self.parameter['beam_width'],
                rejection_weight=self.parameter['rejection_weight'])
            windows[(i, j)] = (ref, fst, tokens)

    def _combine_windows_to_line_graph(self, windows):
        graph = nx.DiGraph()
        line_end_node = max(i+j for i, j in windows)
        graph.add_nodes_from(range(line_end_node + 1))
        for (i, j), (ref, fst, tokens) in windows.items():
            start_node = i
            end_node = i + j
            paths = [(output_str, float(weight)) \
                     for input_str, output_str, weight in \
                         fst.paths().items()]
            if paths:
                for path in paths:
                    LOG.info('({}, {}, \'{}\', {})'.format(\
                        start_node, end_node, path[0], pow(2, -path[1])))
                graph.add_edge(
                    start_node, end_node, element=ref,
                    alternatives=list(map(
                        lambda path:
                            TextEquivType(Unicode=path[0],
                                          conf=pow(2, -path[1])),
                        paths)))
            else:
                LOG.warning('No path from {} to {}.'.format(i, i+j))
        return graph

    def _get_context_from_identifier(self, identifier):
        context = [0]
        if identifier:
            name = identifier.split('/')[-1]
            year = name.split('_')[-1]
            if year.isnumeric():
                year = ceil(int(year)/10)
                context = [year]
        return context


def _page_get_lines(pcgts):
    lines = []
    n_regions = pcgts.get_Page().get_TextRegion()
    if not n_regions:
        LOG.warning('Page contains no text regions')
    for n_region in n_regions:
        n_lines = n_region.get_TextLine()
        if not n_lines:
            LOG.warning("Region '%s' contains no text lines", n_region.id)
        lines.extend(n_lines)
    return lines


def page_update_higher_textequiv_levels(level, pcgts):
    '''
    Update the TextEquivs of all PAGE-XML hierarchy levels above `level`
    for consistency.
    
    Starting with the hierarchy level chosen for processing, join all
    first TextEquiv (by the rules governing the respective level) into
    TextEquiv of the next higher level, replacing them.
    '''
    regions = pcgts.get_Page().get_TextRegion()
    if level != 'region':
        for region in regions:
            lines = region.get_TextLine()
            if level != 'line':
                for line in lines:
                    words = line.get_Word()
                    if level != 'word':
                        for word in words:
                            glyphs = word.get_Glyph()
                            word_unicode = u''.join(
                                glyph.get_TextEquiv()[0].Unicode \
                                    if glyph.get_TextEquiv() \
                                    else u'' \
                                for glyph in glyphs)
                            word.set_TextEquiv(
                                [TextEquivType(Unicode=word_unicode)])
                    line_unicode = u' '.join(
                        word.get_TextEquiv()[0].Unicode \
                            if word.get_TextEquiv() \
                            else u'' \
                        for word in words)
                    line.set_TextEquiv([TextEquivType(Unicode=line_unicode)])
            region_unicode = u'\n'.join(\
                line.get_TextEquiv()[0].Unicode \
                    if line.get_TextEquiv() \
                    else u'' \
                for line in lines)
            region.set_TextEquiv([TextEquivType(Unicode=region_unicode)])


def _line_update_from_path(line, path, entropy):
    line.set_Word([])
    strlen = 0
    for word, textequiv, score in path:
        if word:
            word.set_TextEquiv([textequiv])
            strlen += len(textequiv.Unicode)
            textequiv.set_conf(score)
            line.add_Word(word)
        else:
            strlen += 1
    ent = entropy/strlen
    try:
        avg = pow(2.0, -ent)
        # character level
        ppl = pow(2.0, ent)
        # textequiv level (including spaces/newlines)
        ppll = pow(2.0, ent * strlen/len(path))
        LOG.info("line '%s' avg: %.3f, char ppl: %.3f, word ppl: %.3f",
                 line.id, avg, ppl, ppll)
    except OverflowError:
        LOG.warning("line '%s' overflow while computing perplexity")

