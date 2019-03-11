# cor-asv-fst
    OCR post-correction with error/lexicon FSTs and char-LM LSTMs

## Introduction


## Installation

Required Ubuntu packages:

* Python (``python`` or ``python3``)
* pip (``python-pip`` or ``python3-pip``)
* virtualenv (``python-virtualenv`` or ``python3-virtualenv``)

Create and activate a virtualenv as usual.

To install Python dependencies and this module, then do:
```shell
make deps install
```
Which is the equivalent of:
```shell
pip install -r requirements.txt
pip install -e .
```

## Usage

This packages has two user interfaces:

### command line interface `cor-asv-fst-train`

To be used with string arguments and plain-text files.

...

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-cor-asv-fst-process`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow. Input could be anything with a textual annotation (`TextEquiv` on the given `textequiv_level`). 

...

```json
  "tools": {
    "cor-asv-fst-process": {
      "executable": "cor-asv-fst-process",
      "categories": [
        "Text recognition and optimization"
      ],
      "steps": [
        "recognition/post-correction"
      ],
      "description": "Improve text annotation by FST error and lexicon model with character-level LSTM language model",
      "input_file_grp": [
        "OCR-D-OCR-TESS",
        "OCR-D-OCR-KRAK",
        "OCR-D-OCR-OCRO",
        "OCR-D-OCR-CALA",
        "OCR-D-OCR-ANY"
      ],
      "output_file_grp": [
        "OCR-D-COR-ASV"
      ],
      "parameters": {
        "keraslm_file": {
          "type": "string",
          "format": "uri",
          "content-type": "application/x-hdf;subtype=bag",
          "description": "path of h5py weight/config file for language model trained with keraslm",
          "required": true,
          "cacheable": true
        },
        "errorfst_file": {
          "type": "string",
          "format": "uri",
          "content-type": "application/vnd.openfst",
          "description": "path of FST file for error model",
          "required": true,
          "cacheable": true
        },
        "lexiconfst_file": {
          "type": "string",
          "format": "uri",
          "content-type": "application/vnd.openfst",
          "description": "path of FST file for lexicon model",
          "required": true,
          "cacheable": true
        },
        "textequiv_level": {
          "type": "string",
          "enum": ["word", "glyph"],
          "default": "glyph",
          "description": "PAGE XML hierarchy level to read TextEquiv input on (output will always be word level)"
        },
        "beam_width": {
          "type": "number",
          "format": "integer",
          "description": "maximum number of best partial paths to consider during beam search in language modelling",
          "default": 100
        },
        "lm_weight": {
          "type": "number",
          "format": "float",
          "description": "share of the LM scores over the FST output confidences",
          "default": 0.5
        }
      }
    }
  }
```

...

## Testing

...
