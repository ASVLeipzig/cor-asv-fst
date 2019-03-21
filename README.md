# cor-asv-fst
    OCR post-correction with error/lexicon Finite State Transducers and
    chararacter-level LSTM language models

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

The package has two user interfaces:

### Command Line Interface

The package contains a suite of CLI tools to work with plaintext data (prefix:
`cor-asv-fst-*`). The minimal working examples and data formats are described
below. Additionally, each tool has further optional parameters - for a detailed
description, call the tool with the `--help` option.

#### `cor-asv-fst-train`

Train FST models. The basic invocation is as follows:

```shell
cor-asv-fst-train -l LEXICON_FILE -e ERROR_MODEL_FILE -t TRAINING_FILE
```

This will create two transducers, which will be stored in `LEXICON_FILE` and
`ERROR_MODEL_FILE`, respectively. As the training of the lexicon and the error
model is done independently, any of them can be skipped by omitting the
respective parameter.

`TRAINING_FILE` is a plain text file in tab-separated, two-column format
containing a line of OCR-output and the corresponding ground truth line:

```
» Bergebt mir, daß ih niht weiß, wie	»Vergebt mir, daß ich nicht weiß, wie
aus dem (Geiſte aller Nationen Mahrunq	aus dem Geiſte aller Nationen Nahrung
Kannſt Du mir die re<hée Bahn niché zeigen ?	Kannſt Du mir die rechte Bahn nicht zeigen?
frag zu bringen. —	trag zu bringen. —
ſie ins irdij<he Leben hinein, Mit leichtem,	ſie ins irdiſche Leben hinein. Mit leichtem,
```

Each line is treated independently. Alternatively to the above, the training
data may also be supplied as two files:

```shell
cor-asv-fst-train -l LEXICON_FILE -e ERROR_MODEL_FILE -i INPUT_FILE -g GT_FILE
```

In this variant, `INPUT_FILE` and `GT_FILE` are both in tab-separated,
two-column format, in which the first column is the line ID and the second the
line:

```
>=== INPUT_FILE ===<
alexis_ruhe01_1852_0018_022     ih denke. Aber was die ſelige Frau Geheimräth1n
alexis_ruhe01_1852_0035_019     „Das fann ich niht, c’esl absolument impos-
alexis_ruhe01_1852_0087_027     rend. In dem Augenbli> war 1hr niht wohl zu
alexis_ruhe01_1852_0099_012     ür die fle ſich ſchlugen.“
alexis_ruhe01_1852_0147_009     ſollte. Nur Über die Familien, wo man ſie einführen

>=== GT_FILE ===<
alexis_ruhe01_1852_0018_022     ich denke. Aber was die ſelige Frau Geheimräthin
alexis_ruhe01_1852_0035_019     „Das kann ich nicht, c'est absolument impos—
alexis_ruhe01_1852_0087_027     rend. Jn dem Augenblick war ihr nicht wohl zu
alexis_ruhe01_1852_0099_012     für die ſie ſich ſchlugen.“
alexis_ruhe01_1852_0147_009     ſollte. Nur über die Familien, wo man ſie einführen
```

#### `cor-asv-fst-process`

This tool applies a trained model to correct plaintext data on a line basis.
The basic invocation is:

```shell
cor-asv-fst-process -i INPUT_FILE -o OUTPUT_FILE -l LEXICON_FILE -e ERROR_MODEL_FILE
```

`INPUT_FILE` is in the same format as for the training procedure. `OUTPUT_FILE`
contains the post-correction results in the same format.

#### `cor-asv-fst-evaluate`

This tool can be used to evaluate the post-correction results. The minimal
working invocation is:

```shell
cor-asv-fst-evaluate -i INPUT_FILE -o OUTPUT_FILE -g GT_FILE
```

Additionally, the parameter `-M` can be used to select the evaluation measure
(`Levenshtein` by default). The files should be in the same two-column format
as described above.

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-cor-asv-fst-process`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries)
documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow.
Input could be anything with a textual annotation (`TextEquiv` on the given
`textequiv_level`).

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
