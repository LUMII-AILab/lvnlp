# lvnlp

Latvian NLP tools for morphosyntactic parsing.

## Installation

Install from PyPI:

```bash
pip install lvnlp
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/LUMII-AILab/lvnlp.git
```

## Morphosyntactic parsing

```python
from lvnlp.parser import Parser
from lvnlp.parser.utils import to_conll, tokenize_sentence

parser = Parser.from_pretrained()
sentences = [tokenize_sentence('Jānis brauca uz Rīgu.')]
sentences = parser.parse(sentences, analyzer=False)
for sentence in sentences:
    for token in sentence:
        print(token)
    print()

print(to_conll(sentences))
```

The parser can optionally use the Latvian morphological analyzer during decoding. This improves XPOS tagging and lemmatization.

## Citation

```bibtex
@inproceedings{znotins-2026-improving,
  title = {Improving Latvian Morphosyntactic Parsing with Pretrained Encoders and Analyzer-Constrained Decoding},
  author = {Znotins, Arturs},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {11724--11734},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/5khpzsaiqrzw},
  url = {https://lrec.elra.info/lrec2026-main-918}
}
```

## Acknowledgements

This work was supported by the EU Recovery and Resilience Facility project
[Language Technology Initiative](https://www.vti.lu.lv)
(2.3.1.1.i.0/1/22/I/CFLA/002).
