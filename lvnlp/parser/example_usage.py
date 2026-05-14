import logging

from lvnlp.parser import Parser
from lvnlp.parser.utils import to_conll, tokenize_sentence


def example_usage(device='cpu', analyzer=False):
    parser = Parser.from_pretrained(device=device)
    sentences = [tokenize_sentence('Jānis brauca uz Rīgu.')]
    sentences = parser.parse(sentences, batch_size=1, analyzer=analyzer)
    for sentence in sentences:
        for token in sentence:
            print(token)
        print()

    print(to_conll(sentences))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    example_usage(analyzer=True)
