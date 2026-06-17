from collections import defaultdict
import sys

from pathlib import Path
import regex as re
from loguru import logger
from joblib import Parallel, delayed

logger.remove()
logger.add(sys.stderr, level="DEBUG")


class Tokenizer:
    def __init__(self, corpus, vocab_size, special_tokens):
        self.max_vocab_size = vocab_size
        self.special_tokens: list[str] = special_tokens
        self.corpus: str | None = None
        self.merges: list[tuple[int]] | None = None
        self.vocab: dict[int, str] | None = None
        self.vocab_reverse: dict[str, int] | None = None
        self.chunks: list[tuple[int]] | None = None

        self.__init_load_corpus(corpus)

        self.merges = []
        self.vocab = dict((i, chr(i)) for i in range(256))
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token

        self.vocab_reverse = dict((v, k) for k, v in self.vocab.items())

        self.vocab_size = lambda: len(self.vocab)

        self.chunks = self.__init_chunk_with_regex()

    def __init_load_corpus(self, corpus):
        if Path.exists(corpus):
            self.corpus = open(corpus).read()
        elif isinstance(corpus, str):
            self.corpus = corpus
        else:
            raise ValueError("`corpus` must either be a path or string")

    def __init_chunk_with_regex(self):
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokenized = re.finditer(pattern, self.corpus)
        pre_tokenized_bytes = [
            tuple(map(lambda ch: ord(ch), match.group())) for match in pre_tokenized
        ]

        return pre_tokenized_bytes

    def _compute_byte_pair_frequency(self):
        byte_pairs_with_frequency = defaultdict(int)

        for bytes in self.chunks:
            for i in range(len(bytes) - 1):
                byte_pairs_with_frequency[(bytes[i], bytes[i + 1])] += 1

        return byte_pairs_with_frequency

    def _merge_most_frequent_byte_pair(self, byte_pairs_with_frequency):
        sorted_byte_pairs_with_frequency = sorted(
            byte_pairs_with_frequency.items(),
            key=lambda el: (el[1], el[0]),
            reverse=True,
        )

        most_frequent_byte_pair = sorted_byte_pairs_with_frequency[0][0]
        token_id = len(self.vocab)
        self.vocab[token_id] = (
            self.vocab[most_frequent_byte_pair[0]]
            + self.vocab[most_frequent_byte_pair[1]]
        )
        self.vocab_reverse[self.vocab[token_id]] = token_id
        self.merges.append(most_frequent_byte_pair)

        aux_joined_byte_pair = "-".join(map(str, most_frequent_byte_pair))

        def process_chunk(chunk):
            return tuple(
                map(
                    int,
                    (
                        "-".join(map(str, chunk)).replace(
                            aux_joined_byte_pair,
                            str(token_id),
                        )
                    ).split("-"),
                )
            )

        self.chunks = Parallel(n_jobs=1)(delayed(process_chunk)(c) for c in self.chunks)

    def train(self):
        """Tokenize a corpus with BPE."""

        while self.vocab_size() < self.max_vocab_size:
            self._merge_most_frequent_byte_pair(self._compute_byte_pair_frequency())
            logger.debug(
                f"{ self.merges = }, { self.vocab_size() / self.max_vocab_size :.4f}"
            )

        return self.vocab, self.merges


if __name__ == "__main__":
    t = Tokenizer(
        Path("data/TinyStoriesV2-GPT4-valid.txt"),
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    t.train()
    logger.info(f"{ t.vocab = }")
    logger.info(f"{ t.vocab_reverse = }")
