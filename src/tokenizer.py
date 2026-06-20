from collections import defaultdict
import sys

from pathlib import Path
import regex as re
from loguru import logger
from joblib import Parallel, delayed

logger.remove()
logger.add(sys.stderr, level="INFO")


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
        self.can_merge = True
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

        if not sorted_byte_pairs_with_frequency:
            self.can_merge = False
            return

        logger.debug(f"{ self.vocab_size() = }")

        most_frequent_byte_pair = sorted_byte_pairs_with_frequency[0][0]
        token_id = len(self.vocab)
        self.vocab[token_id] = (
            self.vocab[most_frequent_byte_pair[0]]
            + self.vocab[most_frequent_byte_pair[1]]
        )
        self.vocab_reverse[self.vocab[token_id]] = token_id
        self.merges.append(most_frequent_byte_pair)

        def process_chunk(chunk: tuple[int]) -> tuple[int]:
            replaced_chunk, i = [], 0

            while i < len(chunk) - 1:
                if chunk[i] == most_frequent_byte_pair[0] and chunk[i+1] == most_frequent_byte_pair[1]:
                    replaced_chunk.append(token_id)
                    i += 2
                else:
                    replaced_chunk.append(chunk[i])
                    i += 1

            return tuple(replaced_chunk)

        self.chunks = Parallel(n_jobs=-1)(delayed(process_chunk)(c) for c in self.chunks)

    def train(self):
        """Tokenize a corpus with BPE."""

        while self.vocab_size() < self.max_vocab_size and self.can_merge:
            self._merge_most_frequent_byte_pair(self._compute_byte_pair_frequency())
            logger.debug(
                f"n_vocab % = { self.vocab_size() / self.max_vocab_size :.4f} | { self.merges = }"
            )

        return self.vocab, self.merges


if __name__ == "__main__":
    t = Tokenizer(
        Path("tests/fixtures/corpus.en"),
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    t.train()
    # logger.info(f"{ t.vocab = }")
    # logger.info(f"{ t.vocab_reverse = }")
