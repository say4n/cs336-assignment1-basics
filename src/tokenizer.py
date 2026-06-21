import time
from collections import defaultdict
import sys

from pathlib import Path
import regex as re
from loguru import logger

from tests.common import gpt2_bytes_to_unicode

logger.remove()
logger.add(sys.stderr, level="INFO")


class Tokenizer:
    def __init__(self, corpus: Path | str, vocab_size: int, special_tokens: list[str]):
        self.max_vocab_size = vocab_size
        self.special_tokens: set[str] = set(special_tokens)
        self.corpus: str = None
        self.merges: list[tuple[bytes, bytes]] = list()
        self.vocab: dict[int, bytes] = dict()
        self.vocab_reverse: dict[bytes, int] = dict()
        self.chunks: dict[tuple[int, ...], int] = defaultdict(int)

        self.__init_load_corpus(corpus)

        self.merges = []
        self.can_merge = True
        self.vocab = dict((i, bytes([i])) for i in range(256))
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = bytes(token, "utf-8")

        self.vocab_reverse = dict((v, k) for k, v in self.vocab.items())

        self.vocab_size = lambda: len(self.vocab)

        self.__init_chunk_with_regex()

    def __init_load_corpus(self, corpus):
        if isinstance(corpus, Path):
            self.corpus = open(corpus, encoding="utf-8").read()
        elif isinstance(corpus, str):
            self.corpus = corpus
        else:
            raise ValueError("`corpus` must either be a path or string")

    def __init_chunk_with_regex(self):
        word_boundary_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        pretoken_counts = defaultdict(int)

        if self.special_tokens:
            special_token_pattern = "|".join(
                re.escape(tok)
                for tok in sorted(self.special_tokens, key=len, reverse=True)
            )
            parts = re.split(f"({special_token_pattern})", self.corpus)
        else:
            parts = [self.corpus]

        for part in parts:
            if not part or part in self.special_tokens:
                continue

            for match in word_boundary_pattern.finditer(part):
                pretoken_counts[match.group()] += 1

        for pretoken, count in pretoken_counts.items():
            self.chunks[tuple(pretoken.encode("utf-8"))] += count

    def _compute_byte_pair_frequency(self) -> dict[tuple[int, int], int]:
        byte_pairs_with_frequency = defaultdict(int)

        for bb, count in self.chunks.items():
            for i in range(len(bb) - 1):
                byte_pairs_with_frequency[(bb[i], bb[i + 1])] += count

        return byte_pairs_with_frequency

    def _merge_most_frequent_byte_pair(
        self, byte_pairs_with_frequency: dict[tuple[int, int], int]
    ):
        if not byte_pairs_with_frequency:
            self.can_merge = False
            return

        most_frequent_byte_pair_with_frequency = max(
            byte_pairs_with_frequency.items(),
            key=lambda el: (el[1], tuple(map(lambda tid: self.vocab[tid], el[0]))),
        )

        logger.debug(f"{ self.vocab_size() = }")

        most_frequent_byte_pair = most_frequent_byte_pair_with_frequency[0]
        token_id = len(self.vocab)
        self.vocab[token_id] = (
            self.vocab[most_frequent_byte_pair[0]]
            + self.vocab[most_frequent_byte_pair[1]]
        )
        self.vocab_reverse[self.vocab[token_id]] = token_id
        self.merges.append((
            self.vocab[most_frequent_byte_pair[0]],
            self.vocab[most_frequent_byte_pair[1]],
        ))

        def process_chunk(chunk: tuple[int], frequency: int) -> tuple[tuple[int, ...], int]:
            replaced_chunk, i = [], 0

            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == most_frequent_byte_pair[0] and chunk[i+1] == most_frequent_byte_pair[1]:
                    replaced_chunk.append(token_id)
                    i += 2
                else:
                    replaced_chunk.append(chunk[i])
                    i += 1

            return tuple(replaced_chunk), frequency

        new_chunks = defaultdict(int)
        for k, v in self.chunks.items():
            ret_k, ret_v = process_chunk(k, v)
            new_chunks[ret_k] += ret_v

        self.chunks = new_chunks

    def train(self) -> tuple[dict[int, bytes] | None, list[tuple[bytes, bytes]] | None]:
        """Tokenize a corpus with BPE."""

        while self.vocab_size() < self.max_vocab_size and self.can_merge:
            self._merge_most_frequent_byte_pair(self._compute_byte_pair_frequency())
            logger.debug(
                f"n_vocab % = { self.vocab_size() / self.max_vocab_size :.4f} | { self.merges = }"
            )

        return self.vocab, self.merges

    def write_merges_to_file(self):
        byte_encoder = gpt2_bytes_to_unicode()

        def render(piece: bytes) -> str:
            return "".join(byte_encoder[b] for b in piece)

        with open("merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(render(a) + " " + render(b) + "\n")


if __name__ == "__main__":
    t = Tokenizer(
        Path("data/TinyStoriesV2-GPT4-train.txt"),
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    start_time = time.time()
    vocab, merges = t.train()
    delta_time = time.time() - start_time
    logger.info(f"trained in {delta_time} sec.")

    # t.write_merges_to_file()
