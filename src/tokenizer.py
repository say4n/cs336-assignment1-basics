from collections.abc import Iterable, Iterator
from tqdm import tqdm
import time
from collections import Counter
import sys

from pathlib import Path
import regex as re
from loguru import logger
from joblib import delayed, Parallel

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
        self.chunks: dict[tuple[int, ...], int] = Counter()
        self.byte_pairs_with_frequency: dict[tuple[int, int], int] = Counter()

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

        pretoken_counts = Counter()

        if self.special_tokens:
            special_token_pattern = "|".join(
                re.escape(tok)
                for tok in sorted(self.special_tokens, key=len, reverse=True)
            )
            raw_parts = filter(
                lambda part: part not in self.special_tokens,
                re.split(f"({special_token_pattern})", self.corpus),
            )
        else:
            raw_parts = [self.corpus]

        # Accumulate parts into chunks after removing special tokens.
        parts_of_parts = []
        accumulated = 0
        current = []

        for part in raw_parts:
            current.append(part)
            accumulated += len(part)

            if accumulated >= 100_000:
                parts_of_parts.append(current)
                current = []
                accumulated = 0

        if current:
            parts_of_parts.append(current)

        logger.debug(f"processing { len(parts_of_parts) = }")

        def process_part(part_of_parts: list[str]) -> Counter[str, int]:
            partial_pretoken_counts = Counter()

            for part in part_of_parts:
                for match in word_boundary_pattern.finditer(part):
                    partial_pretoken_counts[match.group()] += 1

            return Counter(partial_pretoken_counts)

        pretoken_counts = Counter()

        for partial_pretoken_count in tqdm(
            Parallel(n_jobs=-1, return_as="generator")(
                delayed(process_part)(part_of_parts) for part_of_parts in parts_of_parts
            ),
            total=len(parts_of_parts),
            desc="Pre-tokenizing",
        ):
            pretoken_counts.update(partial_pretoken_count)

        logger.debug(f"processed { len(parts_of_parts) = }")

        for pretoken, count in pretoken_counts.items():
            self.chunks[tuple(pretoken.encode("utf-8"))] += count

    def _seed_byte_pair_frequency(self) -> dict[tuple[int, int], int]:
        byte_pairs_with_frequency = Counter()

        for bb, count in self.chunks.items():
            for i in range(len(bb) - 1):
                byte_pairs_with_frequency[(bb[i], bb[i + 1])] += count

        self.byte_pairs_with_frequency = byte_pairs_with_frequency

    def _merge_most_frequent_byte_pair(self):
        # Empty sequence, can't merge anymore.
        if not self.byte_pairs_with_frequency:
            self.can_merge = False
            return

        # Pick max by frequency, O(n)
        # Looks like: ((tok_a, tok_b), freq)
        most_frequent_byte_pair_with_frequency = max(
            self.byte_pairs_with_frequency.items(),
            key=lambda el: (el[1], tuple(map(lambda tid: self.vocab[tid], el[0]))),
        )

        logger.debug(f"{ self.vocab_size() = }")

        # Add byte pair as new token to vocab.
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

        # Replace byte pairs in chunk L -> R with new token.
        def process_chunk(chunk: tuple[int], frequency: int) -> tuple[tuple[int, ...], int]:
            replaced_chunk, i = [], 0
            chunk_contains_bp = False

            while i < len(chunk):
                # Byte pair (i, i+1) being replaced.
                if i < len(chunk) - 1 and chunk[i] == most_frequent_byte_pair[0] and chunk[i+1] == most_frequent_byte_pair[1]:
                    replaced_chunk.append(token_id)
                    i += 2
                    chunk_contains_bp = True
                else:
                    replaced_chunk.append(chunk[i])
                    i += 1

            # Recompute byte pairs for chunk.
            if chunk_contains_bp:
                # Subtract counts for previous chunk.
                for i in range(len(chunk) - 1):
                    self.byte_pairs_with_frequency[
                        (chunk[i], chunk[i + 1])
                    ] -= frequency

                    # Remove byte pairs with 0 frequency.
                    if self.byte_pairs_with_frequency[
                        (chunk[i], chunk[i + 1])
                    ] == 0:
                        del self.byte_pairs_with_frequency[(chunk[i], chunk[i + 1])]

                # Add counts using replaced bytes.
                for i in range(len(replaced_chunk) - 1):
                    self.byte_pairs_with_frequency[
                        (replaced_chunk[i], replaced_chunk[i + 1])
                    ] += frequency

            return tuple(replaced_chunk), frequency

        new_chunks = Counter()
        for k, v in self.chunks.items():
            # Check if a chunks actually needs replacing.
            needs_replacing = False
            for i in range(len(k) - 1):
                if k[i] == most_frequent_byte_pair[0] and k[i+1] == most_frequent_byte_pair[1]:
                    needs_replacing = True
                    break

            if needs_replacing:
                ret_k, ret_v = process_chunk(k, v)
                new_chunks[ret_k] += ret_v
            else:
                new_chunks[k] += v

        self.chunks = new_chunks

    def train(self) -> tuple[dict[int, bytes] | None, list[tuple[bytes, bytes]] | None]:
        """Tokenize a corpus with BPE."""
        logger.debug(f"{ len(self.chunks) = }")
        logger.debug(f"{ sum(self.chunks.values()) = }")
        logger.debug(f"{ sum(map(len, self.chunks.keys())) = }")

        with tqdm(total=100, desc="Training", unit_scale=True) as pbar:
            # Seed initial byte pair frequency.
            self._seed_byte_pair_frequency()

            while self.vocab_size() < self.max_vocab_size and self.can_merge:
                self._merge_most_frequent_byte_pair()
                logger.debug(
                    f"n_vocab % = { self.vocab_size() / self.max_vocab_size :.4f} | { self.merges = }"
                )
                delta = 100 * (self.vocab_size() / self.max_vocab_size) - pbar.n
                pbar.update(delta)

        return self.vocab, self.merges

    def write_merges_to_file(self):
        byte_encoder = gpt2_bytes_to_unicode()

        def render(piece: bytes) -> str:
            return "".join(byte_encoder[b] for b in piece)

        with open("merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(render(a) + " " + render(b) + "\n")

class SerializedTokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.inverted_vocab = {v: k for (k, v) in self.vocab.items()}

        if special_tokens:
            assert list(
                bytes(t, encoding="utf-8") in self.inverted_vocab for t in special_tokens
            )

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        raise NotImplementedError("Instantiating from serialized files not supported yet.")

    def encode(self, text: str) -> list[int]:
        text_copy = [text]

        # logger.info(self.special_tokens)
        # logger.info(text_copy)

        # Process special tokens.
        if self.special_tokens:
            for special in self.special_tokens:
                new_parts = []
                for part in text_copy:
                    if special in part:
                        for subpart in part.split(special):
                            new_parts.append(subpart)
                            new_parts.append(self.inverted_vocab[bytes(special, encoding="utf-8")])
                    else:
                        new_parts.append(part)
                        new_parts.append("THIS-WILL-BE-REMOVED")

                # Omit last as we always add one extra.
                text_copy = new_parts[:-1]

        # Process text to bytes for parts that are not already processed as special tokens.
        text_bytes_list = []

        def process_byte_string(string: str) -> list[int]:
            string_bytes = [bytes([b]) for b in bytes(string, encoding="utf-8")]
            string_bytes_list = list(map(lambda ch: self.inverted_vocab[ch], string_bytes))

            return string_bytes_list

        for part in text_copy:
            if isinstance(part, int):
                text_bytes_list.append(part)
            else:
                text_bytes_list.extend(process_byte_string(part))

        logger.debug(f"encode::{text_bytes_list = }")

        # Can't merge < 2 items.
        if len(text_bytes_list) >= 2:
            logger.debug(f"encode::{text_bytes_list}")

            for a, b in self.merges:
                i, merged_text_bytes_list = 0, []
                iva, ivb = self.inverted_vocab[a], self.inverted_vocab[b]
                did_merge = False

                logger.debug(f"encode::trying to merge with {a, b} -> {iva, ivb}")

                while i < len(text_bytes_list) - 1:
                    if (text_bytes_list[i], text_bytes_list[i + 1]) == (iva, ivb):
                        merged_text_bytes_list.append(self.inverted_vocab[a + b])
                        did_merge = True
                        i += 2
                    else:
                        merged_text_bytes_list.append(text_bytes_list[i])
                        i += 1

                    # Last index, can't merge any more.
                    if i == len(text_bytes_list) - 1:
                        merged_text_bytes_list.append(text_bytes_list[i])

                if did_merge:
                    logger.debug(f"encode::{a, b} -> {iva, ivb}")
                text_bytes_list = merged_text_bytes_list[:]

        logger.debug(f"encode::{text_bytes_list = }")

        return text_bytes_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError("Encoding iterables is not supported yet.")

    def decode(self, ids: list[int]) -> str:
        logger.debug(f"decode::{ids = }")

        decoded_seq = list(map(lambda token_id: self.vocab[token_id], ids))
        logger.debug(f"decode::{decoded_seq = }")

        joined_seq = b"".join(decoded_seq)
        logger.debug(f"decode::{joined_seq = }")

        try:
            return joined_seq.decode(encoding="utf-8")
        except UnicodeDecodeError:
            return ""


if __name__ == "__main__":
    t = Tokenizer(
        Path("data/TinyStoriesV2-GPT4-train.txt"),
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    start_time = time.time()
    vocab, merges = t.train()
    delta_time = time.time() - start_time
    logger.info(f"trained tokenizer in {delta_time} sec.")

    # t.write_merges_to_file()
