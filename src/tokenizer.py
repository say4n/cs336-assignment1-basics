from collections import defaultdict
import sys
import click
from pathlib import Path
import regex as re
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@click.command()
@click.option("--input_path", help="Path to corpus", type=Path)
@click.option("--vocab_size", help="Max vocab size in integer", type=int)
@click.option(
    "--special_tokens", help="comma separated list of special_tokens", type=str
)
def tokenize(input_path, vocab_size, special_tokens):
    """Tokenize a corpus with BPE."""
    corpus = open(input_path).read()
    logger.debug(f"{corpus = }")

    # pre_tokenizer_pattern = (
    #     r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # )
    pre_tokenizer_pattern = r"\w+\s+"

    pre_tokenized = re.finditer(pre_tokenizer_pattern, corpus)

    pre_tokenized_counts = defaultdict(int)
    for chunks in pre_tokenized:
        pre_tokenized_counts[chunks.group()] += 1

    logger.debug(f"{pre_tokenized_counts = }")

    # TODO: BPE: merge
    # TODO: BPE: speedup?


if __name__ == "__main__":
    test_corpus = "data/short.txt"
    special_tokens = "<|endoftext|>"
    vocab_size = 400

    ctx = click.Context(tokenize)
    ctx.forward(tokenize, input_path=test_corpus, vocab_size=vocab_size, special_tokens=special_tokens)
