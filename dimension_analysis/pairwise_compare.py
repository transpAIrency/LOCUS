import argparse
import asyncio
import random
import json

from collections import Counter
from tqdm.asyncio import tqdm
from pathlib import Path
from openai import AsyncOpenAI

from judge import judge_pair, AXES, Judgment


SEED = 2026


def sample_pairs(
        dataset: list, num_samples: int, rng: random.Random
) -> list[tuple]:
    """
    Sample unique unordered pairs by rejection — avoids enumerating the
    O(n^2) combination space. Efficient when num_samples << C(n, 2).
    """
    n = len(dataset)
    total = n * (n - 1) // 2
    num_samples = min(num_samples, total)

    seen: set[tuple[int, int]] = set()
    pairs = []
    while len(pairs) < num_samples:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((dataset[i], dataset[j]))
    return pairs


async def run_batch(
        batch_pairs,
        axis: str,
        *,
        concurrency: int = 96,
) -> list[Judgment | BaseException]:
    sem = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(max_retries=5, timeout=120.0)

    pair_results = await tqdm.gather(
        *[judge_pair(client, sem, a, b, axis, model="gpt-5.4-nano") for a, b in batch_pairs]
    )
    return [item for results in pair_results for item in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sampled.jsonl", help="path to jsonl of sampled laws")
    parser.add_argument("--output-dir", default="output/morality", help="directory for results.jsonl")
    parser.add_argument("--axis", default="enforcement_discretion", choices=list(AXES))
    parser.add_argument("--n-laws", type=int, default=1000, help="use the first n laws from --input")
    parser.add_argument("--n-pairs", type=int, default=10000, help="number of unordered pairs to judge")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    results_path = output_dir / "results.jsonl"

    with open(args.input) as fp:
        laws = [json.loads(line) for line in fp][:args.n_laws]

    rng = random.Random(SEED)
    pairs = sample_pairs(laws, args.n_pairs, rng)

    raw_results = asyncio.run(run_batch(pairs, args.axis, concurrency=256))

    judgments = [r for r in raw_results if isinstance(r, Judgment)]
    failures = [r for r in raw_results if isinstance(r, BaseException)]

    with open(results_path, "w", encoding="utf-8") as fp:
        for judgment in judgments:
            fp.write(judgment.model_dump_json() + "\n")
    print(
        f"Wrote {len(judgments)} judgments to {results_path} "
        f"({len(failures)} of {len(pairs) * 2} directions failed)"
    )
    if failures:
        breakdown = Counter(type(e).__name__ for e in failures)
        print(f"Failure breakdown: {dict(breakdown)}")


if __name__ == "__main__":
    main()
