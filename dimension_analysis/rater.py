import argparse
import csv
import json
from pathlib import Path

import trueskill

from judge import Judgment


def refit(judgments: list[Judgment]) -> dict[str, trueskill.Rating]:
    """fresh ratings initialized for every id seen in judgments, then replayed in order."""
    all_ids = {j.law_a_id for j in judgments} | {j.law_b_id for j in judgments}
    return compute_trueskill(judgments, all_ids)


def compute_trueskill(judgments: list[Judgment], all_ids: set[str]) -> dict[str, trueskill.Rating]:
    """initialize default trueskill ratings for all_ids and update them by replaying judgments in order."""
    env = trueskill.TrueSkill()
    ratings = {law_id: env.create_rating() for law_id in all_ids}
    for j in judgments:
        a, b = ratings[j.law_a_id], ratings[j.law_b_id]
        if j.winner_id is None:
            ratings[j.law_a_id], ratings[j.law_b_id] = env.rate_1vs1(a, b, drawn=True)
        elif j.winner_id == j.law_a_id:
            ratings[j.law_a_id], ratings[j.law_b_id] = env.rate_1vs1(a, b)
        else:
            ratings[j.law_b_id], ratings[j.law_a_id] = env.rate_1vs1(b, a)
    return ratings


def build_leaderboard(ratings: dict[str, trueskill.Rating], id_to_law: dict[str, dict]) -> list[dict]:
    """convert ratings to ranked dicts (with full law text) sorted by trueskill expose score."""
    rows = []
    for law_id, rating in ratings.items():
        law = id_to_law.get(law_id, {})
        text = (law.get("header", "") + "\n" + law.get("content", "")).strip()
        score = trueskill.expose(rating)
        rows.append({
            "id": law_id,
            "Skill (mu)": round(rating.mu, 3),
            "Uncertainty (sigma)": round(rating.sigma, 3),
            "Score (mu-3*sigma)": round(score, 3),
            "text": text,
        })
    rows.sort(key=lambda x: x["Score (mu-3*sigma)"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def load_judgments(path: str) -> list[Judgment]:
    judgments = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data.setdefault("order", "forward")
            judgments.append(Judgment.model_validate(data))
    return judgments


def load_laws(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fp:
        return {
            law["id"]: law
            for law in (json.loads(line) for line in fp if line.strip())
        }


def write_leaderboard(rows: list[dict], path: str) -> None:
    fields = ["rank", "id", "Skill (mu)", "Uncertainty (sigma)", "Score (mu-3*sigma)", "text"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="output/enforcement_discretion/results.jsonl")
    parser.add_argument("--laws", default="sampled.jsonl")
    parser.add_argument("--output", default=None,
                        help="leaderboard csv path (default: <results-dir>/leaderboard.csv)")
    args = parser.parse_args()

    judgments = load_judgments(args.results)
    laws = load_laws(args.laws)

    ratings = refit(judgments)
    rows = build_leaderboard(ratings, laws)

    output = args.output or str(Path(args.results).parent / "leaderboard.csv")
    write_leaderboard(rows, output)
    print(f"Wrote {len(rows)} ranked laws from {len(judgments)} judgments to {output}")


if __name__ == "__main__":
    main()
