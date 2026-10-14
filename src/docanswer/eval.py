"""Offline evaluation: retrieval quality, refusal behaviour and access-control leaks.

    python -m docanswer.eval                      # evaluate eval/questions.jsonl
    python -m docanswer.eval --calibrate          # pick the refusal threshold on eval/dev_questions.jsonl

The refusal threshold is calibrated on a separate dev set so the reported test numbers are not
tuned on the test questions themselves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from docanswer.config import Settings
from docanswer.models import Answer, clearance_for
from docanswer.pipeline import DocAnswer
from docanswer.retrieval import Mode

MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")
RETRIEVAL_CATEGORIES = {"answerable", "cross_lingual"}


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_hit_rank(answer: Answer, expected: list[dict]) -> int | None:
    targets = {(e["doc_id"], e["page"]) for e in expected}
    for s in answer.retrieved:
        if (s.chunk.doc_id, s.chunk.page) in targets:
            return s.rank
    return None


def _cites_expected(answer: Answer, expected: list[dict]) -> bool:
    targets = {(e["doc_id"], e["page"]) for e in expected}
    return not answer.refused and any((c.doc_id, c.page) in targets for c in answer.citations)


def _leaks(answer: Answer, role: str) -> int:
    clearance = clearance_for(role)
    return sum(s.chunk.access_level > clearance for s in answer.retrieved)


def _ratio(num: float, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def evaluate(app: DocAnswer, questions: list[dict], mode: Mode = "hybrid") -> dict:
    per_question, ranks, leaks = [], [], 0
    groups: dict[str, list[dict]] = {}
    for q in questions:
        answer = app.ask(q["question"], q["role"], mode=mode)
        rank = _first_hit_rank(answer, q["expected"]) if q["expected"] else None
        row = {
            "id": q["id"],
            "category": q["category"],
            "lang": q["lang"],
            "role": q["role"],
            "refused": answer.refused,
            "confidence": answer.confidence,
            "hit_rank": rank,
            "cited_expected": _cites_expected(answer, q["expected"]),
            "leaks": _leaks(answer, q["role"]),
        }
        leaks += row["leaks"]
        if q["category"] in RETRIEVAL_CATEGORIES:
            ranks.append(rank)
        groups.setdefault(q["category"], []).append(row)
        per_question.append(row)

    answerable = groups.get("answerable", []) + groups.get("cross_lingual", [])
    should_refuse = groups.get("unanswerable", []) + groups.get("restricted", [])
    metrics = {
        "mode": mode,
        "n_questions": len(questions),
        "n_answerable": len(ranks),
        "hit@1": _ratio(sum(r == 1 for r in ranks), len(ranks)),
        "hit@3": _ratio(sum(r is not None and r <= 3 for r in ranks), len(ranks)),
        "mrr@5": _ratio(sum(1 / r for r in ranks if r), len(ranks)),
        "answered_with_correct_citation": _ratio(sum(r["cited_expected"] for r in answerable), len(answerable)),
        "false_refusal_rate": _ratio(sum(r["refused"] for r in answerable), len(answerable)),
        "refusal_accuracy_unanswerable": _ratio(
            sum(r["refused"] for r in groups.get("unanswerable", [])), len(groups.get("unanswerable", []))
        ),
        "refusal_accuracy_restricted": _ratio(
            sum(r["refused"] for r in groups.get("restricted", [])), len(groups.get("restricted", []))
        ),
        "access_control_leaks": leaks,
        "by_category": {
            cat: {
                "n": len(rows),
                "hit@3": _ratio(sum(r["hit_rank"] is not None and r["hit_rank"] <= 3 for r in rows), len(rows)),
                "refused": sum(r["refused"] for r in rows),
            }
            for cat, rows in groups.items()
        },
    }
    return {"metrics": metrics, "questions": per_question, "_should_refuse": len(should_refuse)}


def calibrate(app: DocAnswer, questions: list[dict]) -> float:
    """Pick the threshold that maximises balanced accuracy (answer vs refuse) on the dev set."""
    rows = [
        (
            app.retriever.search(q["question"], q["role"], k=app.settings.top_k).confidence,
            q["category"] in RETRIEVAL_CATEGORIES,
        )
        for q in questions
    ]
    scores: dict[float, float] = {}
    for t in [i / 100 for i in range(5, 96)]:
        pos = [c >= t for c, answerable in rows if answerable]
        neg = [c < t for c, answerable in rows if not answerable]
        scores[t] = (sum(pos) / max(len(pos), 1) + sum(neg) / max(len(neg), 1)) / 2
    best_score = max(scores.values())
    optimal = [t for t, s in scores.items() if s == best_score]
    best = optimal[len(optimal) // 2]  # middle of the optimal band = widest margin on both sides
    print(f"dev confidences (confidence, answerable): {sorted((round(c, 3), a) for c, a in rows)}")
    print(
        f"optimal band {optimal[0]:.2f}-{optimal[-1]:.2f}; chosen threshold {best:.2f} "
        f"(dev balanced accuracy {best_score:.3f})"
    )
    return best


def to_markdown(results: list[dict]) -> str:
    keys = [
        "hit@1",
        "hit@3",
        "mrr@5",
        "answered_with_correct_citation",
        "false_refusal_rate",
        "refusal_accuracy_unanswerable",
        "refusal_accuracy_restricted",
        "access_control_leaks",
    ]
    lines = ["| metric | " + " | ".join(r["metrics"]["mode"] for r in results) + " |", "|---|" + "---|" * len(results)]
    lines += [f"| {k} | " + " | ".join(str(r["metrics"][k]) for r in results) + " |" for k in keys]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocAnswer offline.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("eval/dev_questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval"))
    parser.add_argument("--calibrate", action="store_true", help="only calibrate the threshold on the dev set")
    args = parser.parse_args()

    # Evaluation is always offline and deterministic, whatever the environment says.
    settings = replace(Settings.from_env(), llm_provider="extractive", embedder="tfidf")
    app = DocAnswer.from_settings(settings, rebuild=True)
    if args.calibrate:
        calibrate(app, load_questions(args.dev))
        return

    questions = load_questions(args.questions)
    results = [evaluate(app, questions, mode) for mode in MODES]
    for r in results:
        r.pop("_should_refuse")
    payload = {
        "settings": {
            "llm_provider": settings.llm_provider,
            "embedder": settings.embedder,
            "top_k": settings.top_k,
            "refusal_threshold": settings.refusal_threshold,
            "chunk_max_words": settings.chunk_max_words,
            "n_chunks": len(app.retriever.index.chunks),
        },
        "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    table = to_markdown(results)
    (args.out / "results.md").write_text(
        f"Offline extractive mode, synthetic corpus, {len(questions)} questions, "
        f"threshold {settings.refusal_threshold}.\n\n{table}"
    )
    print(table)
    for row in results[-1]["questions"]:
        flag = (
            "ok"
            if (row["cited_expected"] or (row["refused"] and row["category"] in ("unanswerable", "restricted")))
            else "MISS"
        )
        print(
            f"{row['id']} {row['category']:<13} conf={row['confidence']:.3f} rank={row['hit_rank']} "
            f"refused={row['refused']} {flag}"
        )


if __name__ == "__main__":
    main()
