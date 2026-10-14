from pathlib import Path

from docanswer.eval import calibrate, evaluate, load_questions, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_eval_runs_and_reports_zero_leaks(engine):
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    assert len(questions) >= 30
    result = evaluate(engine, questions, mode="hybrid")
    m = result["metrics"]
    assert m["access_control_leaks"] == 0
    assert m["refusal_accuracy_restricted"] == 1.0
    assert 0 <= m["mrr@5"] <= 1 and m["hit@1"] <= m["hit@3"]
    assert "| hit@3 |" in to_markdown([result])


def test_calibration_returns_threshold_in_range(engine):
    threshold = calibrate(engine, load_questions(ROOT / "eval" / "dev_questions.jsonl"))
    assert 0.05 <= threshold <= 0.95
