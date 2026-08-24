from ham_exam_eval_stats.hamexam_score import score


def test_score_counts_correct_wrong_blank_and_subgroups():
    answers = [
        {"id": "1", "answer": "A"},
        {"id": "2", "answer": "B"},
        {"id": "3", "answer": None},
    ]
    key = {"1": "A", "2": "C", "3": "D"}
    exam = [
        {"id": "1", "cluster": "E1A"},
        {"id": "2", "cluster": "E2B"},
        {"id": "3", "cluster": "E2B"},
    ]

    result = score(answers, key, exam)

    assert result["right"] == 1
    assert result["wrong"] == 1
    assert result["unanswered"] == 1
    assert result["total"] == 3
    assert result["by_sub"] == {"E1": (1, 1), "E2": (0, 2)}
