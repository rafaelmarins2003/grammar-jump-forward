import copy

from jumpforward.replication import first_difference, replication_criteria


def test_first_difference_handles_identical_and_prefix_sequences():
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1], [1, 2]) == 1
    assert first_difference([], []) == 0
    assert first_difference([1, 2], [1, 2]) == 2


def test_replication_separates_performance_from_preservation_and_old_target():
    summary = {
        "complete_pairs": 198,
        "subsets": {str(i): {"wall_speedup": {"n": 38}} for i in range(5)},
        "criteria": dict.fromkeys(
            (
                "ci_lower_above_1",
                "all_subsets_at_least_1",
                "p95_at_most_1_10",
                "actual_grammar_drafts",
            ),
            True,
        ),
        "failure_questions": {"candidate_only": []},
        "failures": [{"error": "generation limit or truncation"}],
    }
    summary["criteria"].update(quality_upper_below_2_percent=False, median_at_least_1_4=False)
    original = copy.deepcopy(summary)
    result = replication_criteria(summary)
    assert result["performance_reproduced"]
    assert not result["preservation_screen_passed"]
    assert not result["old_1_4_target_met_descriptively"]
    assert summary == original
    summary["criteria"]["quality_upper_below_2_percent"] = True
    assert replication_criteria(summary)["preservation_screen_passed"]
    summary["failure_questions"]["candidate_only"] = [["simple", "new-failure"]]
    assert not replication_criteria(summary)["preservation_screen_passed"]
    summary["complete_pairs"] = 180
    assert not replication_criteria(summary)["performance_reproduced"]
