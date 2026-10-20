"""
Test cases for dna_factory/rewards/reasoning.py and its instances in my_rewards.py.

Each case reproduces a leak seen in a Qwen3-4B-Thinking-2507 GRPO run: a truncated completion
whose reasoning happened to contain the gold number, a box changed once per candidate, a
"Final Answer" block repeated to the cap, and verbatim loops.
"""

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory import rewards
from dna_factory.rewards.reasoning import (
    make_answer_format_penalty,
    make_ngram_loop_penalty,
    make_strict_accuracy_reward,
)

MAX_LEN = 100
FINISHED = [[1] * 10]
TRUNCATED = [[1] * MAX_LEN]
ANSWER = "So the answer is 49/3.\n</think>\n\n### Final Answer\n\n$$\n\\boxed{\\dfrac{49}{3}}\n$$"


def _completions(*contents):
    return [[{"role": "assistant", "content": c}] for c in contents]


def _accuracy(
    content, ids=FINISHED, solution="\\boxed{\\frac{49}{3}}", **factory_kwargs
):
    reward = make_strict_accuracy_reward(max_completion_len=MAX_LEN, **factory_kwargs)
    return reward(
        prompts=["p"],
        completions=_completions(content),
        completion_ids=ids,
        solution=[solution],
    )[0]


class TestStrictAccuracyReward:
    def test_closed_correct_answer(self):
        assert _accuracy(ANSWER) == 1.0

    def test_truncated_completion_scores_zero_even_if_correct(self):
        """The ellipse leak: cut at the cap, gold number present -> must not pay."""
        assert _accuracy(ANSWER, ids=TRUNCATED) == 0.0

    def test_unclosed_reasoning_scores_zero(self):
        assert _accuracy("Marked as the correct answer: \\boxed{\\dfrac{49}{3}}") == 0.0

    def test_box_in_reasoning_only_scores_zero(self):
        content = "Maybe \\boxed{\\frac{49}{3}}.\n</think>\n\nThe answer is forty-nine thirds."
        assert _accuracy(content) == 0.0

    def test_spammed_candidates_score_zero(self):
        """\\boxed{-145} -> -146 -> -147: one right box among wrong ones does not pay."""
        content = "</think>\n\n\\boxed{-145} or \\boxed{-146} or \\boxed{-147}"
        assert _accuracy(content, solution="\\boxed{-146}") == 0.0

    def test_repeated_identical_box_is_graded_once(self):
        content = "</think>\n\n\\boxed{40}\n\nFinal Answer: \\boxed{40}"
        assert _accuracy(content, solution="\\boxed{40}") == 1.0

    def test_equivalent_distinct_boxes_all_verify(self):
        content = "</think>\n\n\\boxed{\\frac{1}{2}} = \\boxed{0.5}"
        assert _accuracy(content, solution="\\boxed{\\frac12}") == 1.0

    def test_too_many_distinct_answers(self):
        content = "</think>\n\n" + " ".join(f"\\boxed{{{i}}}" for i in range(5))
        assert _accuracy(content, solution="\\boxed{0}") == 0.0

    def test_units_optional(self):
        content = "</think>\n\n\\boxed{45}"
        assert _accuracy(content, solution="\\boxed{45^\\circ}") == 1.0
        assert (
            _accuracy(content, solution="\\boxed{45^\\circ}", require_units=True) == 0.0
        )
        assert (
            _accuracy(
                "</think>\n\n\\boxed{45^\\circ}",
                solution="\\boxed{45^\\circ}",
                require_units=True,
            )
            == 1.0
        )
        assert (
            _accuracy(
                "</think>\n\n\\boxed{17}", solution="\\boxed{17\\%}", require_units=True
            )
            == 0.0
        )

    def test_missing_or_unparseable_gold_is_none(self):
        reward = make_strict_accuracy_reward(max_completion_len=MAX_LEN)
        assert reward(["p"], _completions(ANSWER), FINISHED) == [None]
        assert _accuracy(ANSWER, solution="") is None

    def test_logs_the_graded_answer(self):
        reward = make_strict_accuracy_reward(max_completion_len=MAX_LEN, name="acc")
        logged = {}
        reward(
            ["p", "p"],
            _completions(ANSWER, ANSWER),
            [FINISHED[0], TRUNCATED[0]],
            solution=["\\boxed{\\frac{49}{3}}"] * 2,
            log_extra=logged.__setitem__,
        )
        assert logged["acc_answer"] == ["\\dfrac{49}{3}", "[truncated]"]


class TestAnswerFormatPenalty:
    @pytest.fixture
    def penalty(self):
        return make_answer_format_penalty(max_completion_len=MAX_LEN)

    def test_single_closure_is_free(self, penalty):
        assert penalty(["p"], _completions(ANSWER), FINISHED) == [0.0]

    def test_repeated_final_answer_blocks(self, penalty):
        block = "### Final Answer\n\n$$\n\\boxed{40}\n$$\n\n"
        content = "</think>\n\n" + block * 3
        # 2 extra boxes + 2 extra "Final Answer" headers.
        assert penalty(["p"], _completions(content), FINISHED) == [-1.0]
        content = "</think>\n\n" + block * 2
        assert penalty(["p"], _completions(content), FINISHED) == [-0.5]

    def test_unclosed_reasoning_ended_by_eos(self, penalty):
        assert penalty(["p"], _completions("no close \\boxed{1}"), FINISHED) == [-1.0]

    def test_truncated_is_left_to_overlong_penalty(self, penalty):
        assert penalty(["p"], _completions("no close"), TRUNCATED) == [None]


class TestNgramLoopPenalty:
    def test_distinct_tokens_are_free(self):
        penalty = make_ngram_loop_penalty(ngram_size=4)
        assert penalty(["p"], [""], [list(range(200))]) == [0.0]

    def test_verbatim_loop_saturates(self):
        penalty = make_ngram_loop_penalty(ngram_size=4)
        assert penalty(["p"], [""], [[1, 2, 3, 4, 5] * 40]) == [-1.0]

    def test_partial_loop_is_linear(self):
        penalty = make_ngram_loop_penalty(ngram_size=1, tolerance=0.0, saturation=0.5)
        # 4 tokens, 2 unique -> repeated share 0.5 -> -1.0; 3 unique of 4 -> 0.25 -> -0.5.
        assert penalty(["p"], [""], [[1, 2, 3, 3]]) == [-0.5]

    def test_short_completion_is_free(self):
        assert make_ngram_loop_penalty()(["p"], [""], [[1] * 10]) == [0.0]

    def test_invalid_band(self):
        with pytest.raises(ValueError):
            make_ngram_loop_penalty(tolerance=0.5, saturation=0.5)


@pytest.mark.parametrize(
    "name",
    [
        "strict_accuracy_reward",
        "answer_format_penalty",
        "ngram_loop_penalty",
        "soft_overlong_penalty_8k",
    ],
)
def test_instances_resolve_by_dotted_path_and_pickle(name):
    """grpo.py resolves via getattr; async GRPO pickles rewards into a spawned worker."""
    reward = getattr(rewards, name)
    assert reward.__name__ == name
    assert pickle.loads(pickle.dumps(reward)).__name__ == name
