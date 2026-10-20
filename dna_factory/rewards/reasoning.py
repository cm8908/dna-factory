"""
Reasoning-completion reward family for GRPO on thinking models, whose completion is
`<reasoning>...</think>\n\n<answer section>` (the opening `<think>` sits in the prompt). A bare 0/1
accuracy reward is blind to how the answer was produced: it grades text cut off mid-reasoning,
accepts a box spammed once per candidate, and scores a padded answer section the same as a
clean one. These factories close those gaps:

- `make_strict_accuracy_reward` — math_verify accuracy, but only on a finished completion's
  answer section (after the single `</think>`), and only when every distinct `\\boxed{}` there is
  correct.
- `make_answer_format_penalty` — non-positive: reasoning never closed, or extra closing blocks
  (`\\boxed`, "Final Answer") repeated in the answer section.
- `make_ngram_loop_penalty` — non-positive: share of repeated token n-grams above a tolerance,
  i.e. verbatim loops.

Concrete instances live in my_rewards.py. Contract: docs/grpo-rewards.md.
"""

import functools
import logging
import re
import threading

from .generative import _to_text, _warn_once
from .verifiable import _normalize

THINK_CLOSE = "</think>"

_BOXED_RE = re.compile(r"\\boxed\{")
_FINAL_ANSWER_RE = re.compile(r"final answer", re.IGNORECASE)
_DEGREE_MARKS = ("\\circ", "°")


def _answer_section(text):
    """Text after the single `</think>`; None when reasoning never closed or closed twice."""
    if text.count(THINK_CLOSE) != 1:
        return None
    return text.split(THINK_CLOSE, 1)[1]


def _all_boxed(text):
    """Every `\\boxed{...}` content in text, scanned with balanced braces."""
    found = []
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            found.append(text[m.end() : i - 1])
    return found


def _units_ok(gold, answer):
    """math_verify drops % and ° when comparing, so a gold that carries one must see it echoed."""
    if "%" in gold and "%" not in answer and "percent" not in answer:
        return False
    has_degree = any(mark in answer for mark in (*_DEGREE_MARKS, "degree"))
    return has_degree or not any(mark in gold for mark in _DEGREE_MARKS)


def _math_verify_timeouts():
    # math_verify times out with signal.alarm(), which only works on the main thread (as in TRL).
    if threading.current_thread() is threading.main_thread():
        return 10, 5
    logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
    logging.getLogger("math_verify.grader").setLevel(logging.ERROR)
    return None, None


def _strict_accuracy_reward(
    prompts,
    completions,
    completion_ids,
    solution=None,
    log_extra=None,
    *,
    max_completion_len,
    max_distinct_answers,
    require_units,
    name,
    **kwargs,
):
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    n = len(completions)
    if solution is None:
        _warn_once(
            f"{name}:missing_solution",
            f"{name}: no `solution` column in this dataset; scoring nothing (all None).",
        )
        return [None] * n

    parsing_timeout, verify_timeout = _math_verify_timeouts()
    # Same extraction as TRL's accuracy_reward, applied to one boxed answer at a time.
    extraction_config = [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(units=True),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )
    ]
    gold_cache = {}

    def grade(text, ids, gold):
        if gold not in gold_cache:
            gold_cache[gold] = parse(gold, parsing_timeout=parsing_timeout)
        gold_parsed = gold_cache[gold]
        if not gold_parsed:
            return None, "[gold unparseable]"
        # A completion cut at the cap never finished, whatever it happened to box on the way.
        if len(ids) >= max_completion_len:
            return 0.0, "[truncated]"
        answer = _answer_section(text)
        if answer is None:
            return 0.0, "[reasoning not closed once]"
        # Dedupe by normalized string so a box repeated verbatim is graded once.
        distinct = list({_normalize(b): b for b in _all_boxed(answer)}.values())
        if not distinct:
            return 0.0, "[no boxed answer]"
        if len(distinct) > max_distinct_answers:
            return 0.0, f"[{len(distinct)} distinct boxed answers]"
        # Every distinct box must be right, so hedging across candidates never pays.
        for boxed in distinct:
            if require_units and not _units_ok(gold, boxed):
                return 0.0, f"[missing unit] {boxed}"
            answer_parsed = parse(
                f"\\boxed{{{boxed}}}",
                extraction_config=extraction_config,
                extraction_mode="first_match",
                parsing_timeout=parsing_timeout,
            )
            if not verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout):
                return 0.0, boxed
        return 1.0, " | ".join(distinct)

    rewards, notes = [], []
    for completion, ids, gold in zip(
        completions, completion_ids, solution, strict=True
    ):
        try:
            reward, note = grade(_to_text(completion), ids, gold)
        except Exception as e:  # noqa: BLE001 - sympy/mpmath raise arbitrary types on poisoned input
            _warn_once(
                f"{name}:grade_error",
                f"{name}: grading raised {type(e).__name__} ({e}); that sample scores None.",
            )
            reward, note = None, f"[error] {type(e).__name__}"
        rewards.append(reward)
        notes.append(note)

    if log_extra is not None:
        log_extra(f"{name}_answer", notes)
    return rewards


def make_strict_accuracy_reward(
    max_completion_len,
    max_distinct_answers=4,
    require_units=False,
    name="strict_accuracy_reward",
):
    """
    Build a 0/1 math accuracy reward that only credits a finished, closed answer.

    Scores 0.0 when the completion hit `max_completion_len` (truncated), when `</think>` does not
    appear exactly once, when the answer section has no `\\boxed{}`, more than
    `max_distinct_answers` distinct ones, or any distinct one that math_verify rejects, and (with
    `require_units`) when a gold `%`/`°` is missing from the box. None when the gold `solution`
    is missing or unparseable, or grading raises. Thinking models only: a completion without
    `</think>` always scores 0.

    Args:
        max_completion_len: the recipe's `max_completion_length`; not auto-synced.
        max_distinct_answers: cap on distinct boxed answers (bounds verify cost).
        require_units: require `%`/`°` in the box when the gold has one. Off by default: a prompt
            that asks for "a number" makes a bare `45` for gold `45^\\circ` the instructed answer,
            and inspect_evals/math strips both marks too.
        name: sets __name__ (the `rewards/<name>/mean` metric key).

    Returns:
        (prompts, completions, completion_ids, solution=None, log_extra=None, **kwargs)
        -> list[float | None].
    """
    reward = functools.partial(
        _strict_accuracy_reward,
        max_completion_len=max_completion_len,
        max_distinct_answers=max_distinct_answers,
        require_units=require_units,
        name=name,
    )
    reward.__name__ = name
    return reward


def _answer_format_penalty(
    prompts,
    completions,
    completion_ids,
    *,
    max_completion_len,
    per_extra_block,
    **kwargs,
):
    scores = []
    for completion, ids in zip(completions, completion_ids, strict=True):
        # Truncation is the overlong penalty's job; don't charge it twice.
        if len(ids) >= max_completion_len:
            scores.append(None)
            continue
        answer = _answer_section(_to_text(completion))
        if answer is None:
            scores.append(-1.0)
            continue
        extra = max(0, len(_BOXED_RE.findall(answer)) - 1) + max(
            0, len(_FINAL_ANSWER_RE.findall(answer)) - 1
        )
        scores.append(-min(1.0, per_extra_block * extra))
    return scores


def make_answer_format_penalty(
    max_completion_len, per_extra_block=0.25, name="answer_format_penalty"
):
    """
    Build a non-positive penalty for a finished completion's answer-section hygiene.

    -1.0 when `</think>` does not appear exactly once (EOS inside the reasoning, or a second
    close); otherwise `-per_extra_block` for every `\\boxed` beyond the first and every
    "final answer" beyond the first in the answer section, floored at -1.0. None for a completion
    at `max_completion_len` (left to the overlong penalty). Gives an all-correct group a
    gradient toward closing once instead of padding.

    Returns:
        (prompts, completions, completion_ids, **kwargs) -> list[float | None].
    """
    reward = functools.partial(
        _answer_format_penalty,
        max_completion_len=max_completion_len,
        per_extra_block=per_extra_block,
    )
    reward.__name__ = name
    return reward


def _ngram_loop_penalty(
    prompts, completions, completion_ids, *, ngram_size, tolerance, saturation, **kwargs
):
    scores = []
    for ids in completion_ids:
        ids = list(ids)
        if len(ids) < 2 * ngram_size:
            scores.append(0.0)
            continue
        ngrams = list(zip(*[ids[i:] for i in range(ngram_size)], strict=False))
        repeated = 1 - len(set(ngrams)) / len(ngrams)
        scores.append(
            -min(1.0, max(0.0, repeated - tolerance) / (saturation - tolerance))
        )
    return scores


def make_ngram_loop_penalty(
    ngram_size=20, tolerance=0.1, saturation=0.5, name="ngram_loop_penalty"
):
    """
    Build a non-positive penalty for verbatim loops: the share of repeated token `ngram_size`-grams,
    0 up to `tolerance`, linear down to -1.0 at `saturation`.

    Long n-grams, unlike TRL's 3-gram repetition penalty, leave ordinary math reasoning alone (its
    3-gram repeat share is ~0.4 at the median); a 20-gram repeat share above ~0.15 is a loop.

    Returns:
        (prompts, completions, completion_ids, **kwargs) -> list[float].
    """
    if not 0 <= tolerance < saturation <= 1:
        raise ValueError(
            f"need 0 <= tolerance < saturation <= 1, got {tolerance=}, {saturation=}"
        )
    reward = functools.partial(
        _ngram_loop_penalty,
        ngram_size=ngram_size,
        tolerance=tolerance,
        saturation=saturation,
    )
    reward.__name__ = name
    return reward
