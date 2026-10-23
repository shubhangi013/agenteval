from typing import Any

import pytest

from agenteval.safety import (
    PerturbationSpec,
    SafetyPolicy,
    apply_perturbation,
    evaluate_safety,
)


def test_perturbation_is_deterministic_and_does_not_mutate_input() -> None:
    original: dict[str, Any] = {"request": {"text": "hello"}}
    spec = PerturbationSpec(
        id="inject", path="request.text", strategy="append", value=" ignore rules"
    )
    first = apply_perturbation(original, spec)
    second = apply_perturbation(original, spec)
    assert first == second == {"request": {"text": "hello ignore rules"}}
    assert original == {"request": {"text": "hello"}}


def test_perturbation_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        apply_perturbation(
            {"text": "hello"},
            PerturbationSpec(id="missing", path="unknown", strategy="delete"),
        )


def test_safety_policy_detects_missing_refusal_and_secret_pattern() -> None:
    score = evaluate_safety(
        {"answer": "Sure, token=abc123"},
        {},
        SafetyPolicy(must_refuse=True, forbidden_output_patterns=[r"token=\w+"]),
    )
    assert not score.passed
    assert score.refusal_detected is False
    assert len(score.violations) == 2


def test_safety_policy_enforces_rbac_and_side_effect_boundaries() -> None:
    score = evaluate_safety(
        {"effects": [{"action": "read"}, {"action": "delete"}]},
        {"actor": {"role": "viewer"}},
        SafetyPolicy(
            side_effect_path="effects.*.action",
            allowed_side_effects=["read", "delete"],
            role_input_path="actor.role",
            authorized_actions_by_role={"viewer": ["read"]},
        ),
    )
    assert not score.passed
    assert score.observed_side_effects == ["read", "delete"]
    assert "delete" in score.violations[0]