from typing import Any, Self

import pytest

from agenteval.discovery import discover_external_suite, generate_adversarial_suite
from agenteval.external import (
    CheckCondition,
    DeclarativeCheck,
    ExternalAgentEvaluator,
    ExternalCase,
    ExternalSuite,
    HttpAgentTarget,
    HttpTargetConfig,
    resolve_path,
    score_output,
)
from agenteval.llm import LLMClient, LLMResponse, MockLLMClient
from agenteval.safety import PerturbationSpec, SafetyPolicy


def valid_output() -> dict[str, Any]:
    return {
        "analysis": {"required_skills": ["Python", "Redis"]},
        "fit": {"score": 82, "priority": "P1"},
        "strengths": ["Agent platform work"],
        "outreach_notes": [{"body": "Python work"}, {"body": "Redis work"}],
    }


def check(name: str, operator: str, path: str, **arguments: Any) -> DeclarativeCheck:
    return DeclarativeCheck.model_validate(
        {"name": name, "operator": operator, "path": path, **arguments}
    )


def test_path_resolution_supports_list_wildcards() -> None:
    assert resolve_path(valid_output(), "outreach_notes.*.body") == [
        "Python work",
        "Redis work",
    ]


def test_declarative_checks_cover_structured_outputs() -> None:
    checks = [
        check("exists", "exists", "analysis"),
        check("equals", "equals", "fit.priority", value="P1"),
        check("in", "in", "fit.priority", values=["P0", "P1"]),
        check("range", "range", "fit.score", minimum=70, maximum=100),
        check("contains", "contains", "analysis.required_skills", terms=["Python"]),
        check("not_contains", "not_contains", "strengths", terms=["unrelated"]),
        check("count", "count", "outreach_notes", minimum=2, maximum=3),
        check("length", "length", "outreach_notes.*.body", maximum=280),
        check("regex", "regex", "strengths", value="agent.*work"),
    ]
    assert all(result.passed for result in score_output(valid_output(), checks))


def test_conditional_check_is_skipped_when_condition_is_false() -> None:
    output = {"verdict": {"recommendation": "skip"}, "outreach_notes": []}
    outreach = check("outreach", "count", "outreach_notes", minimum=2)
    outreach.when = CheckCondition(
        path="verdict.recommendation", operator="not_equals", value="skip"
    )
    result = score_output(output, [outreach])[0]
    assert result.passed
    assert result.skipped


class FakeTarget:
    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        assert inputs["company"] == "Acme"
        return valid_output()


def suite(expected_priority: str = "P1") -> ExternalSuite:
    return ExternalSuite(
        name="test-suite",
        target=HttpTargetConfig(base_url="http://agent", endpoint="/invoke"),
        default_checks=[check("schema", "exists", "fit.score")],
        cases=[
            ExternalCase(
                id="case",
                category="fit",
                input={"company": "Acme"},
                rubric="Return a suitable priority.",
                checks=[
                    check("priority", "equals", "fit.priority", value=expected_priority)
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_external_evaluator_invokes_generic_target() -> None:
    run = await ExternalAgentEvaluator(FakeTarget(), MockLLMClient(), "fake").evaluate(
        suite()
    )
    assert run.suite == "test-suite"
    assert run.pass_rate == 1.0
    assert run.results[0].passed
    assert run.results[0].agent_latency_ms >= 0
    assert run.results[0].judge_latency_ms >= 0


@pytest.mark.asyncio
async def test_external_evaluator_repeats_and_summarizes_reliability() -> None:
    run = await ExternalAgentEvaluator(FakeTarget(), MockLLMClient(), "fake").evaluate(
        suite(), repetitions=3
    )
    assert len(run.results) == 3
    assert [item.trial for item in run.results] == [1, 2, 3]
    assert run.pass_at_k == 1.0
    assert run.pass_to_k == 1.0
    assert run.reliability[0].attempts == 3
    assert run.categories[0].attempts == 3


@pytest.mark.asyncio
async def test_external_evaluator_expands_perturbations_and_reports_robustness() -> None:
    definition = suite()
    definition.perturbations = [
        PerturbationSpec(id="spaces", path="company", strategy="normalize_whitespace")
    ]
    run = await ExternalAgentEvaluator(FakeTarget(), MockLLMClient(), "fake").evaluate(
        definition
    )
    assert len(run.results) == 2
    assert run.results[1].task_id == "case::perturbation:spaces"
    assert run.results[1].perturbation_id == "spaces"
    assert run.robustness[0].variants == 1
    assert run.robustness[0].pass_rate == 1.0


class StatefulTarget:
    def __init__(self) -> None:
        self.count = 99

    async def reset(self) -> None:
        self.count = 0

    async def snapshot(self) -> dict[str, int]:
        return {"count": self.count}

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.count += 1
        return {**valid_output(), "effects": [{"action": "read"}]}


def stateful_suite() -> ExternalSuite:
    definition = suite()
    definition.cases[0].state_preconditions = [
        check("clean", "equals", "count", value=0)
    ]
    definition.cases[0].state_postconditions = [
        check("incremented", "equals", "count", value=1)
    ]
    definition.cases[0].threat_tags = ["stateful", "rbac"]
    definition.cases[0].safety = SafetyPolicy(
        side_effect_path="effects.*.action",
        role_input_path="role",
        authorized_actions_by_role={"viewer": ["read"]},
    )
    definition.cases[0].input["role"] = "viewer"
    return definition


@pytest.mark.asyncio
async def test_stateful_sandbox_resets_and_checks_postconditions() -> None:
    run = await ExternalAgentEvaluator(
        StatefulTarget(), MockLLMClient(), "stateful"
    ).evaluate(stateful_suite())
    result = run.results[0]
    assert result.passed
    assert result.safety is not None and result.safety.passed
    assert result.threat_tags == ["stateful", "rbac"]
    assert {item.name for item in result.checks} >= {
        "state.pre.clean",
        "state.post.incremented",
        "safety_policy",
    }
    assert run.safety_pass_rate == 1.0
    assert {item.threat: item.pass_rate for item in run.safety_categories} == {
        "rbac": 1.0,
        "stateful": 1.0,
    }


@pytest.mark.asyncio
async def test_stateful_sandbox_rejects_concurrent_cases() -> None:
    with pytest.raises(ValueError, match="concurrency=1"):
        await ExternalAgentEvaluator(
            StatefulTarget(), MockLLMClient(), "stateful"
        ).evaluate(stateful_suite(), concurrency=2)


@pytest.mark.asyncio
async def test_failed_safety_output_is_redacted_before_judging() -> None:
    class UnsafeTarget:
        async def run(self, inputs: dict[str, Any]) -> dict[str, str]:
            return {"answer": "token=abc123"}

    class InspectingJudge(LLMClient):
        async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
            assert "token=abc123" not in prompt
            assert "REDACTED_BY_SAFETY_POLICY" in prompt
            return LLMResponse('{"score": 0, "reason": "unsafe"}', 0.0)

    definition = suite()
    definition.default_checks = [check("answer_exists", "exists", "answer")]
    definition.cases[0].checks = []
    definition.cases[0].safety = SafetyPolicy(
        forbidden_output_patterns=[r"token=\w+"]
    )
    run = await ExternalAgentEvaluator(
        UnsafeTarget(), InspectingJudge(), "unsafe"
    ).evaluate(definition)
    assert not run.results[0].passed
    assert run.results[0].output == "[REDACTED_BY_SAFETY_POLICY]"
    assert run.results[0].checks[0].detail == "[REDACTED_BY_SAFETY_POLICY]"


@pytest.mark.asyncio
async def test_failed_declarative_check_is_a_hard_gate() -> None:
    run = await ExternalAgentEvaluator(FakeTarget(), MockLLMClient(), "fake").evaluate(
        suite(expected_priority="P3")
    )
    assert not run.results[0].passed
    assert run.results[0].llm_score == 4
    assert run.disagreement_rate == 1.0


def test_http_target_allows_runtime_url_override() -> None:
    target = HttpAgentTarget(
        HttpTargetConfig(base_url="http://default", endpoint="/invoke"),
        "http://override/",
    )
    assert target.base_url == "http://override"


@pytest.mark.asyncio
async def test_http_target_maps_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"wrapper": {"answer": 4}}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("agenteval.external.httpx.AsyncClient", FakeClient)
    target = HttpAgentTarget(
        HttpTargetConfig(
            base_url="http://agent",
            endpoint="/invoke",
            response_path="wrapper",
            request_overrides={"dry_run": True},
            timeout_s=30,
        )
    )
    result = await target.run({"question": "2 + 2", "dry_run": False})

    assert result == {"answer": 4}
    assert captured == {
        "timeout": 30,
        "url": "http://agent/invoke",
        "json": {"question": "2 + 2", "dry_run": True},
    }


@pytest.mark.asyncio
async def test_discovery_generates_valid_suite_and_pins_target_url() -> None:
    class DiscoveryModel(LLMClient):
        async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
            assert json_mode
            assert "AGENT_EVALUATION_DISCOVERY" in prompt
            generated = suite().model_copy(deep=True)
            generated.target.base_url = "http://model-invented"
            return LLMResponse(generated.model_dump_json(), 0.0)

    discovered = await discover_external_suite(
        DiscoveryModel(),
        base_url="http://deployed-agent/",
        openapi={"openapi": "3.1.0", "paths": {"/invoke": {"post": {}}}},
        notes="This agent prioritizes suitable opportunities.",
        samples=[{"fit": {"score": 82, "priority": "P1"}}],
    )

    assert discovered.target.base_url == "http://deployed-agent"
    assert discovered.target.endpoint == "/invoke"
    assert discovered.cases[0].rubric == "Return a suitable priority."
    assert discovered.draft


@pytest.mark.asyncio
async def test_discovery_rejects_invented_endpoint() -> None:
    class DiscoveryModel(LLMClient):
        async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
            generated = suite().model_copy(deep=True)
            generated.target.endpoint = "/delete-everything"
            return LLMResponse(generated.model_dump_json(), 0.0)

    with pytest.raises(ValueError, match="not a POST operation"):
        await discover_external_suite(
            DiscoveryModel(),
            base_url="http://deployed-agent",
            openapi={"openapi": "3.1.0", "paths": {"/invoke": {"post": {}}}},
        )


@pytest.mark.asyncio
async def test_adversarial_generator_pins_reviewed_target_and_marks_draft() -> None:
    source = suite()
    source.target.request_overrides = {"save": False}

    class AdversarialModel(LLMClient):
        async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
            assert json_mode
            assert "AGENT_ADVERSARIAL_EVALUATION_DISCOVERY" in prompt
            assert "Acme" not in prompt
            assert '"company": "<str>"' in prompt
            generated = suite().model_copy(deep=True)
            generated.target.base_url = "http://invented"
            generated.target.endpoint = "/admin/delete"
            generated.cases[0].threat_tags = ["prompt_injection"]
            generated.cases[0].safety = SafetyPolicy(must_refuse=True)
            return LLMResponse(generated.model_dump_json(), 0.0)

    draft = await generate_adversarial_suite(AdversarialModel(), source, case_count=1)
    assert draft.draft
    assert draft.source_suite == source.name
    assert draft.target == source.target
    assert draft.target.request_overrides == {"save": False}