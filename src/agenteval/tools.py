from __future__ import annotations

import ast
import math
import operator
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field, HttpUrl

from agenteval.config import Settings
from agenteval.external import (
    DeclarativeCheck,
    ExternalAgentEvaluator,
    ExternalCase,
    ExternalSuite,
    HttpAgentTarget,
    HttpTargetConfig,
)
from agenteval.llm import create_llm_client

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Tool(Protocol[InputT, OutputT]):
    name: str
    input_schema: type[InputT]
    output_schema: type[OutputT]

    async def invoke(self, inputs: InputT) -> OutputT: ...


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=500)


class CalculatorOutput(BaseModel):
    value: float


_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        value = _BINARY_OPS[type(node.op)](_eval_math(node.left), _eval_math(node.right))
        if not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("result outside safe numeric range")
        return float(value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_math(node.operand)))
    raise ValueError("unsupported expression")


class CalculatorTool:
    name = "calculator"
    input_schema = CalculatorInput
    output_schema = CalculatorOutput

    async def invoke(self, inputs: CalculatorInput) -> CalculatorOutput:
        tree = ast.parse(inputs.expression, mode="eval")
        return CalculatorOutput(value=_eval_math(tree))


class WebFetchInput(BaseModel):
    url: HttpUrl
    max_bytes: int = Field(default=100_000, ge=1, le=1_000_000)


class WebFetchOutput(BaseModel):
    url: str
    status_code: int
    text: str


class WebFetchTool:
    name = "web_fetch"
    input_schema = WebFetchInput
    output_schema = WebFetchOutput

    async def invoke(self, inputs: WebFetchInput) -> WebFetchOutput:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(str(inputs.url))
            response.raise_for_status()
        body = response.content[: inputs.max_bytes].decode("utf-8", errors="replace")
        return WebFetchOutput(url=str(response.url), status_code=response.status_code, text=body)


class PythonExecInput(BaseModel):
    code: str = Field(min_length=1, max_length=5_000)


class PythonExecOutput(BaseModel):
    result: Any


_ALLOWED_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Subscript,
    ast.Slice,
    ast.ListComp,
    ast.comprehension,
    ast.Call,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
}


class PythonExecTool:
    name = "python_exec"
    input_schema = PythonExecInput
    output_schema = PythonExecOutput

    async def invoke(self, inputs: PythonExecInput) -> PythonExecOutput:
        tree = ast.parse(inputs.code, mode="exec")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"disallowed syntax: {type(node).__name__}")
            if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS
            ):
                raise ValueError("only allowlisted builtins may be called")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError("dunder names are disallowed")
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(tree, "<sandbox>", "exec"), namespace, namespace)
        result = namespace.get("result")
        return PythonExecOutput(result=result)


class ExternalEvaluationInput(BaseModel):
    target: HttpTargetConfig
    case: ExternalCase
    default_checks: list[DeclarativeCheck] = Field(default_factory=list)
    min_llm_score: float = Field(default=3.0, ge=0, le=5)
    trial: int = Field(default=1, ge=1)
    perturbation_id: str | None = None


class ExternalEvaluationOutput(BaseModel):
    result: dict[str, Any]


class ExternalEvaluationTool:
    name = "external_evaluate"
    input_schema = ExternalEvaluationInput
    output_schema = ExternalEvaluationOutput

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def invoke(self, inputs: ExternalEvaluationInput) -> ExternalEvaluationOutput:
        suite = ExternalSuite(
            name="distributed-node",
            target=inputs.target,
            default_checks=inputs.default_checks,
            cases=[inputs.case],
            min_llm_score=inputs.min_llm_score,
        )
        evaluator = ExternalAgentEvaluator(
            HttpAgentTarget(inputs.target),
            create_llm_client(self.settings.judge_provider, self.settings),
            target_name=inputs.target.base_url,
        )
        result = await evaluator.evaluate_case(
            inputs.case,
            suite.default_checks,
            suite.min_llm_score,
            inputs.trial,
            inputs.perturbation_id,
        )
        return ExternalEvaluationOutput(result=result.model_dump(mode="json"))


class EvaluationBarrierInput(BaseModel):
    expected_nodes: int = Field(ge=1)


class EvaluationBarrierOutput(BaseModel):
    completed_nodes: int


class EvaluationBarrierTool:
    name = "evaluation_barrier"
    input_schema = EvaluationBarrierInput
    output_schema = EvaluationBarrierOutput

    async def invoke(self, inputs: EvaluationBarrierInput) -> EvaluationBarrierOutput:
        return EvaluationBarrierOutput(completed_nodes=inputs.expected_nodes)


ToolType = (
    CalculatorTool
    | WebFetchTool
    | PythonExecTool
    | ExternalEvaluationTool
    | EvaluationBarrierTool
)


def default_tools(settings: Settings | None = None) -> dict[str, ToolType]:
    resolved = settings or Settings()
    tools: list[ToolType] = [
        CalculatorTool(),
        WebFetchTool(),
        PythonExecTool(),
        ExternalEvaluationTool(resolved),
        EvaluationBarrierTool(),
    ]
    return {tool.name: tool for tool in tools}
