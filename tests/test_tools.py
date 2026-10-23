import pytest

from agenteval.tools import CalculatorInput, CalculatorTool, PythonExecInput, PythonExecTool


@pytest.mark.asyncio
async def test_calculator_evaluates_allowlisted_math() -> None:
    output = await CalculatorTool().invoke(CalculatorInput(expression="(17 * 19) + 4"))
    assert output.value == 327


@pytest.mark.asyncio
async def test_calculator_rejects_calls() -> None:
    with pytest.raises(ValueError, match="unsupported expression"):
        await CalculatorTool().invoke(CalculatorInput(expression="abs(-1)"))


@pytest.mark.asyncio
async def test_python_exec_returns_result() -> None:
    output = await PythonExecTool().invoke(
        PythonExecInput(code="values = [3, 1, 2]\nresult = sorted(values)")
    )
    assert output.result == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["import os", "result = open('x')", "result = (1).__class__"])
async def test_python_exec_rejects_unsafe_syntax(code: str) -> None:
    with pytest.raises(ValueError, match=r"disallowed syntax|allowlisted"):
        await PythonExecTool().invoke(PythonExecInput(code=code))
