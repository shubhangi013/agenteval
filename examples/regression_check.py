"""Compare two benchmark artifacts without shelling out to the CLI.

Run with: uv run python examples/regression_check.py BASELINE.json CANDIDATE.json
"""

import sys
from pathlib import Path

from agenteval.evaluation import EvalRun, regression_deltas


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: regression_check.py BASELINE.json CANDIDATE.json")
        return 2
    baseline = EvalRun.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    candidate = EvalRun.model_validate_json(Path(sys.argv[2]).read_text(encoding="utf-8"))
    lines, regressed = regression_deltas(baseline, candidate)
    print("\n".join(lines))
    return int(regressed)


if __name__ == "__main__":
    raise SystemExit(main())
