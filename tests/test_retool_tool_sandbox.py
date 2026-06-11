import asyncio
import sys
from pathlib import Path

RETOOL_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "retool_post_train_all"
sys.path.insert(0, str(RETOOL_EXAMPLE_DIR))

from tool_sandbox import PythonSandbox  # noqa: E402


def test_python_sandbox_import_check_ignores_from_in_comments():
    code = """boxes = [0] * 7
for card in range(1, 2016):  # Place cards from 1 to 2015
    box = card % 7
print(box)
"""

    result = asyncio.run(PythonSandbox(timeout=5).execute_code(code))

    assert "Import of '1'" not in result
    assert "Output:" in result


def test_python_sandbox_rejects_sympy_import():
    safe, message = PythonSandbox()._check_code_safety("from sympy import symbols, Eq, solve")

    assert not safe
    assert message == "Import of 'sympy' is not allowed"
