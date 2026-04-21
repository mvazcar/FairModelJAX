"""Shell-out helper for Fair's FP.EXE — golden-file validation.

Fair's ``OUT`` files (``03_us_model/fmout.txt``,
``04_mc_model/mcj_extracted/OUT``) are the output of running FP.EXE on
the paired ``fminput.txt`` / ``MC.INP`` spec + the paired data files.
Those OUT files are the reference our test suite compares against.

This module provides:

* ``run_fpexe(inp_file, working_dir=None, timeout=300)`` — invoke
  FP.EXE on one of Fair's INP files and capture its console output.
* ``extract_equation_block(out_text, eq_number)`` — pull one equation's
  coefficient block out of an OUT string.
* ``diff_coefs_vs_fpexe(our_coefs, fpexe_out_text, eq_number)`` — compare
  our coefficients to what FP.EXE produced.

Intended usage: regenerate Fair's OUT from scratch after upgrading
inputs, or spot-check that a specific equation our pipeline estimates
matches what FP.EXE produces on the same data.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .. import config


FP_EXE_PATH = config.PROJECT_ROOT / "02_executable" / "FP.EXE"


def run_fpexe(
    inp_file: Path | str,
    working_dir: Path | str | None = None,
    timeout: int = 600,
) -> str:
    """Run FP.EXE on ``inp_file`` and return its console output.

    Args:
      inp_file: Path to a Fair INP file (e.g. ``MC.INP``). Relative
        paths resolve against ``working_dir``.
      working_dir: Directory FP.EXE runs in. Defaults to the parent of
        ``inp_file`` so the program can find companion .DAT files.
      timeout: Seconds before aborting (FP.EXE on the full MC.INP
        takes 3–5 minutes on a 2020-era laptop).

    Returns:
      Captured stdout as a single string.

    Raises:
      FileNotFoundError: If FP.EXE isn't available.
      subprocess.TimeoutExpired: If the run exceeds ``timeout``.
    """
    if not FP_EXE_PATH.exists():
        raise FileNotFoundError(f"FP.EXE not found at {FP_EXE_PATH}")

    inp_path = Path(inp_file)
    if working_dir is None:
        working_dir = inp_path.parent
    inp_name = inp_path.name if inp_path.is_absolute() else str(inp_path)

    proc = subprocess.run(
        [str(FP_EXE_PATH)],
        input=f"INPUT FILE={inp_name};\n",
        cwd=str(working_dir),
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout


def extract_equation_block(out_text: str, eq_number: int) -> str | None:
    """Return the text of one ``Equation number = N`` block.

    The block runs from the header line through the ``SE of equation``
    line (the terminator Fair's OUT uses). Returns None if the equation
    isn't present.
    """
    pattern = re.compile(
        rf"^Equation number =\s+{eq_number}\b.*?SE of equation[^\n]*",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(out_text)
    return m.group(0) if m else None


def extract_coefs_from_block(block: str) -> dict[str, float]:
    """Parse a single equation block for its ``{variable: coef}`` map.

    Uses the same line format our ``mc_model.parse_mc_out`` handles: rows
    ``<index> <name> (<lag>) <coef> <SE> <t-stat> <mean>``.
    """
    coef_re = re.compile(
        r"^\s*\d+\s+(\S+)\s*\(\s*(-?\d+)\)\s+([-+\d\.E]+)",
        re.MULTILINE,
    )
    out: dict[str, float] = {}
    for name, lag, val in coef_re.findall(block):
        key = f"{name}({int(lag):+d})" if lag != "0" else f"{name}(0)"
        out[key] = float(val)
    return out


def diff_coefs_vs_fpexe(
    our_coefs: dict[str, float],
    fpexe_block: str,
) -> dict[str, float]:
    """Return ``{key: our_coef − fpexe_coef}`` for every matching key.

    Useful for regression-testing that our estimator reproduces FP.EXE's
    numerical output.
    """
    fair = extract_coefs_from_block(fpexe_block)
    deltas: dict[str, float] = {}
    for key, our in our_coefs.items():
        if key in fair:
            deltas[key] = our - fair[key]
    return deltas
