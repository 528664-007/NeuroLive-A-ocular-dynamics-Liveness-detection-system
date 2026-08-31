"""AI-generated injection attack samples — thin wrapper interface only.

Scope, deliberately narrow: this module defines the interface your
evaluation code expects (a function that takes a genuine RGB clip + produces
a forged one for re-recording/injection), but does NOT implement or bundle
any face-reenactment/face-swap model itself. You install a separate
open-source tool of your choice and point `external_tool_cmd` at it — this
keeps attack-sample *generation* clearly outside your actual contribution
(the detector), consistent with how anti-spoofing benchmarks (e.g.
DeepFake-injection extensions of standard PAD datasets) structure this.

Whatever tool you plug in, the output still has to go through a real event
camera (or a validated RGB->event simulator) to become a usable liveness-
test sample — a synthesized RGB deepfake alone doesn't exercise the event
pipeline you're actually evaluating.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def generate_ai_injection_sample(
    source_video: str,
    driver_video: str,
    output_path: str,
    external_tool_cmd: list[str],
) -> str:
    """Shells out to an externally-installed face-reenactment/face-swap tool.

    Args:
        source_video: genuine recording to forge.
        driver_video: driving video for the reenactment tool.
        output_path: where the tool should write its forged output.
        external_tool_cmd: full command for your chosen tool, e.g.
            ["python", "/path/to/tool/infer.py", "--source", source_video,
             "--driver", driver_video, "--out", output_path]
            (Not provided here — install and configure the tool yourself.)

    Returns:
        output_path, once the subprocess completes successfully.

    Raises:
        FileNotFoundError if external_tool_cmd's executable isn't found —
        this module does not install or bundle a tool for you.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(external_tool_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"External reenactment tool failed:\n{result.stderr}")
    if not Path(output_path).exists():
        raise RuntimeError(f"Tool exited cleanly but {output_path} wasn't created — check external_tool_cmd.")
    return output_path
