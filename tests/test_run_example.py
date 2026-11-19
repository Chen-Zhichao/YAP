"""The example launcher delegates to existing entry points; no mechanics here."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/run_example.sh'


def test_help_requires_no_setup():
    result = subprocess.run(['bash', str(SCRIPT), '--help'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and '--run-dir' in result.stdout


@pytest.mark.parametrize('args', [[], ['--unknown'], ['--run-dir'], ['--run-dir', 'relative'], ['--run-dir', '/']])
def test_invalid_arguments_fail_before_generation(args):
    result = subprocess.run(['bash', str(SCRIPT), *args], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'Usage:' in result.stderr


@pytest.mark.parametrize('plan_only', [False, True])
def test_existing_commands_are_quoted_and_storage_is_explicit(tmp_path, plan_only):
    data = tmp_path / 'data with spaces'
    (data / 'distribution/operators').mkdir(parents=True)
    python = tmp_path / 'python fixture'
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" __CALL__ "$NUMBA_NUM_THREADS" "$TMPDIR" "$YAP_RUN_ROOT" "$@" >> "$CAPTURE"\n')
    python.chmod(0o700)
    capture = tmp_path / 'calls'
    run = tmp_path / 'output with spaces'
    env = dict(os.environ, CAPTURE=str(capture), NUMBA_NUM_THREADS='1', TMPDIR='stale-path', YAP_RUN_ROOT='stale-root')
    args = ['bash', str(SCRIPT), '--run-dir', str(run), '--data-root', str(data), '--python', str(python)]
    if plan_only:
        args.append('--plan-only')
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    calls = [part.split(b'\0') for part in capture.read_bytes().split(b'__CALL__\0')[1:]]
    assert len(calls) == 3
    generation, calculation = calls[1], calls[2]
    assert generation[0] == calculation[0] == b'8'
    trial = calculation[2].decode()
    assert trial.startswith(str(run / 'trial_')) and calculation[1].decode() == trial + '/work/temp'
    assert str(ROOT / 'generate_assembly_design.py').encode() in generation
    assert b'coupon:BI4_Prepared50_Quadrature19626' in generation
    assert str(ROOT / 'D2W/calculator_main.py').encode() in calculation
    assert (b'--plan-only' in calculation) is plan_only
    assert not run.exists(), 'the shell itself must not create paths before application storage validation'


def test_missing_data_has_a_direct_error(tmp_path):
    result = subprocess.run(['bash', str(SCRIPT), '--run-dir', str(tmp_path / 'run'),
        '--data-root', str(tmp_path / 'missing')], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'Operator data not found' in result.stderr
