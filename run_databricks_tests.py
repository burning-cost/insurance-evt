"""Run insurance-evt tests on Databricks serverless compute.

Embeds all source files as base64 in the notebook to avoid
filesystem permission issues on serverless.
"""
import os
import sys
import time
import base64
import pathlib

_env_path = pathlib.Path.home() / ".config" / "burning-cost" / "databricks.env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.jobs import NotebookTask, SubmitTask

w = WorkspaceClient()

base = pathlib.Path("/home/ralph/repos/insurance-evt")
src_dir = base / "src" / "insurance_evt"
test_dir = base / "tests"

source_files = {}
for f in src_dir.glob("*.py"):
    source_files[f"src/insurance_evt/{f.name}"] = f.read_text()

test_files = {}
for f in test_dir.glob("*.py"):
    test_files[f"tests/{f.name}"] = f.read_text()

pyproject = (base / "pyproject.toml").read_text()

# Build notebook lines
lines = [
    "# Databricks notebook source",
    "import subprocess, sys, os, base64",
    "",
    "os.makedirs('/tmp/pkg/src/insurance_evt', exist_ok=True)",
    "os.makedirs('/tmp/pkg/tests', exist_ok=True)",
    "",
]

for relpath, content in source_files.items():
    encoded = base64.b64encode(content.encode()).decode()
    lines.append(
        f"open('/tmp/pkg/{relpath}', 'w').write(base64.b64decode('{encoded}').decode())"
    )

encoded_pyproject = base64.b64encode(pyproject.encode()).decode()
lines.append(
    f"open('/tmp/pkg/pyproject.toml', 'w').write(base64.b64decode('{encoded_pyproject}').decode())"
)

for relpath, content in test_files.items():
    encoded = base64.b64encode(content.encode()).decode()
    lines.append(
        f"open('/tmp/pkg/{relpath}', 'w').write(base64.b64decode('{encoded}').decode())"
    )

lines.extend([
    "",
    "print('Files written to /tmp/pkg')",
    "",
    "# Install test dependencies only",
    "r = subprocess.run(",
    "    [sys.executable, '-m', 'pip', 'install', '-q',",
    "     'numpy', 'scipy', 'pandas', 'matplotlib', 'pytest'],",
    "    capture_output=True, text=True",
    ")",
    "print('Deps install:', r.returncode)",
    "if r.returncode != 0:",
    "    print(r.stderr[-500:])",
    "",
    "# Use PYTHONPATH to make insurance_evt importable",
    "env = os.environ.copy()",
    "env['PYTHONPATH'] = '/tmp/pkg/src:' + env.get('PYTHONPATH', '')",
    "",
    "result = subprocess.run(",
    "    [sys.executable, '-m', 'pytest', '/tmp/pkg/tests/', '-v', '--tb=short'],",
    "    capture_output=True, text=True, cwd='/tmp/pkg', env=env",
    ")",
    "output = result.stdout[-10000:]",
    "if result.stderr:",
    "    output += '\\nSTDERR:\\n' + result.stderr[-500:]",
    "",
    "print(output)",
    "",
    "try:",
    "    dbutils.notebook.exit(output)",
    "except NameError:",
    "    pass",
    "",
    "if result.returncode != 0:",
    "    raise Exception(f'Tests failed (exit code {result.returncode})')",
])

notebook_source = "\n".join(lines)
workspace_path = "/insurance-evt-tests"

w.workspace.import_(
    path=workspace_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(notebook_source.encode()).decode(),
    overwrite=True,
)
print(f"Notebook uploaded to {workspace_path}")

run = w.jobs.submit(
    run_name="insurance-evt-tests",
    tasks=[
        SubmitTask(
            task_key="run-tests",
            notebook_task=NotebookTask(notebook_path=workspace_path),
            timeout_seconds=900,
        )
    ],
)
run_id = run.run_id
print(f"Run ID: {run_id}")
print(f"https://dbc-150a27f5-e1e7.cloud.databricks.com/#job/run/{run_id}")
print("Waiting for completion...")

while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state
    result_state = run_state.state.result_state
    print(f"  State: {life_cycle} / {result_state}")
    if life_cycle in (
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ):
        break
    time.sleep(20)

if run_state.tasks:
    for task_run in run_state.tasks:
        try:
            output = w.jobs.get_run_output(run_id=task_run.run_id)
            if output.notebook_output:
                print("\n=== NOTEBOOK OUTPUT ===")
                print(output.notebook_output.result[-10000:])
            if output.error:
                print("\n=== ERROR ===")
                print(output.error)
            if output.error_trace:
                print("\n=== ERROR TRACE ===")
                print(output.error_trace[:3000])
        except Exception as e:
            print(f"Could not get task output: {e}")

final_state = run_state.state.result_state
print(f"\nFinal state: {final_state}")

if str(final_state) != "RunResultState.SUCCESS":
    sys.exit(1)

print("SUCCESS")
