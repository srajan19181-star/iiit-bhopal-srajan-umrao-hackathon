"""Root-level convenience entry point.

Usage (from repo root):

  Run the batch pipeline:
      py main.py pipeline

  Launch the Streamlit dashboard:
      py main.py dashboard

  Start the FastAPI server (optional):
      py main.py api

  Run smoke tests:
      py main.py test
"""
import subprocess
import sys


def main():
    cmds = {
        "pipeline":  [sys.executable, "-m", "src.risk_engine.run_batch"],
        "dashboard": [sys.executable, "-m", "streamlit", "run", "src/app.py"],
        "api":       [sys.executable, "-m", "uvicorn", "src.risk_engine.api:app", "--reload"],
        "test":      [sys.executable, "scripts/smoke_test.py"],
    }

    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("Usage: py main.py [pipeline|dashboard|api|test]")
        print()
        print("  pipeline   — Run the NLP batch pipeline (data/raw → data/output)")
        print("  dashboard  — Launch the Streamlit dashboard at http://localhost:8501")
        print("  api        — Start the FastAPI server at http://127.0.0.1:8000")
        print("  test       — Run end-to-end smoke tests")
        sys.exit(1)

    cmd = cmds[sys.argv[1]]
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
