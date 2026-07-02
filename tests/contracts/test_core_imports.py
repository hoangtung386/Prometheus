import subprocess
import sys


def test_public_api_does_not_import_optional_visualization_stack() -> None:
    command = (
        "import sys; import prometheus.api; "
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
