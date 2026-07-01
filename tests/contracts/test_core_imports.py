import subprocess
import sys


def test_public_api_does_not_import_legacy_namespace() -> None:
    command = (
        "import sys; import prometheus.api; "
        "assert not any(name.startswith('prometheus.legacy') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
