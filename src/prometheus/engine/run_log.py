"""Durable training log.

Printing to stdout is not enough on the supported workstation. Colab runtimes disconnect
mid-run — the trainer is built to resume from ``last.ckpt`` precisely because they do — and
when that happens the whole printed history is gone. ``metrics.jsonl`` survives, but it only
holds numbers: the class weights that were actually used, the pretrained-loading report and
any stale-cache warning are exactly the lines you want when a run comes out wrong, and they
are not numbers.

:class:`RunLog` writes those lines to ``<run_dir>/train.log`` as well as stdout, appending so
a resumed run continues the same file rather than truncating the history it is resuming from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RunLog"]


class RunLog:
    """Write a line to stdout and append it to a log file.

    Each line is timestamped in the file but not on stdout, so the notebook stays readable
    while the persisted copy stays useful for reconstructing a timeline after the fact.
    """

    def __init__(self, path: str | Path, echo: bool = True) -> None:
        self.path = Path(path)
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def session(self, header: str) -> None:
        """Mark the start of a run or resume, so appended sessions stay distinguishable."""
        self(f"{'=' * 20} {header} {'=' * 20}")

    def __call__(self, message: str) -> None:
        if self.echo:
            print(message)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
