"""Benchmark spaCy multiprocessing (`n_process`) on the machine you run
this on -- run it directly on wherever the app actually deploys, since
the answer depends heavily on the platform's multiprocessing start
method (fork vs. spawn), available CPU cores, and spaCy pipeline size,
none of which a developer's laptop can stand in for reliably.

Usage:
    python scripts/benchmark_n_process.py
    python scripts/benchmark_n_process.py --sizes 100 500 2000 --n-process 1 2 4

What to do with the results: pass the corpus size where a higher
n_process first *consistently* beats n_process=1 as `texts_per_process`
to `lexical_profiler.auto_n_process()` (see its docstring), or thread a
custom value through Reference.from_corpus(..., n_process=...) /
LexicalProfiler.profile_texts(..., n_process=...) directly if you'd
rather not touch the shared default.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import platform
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lexical_profiler import Reference  # noqa: E402


def _cgroup_cpu_quota() -> str:
    """Best-effort check for a Linux cgroup CPU limit, since a shared/
    containerized host's os.cpu_count() often reports the *host's* core
    count, not the (possibly much smaller) quota actually allotted to
    this container -- a real risk on shared university hosting."""
    for path in (
        "/sys/fs/cgroup/cpu.max",  # cgroup v2: "<quota> <period>" or "max <period>"
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",  # cgroup v1
    ):
        try:
            return Path(path).read_text().strip()
        except OSError:
            continue
    return "not found (not on Linux, or no cgroup limit set)"


def make_texts(n: int) -> dict[str, str]:
    fables_dir = REPO_ROOT / "examples" / "targets"
    fables = [p.read_text(encoding="utf-8") for p in sorted(fables_dir.glob("*.txt"))]
    if not fables:
        raise SystemExit(f"No sample texts found under {fables_dir}")
    return {f"doc{i}.txt": fables[i % len(fables)] for i in range(n)}


def time_build(texts: dict[str, str], n_process: int) -> float:
    t0 = time.perf_counter()
    Reference.from_corpus(texts, band_size=10, lemmatize=True, n_process=n_process)
    return time.perf_counter() - t0


def run(sizes: list[int], n_processes: list[int]) -> None:
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"multiprocessing start method: {multiprocessing.get_start_method()}")
    print(f"os.cpu_count(): {os.cpu_count()}")
    print(f"cgroup CPU quota: {_cgroup_cpu_quota()}")
    print()
    print(f"{'n_docs':>8} | " + " | ".join(f"n={n:<3}" for n in n_processes))
    for n_docs in sizes:
        texts = make_texts(n_docs)
        times = [time_build(texts, n) for n in n_processes]
        cells = " | ".join(f"{t:6.2f}s" for t in times)
        print(f"{n_docs:>8} | {cells}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 2000, 6000],
                         help="Corpus sizes (document counts) to test.")
    parser.add_argument("--n-process", type=int, nargs="+", default=[1, 2, 4],
                         dest="n_processes", help="n_process values to compare at each size.")
    args = parser.parse_args()

    # Run inside a thread, mirroring the Flask app's background-job
    # execution context (webapp_flask/services/jobs.py) rather than bare
    # top-level script code -- the safe pattern documented on
    # lexical_profiler.auto_n_process/classify_texts for using n_process > 1.
    th = threading.Thread(target=run, args=(args.sizes, args.n_processes))
    th.start()
    th.join()
