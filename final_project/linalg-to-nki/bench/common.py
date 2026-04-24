"""
Shared benchmark utilities.

Timing model
------------
torch_xla kernel launches are asynchronous: the XLA client queues work and
control returns immediately. To get meaningful wall-clock numbers we:

  1. call the kernel,
  2. reference its output (materializes the graph),
  3. `xm.wait_device_ops()` to flush the queue,
  4. stop the timer.

`.cpu()` would also force a sync but adds an HBM->host copy we don't want
inside the loop.

Each configuration is warmed up (N_WARMUP runs -- the first one triggers
the NKI compile, which can be multi-second) and then measured N_MEASURE
times; the returned ms/iter is the median, which is robust to the
occasional GC hiccup or DMA stall.

Output model
------------
Callers accumulate `BenchRow` records via `run_case()` and then pass the
full list to `render(...)`, which prints ONE clean summary table at the
end (plus an optional CSV dump and matplotlib PNGs). Per-iteration stdout
noise from the NKI compiler is captured and dropped inside the timing
loops.
"""
from __future__ import annotations

import contextlib
import csv
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


N_WARMUP = 3
N_MEASURE = 10

# Toggle to True if you want to see the NKI compiler logs during the run.
VERBOSE_COMPILE = False


@dataclass
class BenchRow:
    bench: str           # "add", "matmul", "matmul_relu"
    config: str          # e.g. "BS=1024" or "CFG=64x64x64"
    shape: str           # e.g. "n=65536" or "MKN=128x128x512"
    case_tag: str        # human label
    gen_ms: float
    ref_ms: float
    gen_err: float
    ref_err: float
    gen_ok: bool
    ref_ok: bool

    @property
    def speedup(self) -> float:
        return self.ref_ms / self.gen_ms if self.gen_ms > 0 else float("nan")


@contextlib.contextmanager
def _maybe_silence():
    """Redirect stdout+stderr to /dev/null unless VERBOSE_COMPILE is set."""
    if VERBOSE_COMPILE:
        yield
        return
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


def _sync():
    xm.wait_device_ops()


def _time_kernel(kernel, args) -> tuple[float, float, float, torch.Tensor]:
    """Return (median_ms, p10_ms, p90_ms, last_output)."""
    with _maybe_silence():
        for _ in range(N_WARMUP):
            out = kernel(*args)
            _ = out.shape
        _sync()

        samples_ms: list[float] = []
        for _ in range(N_MEASURE):
            t0 = time.perf_counter()
            out = kernel(*args)
            _ = out.shape
            _sync()
            t1 = time.perf_counter()
            samples_ms.append((t1 - t0) * 1e3)

    samples_ms.sort()
    median = statistics.median(samples_ms)
    p10 = samples_ms[max(0, len(samples_ms) // 10)]
    p90 = samples_ms[min(len(samples_ms) - 1, (9 * len(samples_ms)) // 10)]
    return median, p10, p90, out


def run_case(
    bench: str, config: str, shape: str, case_tag: str,
    gen_kernel, ref_kernel, args, ground_truth: torch.Tensor,
    atol: float = 1e-4, rtol: float = 1e-2,
) -> BenchRow:
    """Run gen + ref on the same inputs, collect timing + error."""
    gen_med, _, _, gen_out = _time_kernel(gen_kernel, args)
    ref_med, _, _, ref_out = _time_kernel(ref_kernel, args)

    gen_err = (gen_out - ground_truth).abs().max().item()
    ref_err = (ref_out - ground_truth).abs().max().item()
    gen_ok = torch.allclose(gen_out, ground_truth, atol=atol, rtol=rtol)
    ref_ok = torch.allclose(ref_out, ground_truth, atol=atol, rtol=rtol)

    return BenchRow(
        bench=bench, config=config, shape=shape, case_tag=case_tag,
        gen_ms=gen_med, ref_ms=ref_med,
        gen_err=gen_err, ref_err=ref_err,
        gen_ok=gen_ok, ref_ok=ref_ok,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ok(b: bool) -> str:
    return "\u2713" if b else "\u2717"  # check / cross


def render_table(rows: list[BenchRow]) -> str:
    if not rows:
        return "(no rows)\n"
    headers = ["config", "shape", "case",
               "gen ms", "ref ms", "speedup", "gen err", "ref err", "ok"]
    table: list[list[str]] = [headers]
    for r in rows:
        table.append([
            r.config,
            r.shape,
            r.case_tag,
            f"{r.gen_ms:.3f}",
            f"{r.ref_ms:.3f}",
            f"{r.speedup:.2f}x",
            f"{r.gen_err:.1e}",
            f"{r.ref_err:.1e}",
            f"{_ok(r.gen_ok)}/{_ok(r.ref_ok)}",
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    sep = "  "
    out_lines: list[str] = []
    for idx, row in enumerate(table):
        line = sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        out_lines.append(line)
        if idx == 0:
            out_lines.append(sep.join("-" * w for w in widths))
    return "\n".join(out_lines) + "\n"


def render_summary(rows: list[BenchRow]) -> str:
    """One-line aggregate: geomean speedup, # correct, worst gen error."""
    if not rows:
        return ""
    import math
    speedups = [r.speedup for r in rows if r.gen_ms > 0]
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    all_gen_ok = sum(r.gen_ok for r in rows)
    all_ref_ok = sum(r.ref_ok for r in rows)
    worst = max(r.gen_err for r in rows)
    return (
        f"summary: {len(rows)} cases  |  geomean speedup = {geomean:.2f}x  |  "
        f"gen correct = {all_gen_ok}/{len(rows)}  |  "
        f"ref correct = {all_ref_ok}/{len(rows)}  |  "
        f"worst gen |err| = {worst:.2e}\n"
    )


def save_csv(rows: list[BenchRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) + ["speedup"])
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["speedup"] = r.speedup
            w.writerow(d)


def save_plot(rows: list[BenchRow], path: Path, title: str) -> None:
    """Grouped bar chart: one group per (config, shape), two bars (gen, ref)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"(matplotlib not available -- skipping plot {path.name})",
              file=sys.stderr)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.config}\n{r.shape}" for r in rows]
    gen = [r.gen_ms for r in rows]
    ref = [r.ref_ms for r in rows]
    x = list(range(len(rows)))
    w = 0.4

    fig_w = max(8, 0.7 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar([xi - w / 2 for xi in x], gen, width=w, label="generated", color="#1f77b4")
    ax.bar([xi + w / 2 for xi in x], ref, width=w, label="reference",  color="#ff7f0e")
    ax.set_ylabel("ms / iter (median of 10)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate speedup on top of the gen bar.
    for xi, r in zip(x, rows):
        ax.text(xi - w / 2, r.gen_ms, f"{r.speedup:.2f}x",
                ha="center", va="bottom", fontsize=7, color="#1f77b4")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render(rows: list[BenchRow], bench_name: str,
           out_dir: Path = Path(__file__).resolve().parent / "results") -> None:
    """Print table + summary; also write CSV and PNG under out_dir/."""
    print("\n" + render_table(rows))
    print(render_summary(rows))
    if not rows:
        return
    csv_path = out_dir / f"{bench_name}.csv"
    png_path = out_dir / f"{bench_name}.png"
    save_csv(rows, csv_path)
    save_plot(rows, png_path, title=f"{bench_name}: generated vs nki-samples reference")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
