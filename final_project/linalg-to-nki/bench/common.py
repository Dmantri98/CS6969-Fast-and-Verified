"""
Shared benchmark utilities.

Timing model
------------
torch_xla kernel launches are asynchronous: the XLA client queues work and
control returns immediately. To get meaningful wall-clock numbers we:

  1. call the kernel,
  2. reference its output (materializes the graph),
  3. `xm.wait_device_ops(device)` to flush the queue,
  4. stop the timer.

`.cpu()` would also force a sync but adds an HBM->host copy we don't want
inside the loop.

Each configuration is warmed up (N_WARMUP runs — the first one triggers the
NKI compile, which can be multi-second) and then measured N_MEASURE times;
the returned ms/iter is the median, which is robust to the occasional GC
hiccup or DMA stall.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch
from torch_xla.core import xla_model as xm


N_WARMUP = 3
N_MEASURE = 10


@dataclass
class BenchResult:
    label: str
    ms_median: float
    ms_p10: float
    ms_p90: float
    max_abs_err: float
    ok: bool


def _sync(device):
    # wait_device_ops expects a Sequence[str] or nothing (all devices).
    # Passing no args is safe and matches "flush everything queued so far".
    xm.wait_device_ops()


def time_kernel(kernel, args, device) -> tuple[float, float, float]:
    """Return (median_ms, p10_ms, p90_ms) over N_MEASURE runs after warmup."""
    for _ in range(N_WARMUP):
        out = kernel(*args)
        _ = out.shape  # touch to materialize
    _sync(device)

    samples_ms: list[float] = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        out = kernel(*args)
        _ = out.shape
        _sync(device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1e3)

    samples_ms.sort()
    median = statistics.median(samples_ms)
    # Clamp indices so degenerate-small N still works.
    p10 = samples_ms[max(0, len(samples_ms) // 10)]
    p90 = samples_ms[min(len(samples_ms) - 1, (9 * len(samples_ms)) // 10)]
    return median, p10, p90


def compare(label: str, kernel, args, reference_out, device,
            atol=1e-4, rtol=1e-2) -> BenchResult:
    """Run `kernel(*args)`, compare to `reference_out`, and time it."""
    nki_out = kernel(*args)
    max_abs = (nki_out - reference_out).abs().max().item()
    ok = torch.allclose(nki_out, reference_out, atol=atol, rtol=rtol)

    median, p10, p90 = time_kernel(kernel, args, device)
    return BenchResult(
        label=label, ms_median=median, ms_p10=p10, ms_p90=p90,
        max_abs_err=max_abs, ok=ok,
    )


def print_header(cols: list[str]):
    print("  ".join(cols))
    print("  ".join("-" * len(c) for c in cols))


def fmt_row(shape_tag: str, gen: BenchResult, ref: BenchResult) -> str:
    speedup = ref.ms_median / gen.ms_median if gen.ms_median > 0 else float("nan")
    gen_mark = "OK " if gen.ok else "FAIL"
    ref_mark = "OK " if ref.ok else "FAIL"
    return (
        f"{shape_tag:<32}  "
        f"gen={gen.ms_median:>7.3f}ms [{gen_mark} err={gen.max_abs_err:.2e}]  "
        f"ref={ref.ms_median:>7.3f}ms [{ref_mark} err={ref.max_abs_err:.2e}]  "
        f"speedup={speedup:>5.2f}x"
    )
