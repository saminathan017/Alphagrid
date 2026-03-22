"""
scripts/monitor_training.py  —  AlphaGrid Training Monitor
============================================================
Live progress display for train_models.py runs.

Usage:
    python scripts/monitor_training.py                  # auto-detect latest log
    python scripts/monitor_training.py logs/training_X.log
    python scripts/monitor_training.py --once            # print once and exit
"""
from __future__ import annotations

import re
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich import box

# ── Config ────────────────────────────────────────────────────────────────────

TOTAL_SYMBOLS = 150   # US_SYMBOLS[:100] + FOREX_SYMBOLS[:50]

KEY_SYMBOLS = {          # symbol → position index (0-based) in training order
    "SOXL":     98,
    "TQQQ":     99,
    "AUDUSD=X": 104,
    "USDTRY=X": 133,
}

HIT_THRESHOLD = 0.80
REFRESH_SECS  = 5

# ── Regexes ───────────────────────────────────────────────────────────────────

RE_TRAINING  = re.compile(r"TRAINING: (\S+)")
RE_RESULTS   = re.compile(r"RESULTS — (\S+) \| elapsed=([\d.]+)s")
RE_MODEL_ROW = re.compile(
    r"(QuantLSTM|Transformer|LightGBM|MetaEnsemble)\s+"
    r"\[([A-Z])\]\s+"
    r"([\d.]+)\s+"    # acc
    r"([\d.]+)\s+"    # f1
    r"([\d.]+)\s+"    # auc
    r"(\S+)\s+"       # ic
    r"(\S+)\s+"       # icir
    r"([\d.]+)\s+"    # hit@70
    r"([\d.]+)\s+"    # hit@80
    r"([\d.]+)"       # hit@90
)
RE_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_FINAL     = re.compile(r"FINAL SUMMARY")

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(path: Path, pre_done: int = 0, total_override: int | None = None) -> dict:
    completed: list[dict] = []
    current_symbol: str | None = None   # symbol currently being trained
    pending_entry:  dict | None = None  # entry awaiting model rows (post-RESULTS)
    elapsed_times: list[float] = []
    last_ts: datetime | None = None
    final_seen = False

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = RE_TIMESTAMP.match(line)
        if m:
            try:
                last_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        # RESULTS — SYMBOL | elapsed=Xs  (comes BEFORE the model rows in the log)
        m = RE_RESULTS.search(line)
        if m:
            sym, elapsed = m.group(1), float(m.group(2))
            elapsed_times.append(elapsed)
            pending_entry = {"symbol": sym, "elapsed": elapsed, "models": {}}
            completed.append(pending_entry)
            current_symbol = None   # symbol is done training
            continue

        # Model row (QuantLSTM / Transformer / LightGBM / MetaEnsemble)
        m = RE_MODEL_ROW.search(line)
        if m:
            entry = pending_entry  # attach to last RESULTS entry
            if entry is not None:
                entry["models"][m.group(1)] = {
                    "tier":  m.group(2),
                    "acc":   float(m.group(3)),
                    "f1":    float(m.group(4)),
                    "auc":   float(m.group(5)),
                    "hit70": float(m.group(8)),
                    "hit80": float(m.group(9)),
                    "hit90": float(m.group(10)),
                }
            continue

        # TRAINING: SYMBOL — new symbol starting; close pending entry
        m = RE_TRAINING.search(line)
        if m:
            current_symbol = m.group(1)
            pending_entry  = None   # model rows now belong to next symbol
            continue

        if RE_FINAL.search(line):
            final_seen = True

    # Deduplicate: if a symbol was trained twice (restart), keep the latest result
    seen: dict[str, dict] = {}
    for entry in completed:
        seen[entry["symbol"]] = entry
    completed = list(seen.values())
    elapsed_times = [e["elapsed"] for e in completed]

    total = total_override if total_override else (pre_done + len(completed))

    return {
        "completed":     completed,
        "pre_done":      pre_done,
        "total":         total,
        "current":       current_symbol,
        "elapsed_times": elapsed_times,
        "last_ts":       last_ts,
        "done":          final_seen,
    }


def find_latest_log() -> Path | None:
    logs = sorted(Path("logs").glob("training_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None

# ── Colors ────────────────────────────────────────────────────────────────────

def tier_color(t: str) -> str:
    return {"S": "bold green", "A": "green", "B": "yellow", "C": "dark_orange", "D": "red"}.get(t, "white")

def hit_color(v: float) -> str:
    if v >= 0.90: return "bold green"
    if v >= 0.80: return "green"
    if v >= 0.60: return "yellow"
    return "dim"

# ── Build display ─────────────────────────────────────────────────────────────

def build_display(data: dict) -> Group:
    completed     = data["completed"]
    current       = data["current"]
    elapsed_times = data["elapsed_times"]
    last_ts       = data["last_ts"]
    n_done        = len(completed) + data.get("pre_done", 0)
    total         = data.get("total", n_done)

    avg_sec   = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 120.0
    remaining = max(0, total - n_done)
    eta_sec   = int(remaining * avg_sec)
    eta_str   = str(timedelta(seconds=eta_sec))

    # ── Progress bar ─────────────────────────────────────────────────────────
    bar_w  = 44
    filled = int(bar_w * n_done / total) if total else 0
    bar    = "█" * filled + "░" * (bar_w - filled)
    pct    = 100.0 * n_done / total if total else 0

    done_set   = {r["symbol"] for r in completed}
    # all symbols being trained in this run (order matters for ETA)
    run_syms   = [r["symbol"] for r in completed]
    key_parts: list[str] = []
    for sym, idx in sorted(KEY_SYMBOLS.items(), key=lambda x: x[1]):
        if sym in done_set:
            key_parts.append(f"[green]{sym} ✓[/green]")
        else:
            # how many undone symbols appear before this key symbol in the run order
            try:
                pos = run_syms.index(sym)
                steps_away = 0  # already in list means it's done — shouldn't reach here
            except ValueError:
                steps_away = max(0, idx - n_done)
            secs_away = steps_away * avg_sec
            key_parts.append(f"[yellow]{sym}[/yellow] ~{timedelta(seconds=int(secs_away))}")

    age_str = ""
    if last_ts:
        age = int((datetime.now() - last_ts).total_seconds())
        age_str = f"  [dim]log {age}s ago[/dim]"

    now_str = f"\n  [dim]Now training:[/dim] [bold white]{current}[/bold white]" if current else ""

    progress_panel = Panel(
        Text.from_markup(
            f"  [bold cyan]{n_done} / {total}[/bold cyan]  "
            f"[white]{bar}[/white]  "
            f"[bold]{pct:.1f}%[/bold]\n"
            f"  ETA [bold yellow]{eta_str}[/bold yellow]  "
            f"avg [dim]{avg_sec:.0f}s/sym[/dim]"
            f"{age_str}"
            f"{now_str}"
        ),
        title="[bold]AlphaGrid v6 — Training Progress[/bold]",
        subtitle="  ".join(key_parts),
        border_style="cyan",
        padding=(0, 1),
    )

    # ── Stars: genuine hit@70 ≥ 0.80 (exclude degenerate LSTM) ─────────────
    stars: list[str] = []
    for entry in completed:
        for mname, m in entry["models"].items():
            is_degenerate_lstm = (mname == "QuantLSTM" and m["auc"] < 0.52)
            if m["hit70"] >= HIT_THRESHOLD and not is_degenerate_lstm:
                stars.append(
                    f"[bold cyan]{entry['symbol']:12s}[/bold cyan] "
                    f"[dim]{mname:14s}[/dim] "
                    f"tier=[{tier_color(m['tier'])}]{m['tier']}[/{tier_color(m['tier'])}]  "
                    f"acc=[bold]{m['acc']:.3f}[/bold]  "
                    f"hit@70=[green]{m['hit70']:.3f}[/green]  "
                    f"hit@80=[{'green' if m['hit80']>=0.8 else 'yellow'}]{m['hit80']:.3f}[/{'green' if m['hit80']>=0.8 else 'yellow'}]  "
                    f"hit@90=[{'green' if m['hit90']>=0.8 else 'yellow'}]{m['hit90']:.3f}[/{'green' if m['hit90']>=0.8 else 'yellow'}]"
                )

    star_panel = None
    if stars:
        star_panel = Panel(
            Text.from_markup("\n".join(f"  ★  {s}" for s in stars)),
            title=f"[bold green]High-Confidence Signals — hit@70 ≥ {HIT_THRESHOLD:.0%} (genuine, non-degenerate)[/bold green]",
            border_style="green",
        )

    # ── Results table (last 12 symbols) ──────────────────────────────────────
    tbl = Table(
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_edge=False,
        pad_edge=True,
        min_width=110,
    )
    tbl.add_column("Symbol",      min_width=12, no_wrap=True)
    tbl.add_column("Time",        min_width=6,  justify="right", no_wrap=True)
    tbl.add_column("Model",       min_width=14, no_wrap=True)
    tbl.add_column("Tier",        min_width=4,  justify="center", no_wrap=True)
    tbl.add_column("Acc",         min_width=6,  justify="right", no_wrap=True)
    tbl.add_column("AUC",         min_width=6,  justify="right", no_wrap=True)
    tbl.add_column("Hit@70",      min_width=7,  justify="right", no_wrap=True)
    tbl.add_column("Hit@80",      min_width=7,  justify="right", no_wrap=True)
    tbl.add_column("Hit@90",      min_width=7,  justify="right", no_wrap=True)

    for entry in reversed(completed[-12:]):
        sym  = entry["symbol"]
        first = True
        for mname, m in entry["models"].items():
            tbl.add_row(
                f"[bold]{sym}[/bold]" if first else "",
                f"{entry['elapsed']:.0f}s"          if first else "",
                mname,
                Text(m["tier"], style=tier_color(m["tier"])),
                f"{m['acc']:.3f}",
                f"{m['auc']:.3f}",
                Text(f"{m['hit70']:.3f}", style=hit_color(m["hit70"])),
                Text(f"{m['hit80']:.3f}", style=hit_color(m["hit80"])),
                Text(f"{m['hit90']:.3f}", style=hit_color(m["hit90"])),
            )
            first = False
        tbl.add_section()

    results_panel = Panel(
        tbl,
        title="[bold]Recent Results[/bold] (last 12 symbols, newest first)",
        border_style="blue",
    )

    renderables: list = [progress_panel]
    if star_panel:
        renderables.append(star_panel)
    renderables.append(results_panel)
    return Group(*renderables)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monitor AlphaGrid training")
    parser.add_argument("log",   nargs="?", help="Log file (default: latest in logs/)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--pre-done", type=int, default=0,
                        help="Symbols already completed before this log (adds to count)")
    parser.add_argument("--total", type=int, default=0,
                        help="Override total symbol count (default: auto from log)")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else find_latest_log()
    if not log_path or not log_path.exists():
        print("No training log found. Pass the log path explicitly.")
        sys.exit(1)

    console = Console(width=160)

    pre_done = args.pre_done
    total_override = args.total if args.total > 0 else None

    if args.once:
        data = parse_log(log_path, pre_done, total_override)
        console.print(build_display(data))
        return

    console.print(f"[dim]Monitoring:[/dim] {log_path}  [dim](Ctrl-C to exit)[/dim]\n")

    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            try:
                data   = parse_log(log_path, pre_done, total_override)
                live.update(build_display(data))
                if data["done"]:
                    console.print("\n[bold green]✓ Training complete![/bold green]")
                    break
                time.sleep(REFRESH_SECS)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
