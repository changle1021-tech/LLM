#!/usr/bin/env python3
"""Single entry point for visualising ns-3 serving communication results.

Generates, from one `ns3/output/` directory, both the interactive HTML plots
(reusing `ns3-network-plots.py`) and a set of PNG figures covering:

  1. per-port throughput + utilization (HTML)
  2. GPU-pair communication matrix (PNG)
  3. flow-completion-time (FCT) analysis (PNG)
  4. link-load grouping: GPU-GPU direct vs switch fabric (PNG)

Usage:
    python3 scripts/comm-plots.py RESULTS_DIR/ns3/output [options]

PNG figures need matplotlib + numpy. The base interpreter may not have them;
run with an environment that does, e.g.

    conda run -n llm python scripts/comm-plots.py results/moe_ns3/ns3/output

The HTML part is pure stdlib and always runs. Rows with an unexpected field
count (which happen while a run is still writing its CSVs) are skipped.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

# ns-3 stores the serving output in ns; we display ms / microseconds.
NS_PER_MS = 1e6
NS_PER_US = 1e3
BYTES_PER_GB = 1e9

# --------------------------------------------------------------------------- #
# CSV helpers (robust against partially-written files)
# --------------------------------------------------------------------------- #


def _reader_limit():
    # qp_rate.csv can contain very wide rows; lift the default field cap.
    try:
        csv.field_size_limit(1 << 30)
    except OverflowError:
        pass


def iter_port_rows(path):
    """Yield dicts from port_bandwidth.csv, skipping malformed rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if len(row) != 10:
                continue
            try:
                yield {
                    "time_ns": int(row[0]),
                    "node": int(row[1]),
                    "port": int(row[2]),
                    "peer": int(row[3]),
                    "cap": int(row[4]),
                    "interval_ns": int(row[5]),
                    "tx_bytes": int(row[6]),
                    "rx_bytes": int(row[7]),
                    "tx_gbps": float(row[8]),
                    "rx_gbps": float(row[9]),
                }
            except (ValueError, OverflowError):
                continue


def iter_qp_rows(path):
    """Yield dicts from qp_rate.csv, skipping malformed rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if len(row) != 7:
                continue
            try:
                yield {
                    "time_ns": int(row[0]),
                    "src": int(row[1]),
                    "dst": int(row[2]),
                    "sport": int(row[3]),
                    "dport": int(row[4]),
                    "size": int(row[5]),
                    "rate": int(row[6]),
                }
            except (ValueError, OverflowError):
                continue


def iter_fct_rows(path):
    """Yield dicts from fct.txt (space separated), skipping malformed rows.

    Columns (entry.h): sip dip sport dport size(B) start(ns) fct(ns) standalone_fct(ns)
    Node id recovered with ip_to_node_id = (ip >> 8) & 0xffff.
    """
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 8:
                continue
            try:
                sip = int(parts[0], 16)
                dip = int(parts[1], 16)
                yield {
                    "src": (sip >> 8) & 0xFFFF,
                    "dst": (dip >> 8) & 0xFFFF,
                    "sport": int(parts[2]),
                    "dport": int(parts[3]),
                    "size": int(parts[4]),
                    "start_ns": int(parts[5]),
                    "fct_ns": int(parts[6]),
                    "ideal_ns": int(parts[7]),
                }
            except (ValueError, OverflowError):
                continue


def decimate(x, y, max_points=4000):
    """Evenly thin a time series for plotting while keeping the ends."""
    if len(x) <= max_points:
        return x, y
    n = len(x)
    step = (n - 1) / (max_points - 1)
    idx = sorted({0, n - 1} | {int(round(i * step)) for i in range(max_points)})
    return [x[i] for i in idx], [y[i] for i in idx]


# --------------------------------------------------------------------------- #
# Manifest / topology
# --------------------------------------------------------------------------- #


def load_manifest(input_dir):
    """Return (npu_count, switches) from network_manifest.json if present."""
    path = input_dir.parent / "network_manifest.json"
    if not path.exists():
        return None, []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    return manifest.get("npu_endpoints"), manifest.get("switches", [])


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #


def collect_ports(path, npu_count):
    """Read port_bandwidth.csv into per-port series plus group aggregates."""
    ports = {}  # (node, port) -> dict(peer, cap, t[], tx[], rx[])
    time_total = {}  # time_ns -> [sum tx_gbps, sum rx_gbps]
    group_total = {}  # group -> time_ns -> [sum tx_gbps, sum cap(bps)]

    def group_of(node, peer):
        if npu_count is not None and node < npu_count and peer < npu_count:
            return "gpu-gpu"
        return "fabric"

    for row in iter_port_rows(path):
        key = (row["node"], row["port"])
        entry = ports.get(key)
        if entry is None:
            entry = ports[key] = {
                "peer": row["peer"],
                "cap": row["cap"],
                "t": [],
                "tx": [],
                "rx": [],
            }
        if entry["t"] and row["time_ns"] < entry["t"][-1]:
            continue  # timestamps must be monotonic
        entry["t"].append(row["time_ns"])
        entry["tx"].append(row["tx_gbps"])
        entry["rx"].append(row["rx_gbps"])

        t = row["time_ns"]
        txrx = time_total.setdefault(t, [0.0, 0.0])
        txrx[0] += row["tx_gbps"]
        txrx[1] += row["rx_gbps"]

        grp = group_of(row["node"], row["peer"])
        pair = group_total.setdefault(grp, {}).setdefault(t, [0.0, 0.0])
        pair[0] += row["tx_gbps"]
        pair[1] += row["cap"]

    return ports, time_total, group_total


def collect_fct(path, npu_count):
    """Stream fct.txt; return byte matrix and FCT arrays."""
    matrix = [[0.0] * npu_count for _ in range(npu_count)]
    fct, ideal = [], []
    for row in iter_fct_rows(path):
        if row["src"] < npu_count and row["dst"] < npu_count:
            matrix[row["src"]][row["dst"]] += row["size"]
        fct.append(row["fct_ns"])
        ideal.append(row["ideal_ns"])
    return matrix, fct, ideal


# --------------------------------------------------------------------------- #
# HTML plots (reuse ns3-network-plots.py)
# --------------------------------------------------------------------------- #


def generate_html(input_dir, output_dir, max_points):
    script_path = Path(__file__).resolve().parent / "ns3-network-plots.py"
    spec = importlib.util.spec_from_file_location("ns3_network_plots", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = sys.argv
    sys.argv = [
        "ns3-network-plots.py",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--max-points",
        str(max_points),
    ]
    try:
        module.main()
    finally:
        sys.argv = saved


# --------------------------------------------------------------------------- #
# PNG figures
# --------------------------------------------------------------------------- #


def import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "matplotlib/numpy not available: run with an environment that has "
            "them (e.g. `conda run -n llm python scripts/comm-plots.py ...`)"
        ) from exc
    return matplotlib, plt, np


def _save(plt, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


def fig_gpu_pair_matrix(matrix, npu_count, out_dir):
    _, plt, np = import_mpl()
    mat = np.array(matrix, dtype=float) / BYTES_PER_GB  # GB
    labels = [str(i) for i in range(npu_count)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(npu_count), labels)
    ax.set_yticks(range(npu_count), labels)
    ax.set_xlabel("Destination GPU rank")
    ax.set_ylabel("Source GPU rank")
    ax.set_title("GPU-pair communication volume (GB, completed QPs)")
    peak = float(mat.max()) if mat.size else 0.0
    for i in range(npu_count):
        for j in range(npu_count):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}" if v > 0 else "0",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > peak * 0.4 else "black")
    fig.colorbar(im, ax=ax, label="GB")
    _save(plt, out_dir / "figures" / "02_gpu_pair_matrix.png")


def fig_fct(fct, ideal, out_dir):
    _, plt, np = import_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fct_pos = np.asarray([v for v in fct if v > 0], dtype=float)
    ideal_pos = np.asarray([v for v in ideal if v > 0], dtype=float)

    if fct_pos.size:
        xs = np.sort(fct_pos) / NS_PER_US
        ax1.plot(xs, np.linspace(0, 1, xs.size), label="measured")
    if ideal_pos.size:
        xs = np.sort(ideal_pos) / NS_PER_US
        ax1.plot(xs, np.linspace(0, 1, xs.size), label="ideal (standalone)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Flow completion time (us)")
    ax1.set_ylabel("CDF")
    ax1.set_title("FCT CDF: measured vs ideal")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    if fct_pos.size and ideal_pos.size:
        ax2.scatter(ideal_pos / NS_PER_US, fct_pos / NS_PER_US,
                    s=2, alpha=0.3)
        lims = [min(ideal_pos.min(), fct_pos.min()) / NS_PER_US * 0.9,
                max(ideal_pos.max(), fct_pos.max()) / NS_PER_US * 1.1]
        ax2.plot(lims, lims, "k--", linewidth=1, label="y = x (no slowdown)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Ideal / standalone FCT (us)")
    ax2.set_ylabel("Measured FCT (us)")
    ax2.set_title("Measured vs ideal FCT (per flow)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    _save(plt, out_dir / "figures" / "03_fct.png")


def fig_link_groups(group_total, out_dir):
    _, plt, np = import_mpl()
    names = {
        "gpu-gpu": "GPU-GPU direct (NVLink)",
        "fabric": "Switch fabric (400G)",
    }
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for grp, label in names.items():
        agg = group_total.get(grp)
        if not agg:
            continue
        times = sorted(agg)
        util = []
        for t in times:
            tx_gbps = agg[t][0]
            cap_gbps = agg[t][1] / 1e9  # bps -> Gbps
            util.append(tx_gbps / cap_gbps * 100 if cap_gbps else 0.0)
        tms, util = decimate([t / NS_PER_MS for t in times], util, 8000)
        ax.plot(tms, util, linewidth=1.2, label=f"{label} utilization")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Link utilization (%)")
    ax.set_title("Link utilization by group (TX / aggregate capacity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(plt, out_dir / "figures" / "05_link_groups.png")


def write_gallery(output_dir, png_files):
    """Small HTML page linking the PNGs alongside the interactive index."""
    imgs = "\n".join(
        f'<div class="card"><h2>{p.stem}</h2>'
        f'<img src="figures/{p.name}" style="max-width:100%"></div>'
        for p in png_files
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Communication plots</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#f5f7fb}"
        "h1{font-size:24px}.card{background:#fff;padding:16px;margin:16px 0;"
        "border:1px solid #dde3ee;border-radius:12px}</style></head><body>"
        f"<h1>Communication plots</h1>"
        f"<p><a href='index.html'>Interactive per-port / QP charts</a></p>{imgs}"
        "</body></html>"
    )
    (output_dir / "gallery.html").write_text(html, encoding="utf-8")
    print(f"  wrote {output_dir / 'gallery.html'}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML + PNG communication plots from an ns-3 output dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Directory containing port_bandwidth.csv / qp_rate.csv / fct.txt")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: INPUT_DIR/plots)")
    parser.add_argument("--max-points", type=int, default=2048,
                        help="Max displayed points per HTML series")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip the interactive HTML plots")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip the PNG figures")
    args = parser.parse_args()

    _reader_limit()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    port_path = input_dir / "port_bandwidth.csv"
    qp_path = input_dir / "qp_rate.csv"
    fct_path = input_dir / "fct.txt"

    npu_count, switches = load_manifest(input_dir)

    # Determine NPU count if the manifest is absent (peek at qp endpoints).
    if npu_count is None and qp_path.exists():
        seen = set()
        for row in iter_qp_rows(qp_path):
            seen.add(row["src"])
            seen.add(row["dst"])
            if len(seen) > 1024:
                break
        if seen:
            npu_count = max(seen) + 1

    print(f"NPU endpoints: {npu_count}, switches: {switches}")

    if not args.no_html:
        print("Generating interactive HTML plots ...")
        try:
            generate_html(input_dir, output_dir, args.max_points)
        except BaseException as exc:  # ns3-network-plots.py exits via parser.exit() on bad rows
            print(f"  warning: HTML plots skipped ({exc}); PNG figures still generated")

    if args.no_png:
        return

    # Collect data once, then render each PNG figure once.
    group_total = None
    if port_path.exists():
        _, _, group_total = collect_ports(port_path, npu_count)

    png_files = []
    if group_total:
        fig_link_groups(group_total, output_dir)
        png_files.append(Path("05_link_groups.png"))
    else:
        print("  port_bandwidth.csv missing; skipping link-group figure")

    if fct_path.exists() and npu_count:
        matrix, fct, ideal = collect_fct(fct_path, npu_count)
        print(f"  loaded {len(fct)} flows")
        fig_gpu_pair_matrix(matrix, npu_count, output_dir)
        png_files.append(Path("02_gpu_pair_matrix.png"))
        fig_fct(fct, ideal, output_dir)
        png_files.append(Path("03_fct.png"))
    else:
        print("  fct.txt missing (or NPU count unknown); skipping matrix/FCT figures")

    write_gallery(output_dir, sorted(png_files))
    print(f"Done. Open {output_dir / 'gallery.html'} or "
          f"{output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
