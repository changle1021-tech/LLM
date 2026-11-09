#!/usr/bin/env python3
"""Additional communication plots — 9 figures (entry point #2).

Complements `comm-plots.py`. One ns-3 `output/` directory plus optional serving
/ comparison inputs produce a numbered set of PNG figures:

  01  whole-fabric port x time utilization heatmap
  03  FCT slowdown vs message size and vs flow start time
  05  message-size spectrum
  06  GPU-pair matrix over time (small multiples)
  07  cumulative bytes over time (per group / per link)
  08  direct vs switch-fabric traffic split
  09  per-link traffic ranking
  10  request TTFT/TPOT over time                                   (--serving-csv)
  14  serving latency / TTFT comparison                             (--cmp-serving-csv)

Requires matplotlib + numpy:

    conda run -n llm python scripts/comm-plots-extra.py results/moe_ns3/ns3/output \
        --serving-csv results/moe_ns3.csv \
        --cmp-serving-csv results/moe_an.csv \
        --label-self moe_ns3 --label-cmp-serving analytical

Rows with an unexpected field count (an in-progress run) are skipped.
"""

import argparse
import csv
import importlib.util
import math
from pathlib import Path

# --- reuse helpers from comm-plots.py ------------------------------------ #


def _load_base():
    path = Path(__file__).resolve().parent / "comm-plots.py"
    spec = importlib.util.spec_from_file_location("comm_plots", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()

iter_port_rows = _base.iter_port_rows
iter_qp_rows = _base.iter_qp_rows
iter_fct_rows = _base.iter_fct_rows
decimate = _base.decimate
load_manifest = _base.load_manifest
collect_ports = _base.collect_ports
import_mpl = _base.import_mpl
_save = _base._save
NS_PER_MS = _base.NS_PER_MS
BYTES_PER_GB = _base.BYTES_PER_GB


def group_of(node, peer, npu_count):
    if npu_count is not None and node < npu_count and peer < npu_count:
        return "gpu-gpu"
    return "fabric"


def load_serving(path):
    arrivals, latencies, ttfts, tpots = [], [], [], []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            try:
                arrivals.append(float(row["arrival"]) / NS_PER_MS)
                latencies.append(float(row["latency"]) / NS_PER_MS)
                ttfts.append(float(row["TTFT"]) / NS_PER_MS)
                tpots.append(float(row["TPOT"]) / NS_PER_MS)
            except (KeyError, ValueError):
                continue
    return arrivals, latencies, ttfts, tpots


def collect_byte_data(path, npu_count):
    """Read port_bandwidth.csv for byte-based accounting.

    Returns (per_port, group_time, group_total):
      per_port[(node,port)] -> {peer, cap, t[], cum_tx[], total_tx, total_rx}
      group_time[grp] -> {t: bytes}
      group_total[grp] -> total bytes
    """
    per_port = {}
    group_time = {}
    group_total = {}
    for row in iter_port_rows(path):
        key = (row["node"], row["port"])
        p = per_port.get(key)
        if p is None:
            p = per_port[key] = {
                "peer": row["peer"], "cap": row["cap"],
                "t": [], "cum": [], "total_tx": 0, "total_rx": 0,
            }
        if p["t"] and row["time_ns"] <= p["t"][-1]:
            continue  # skip out-of-order / duplicate rows
        p["total_tx"] += row["tx_bytes"]
        p["total_rx"] += row["rx_bytes"]
        p["t"].append(row["time_ns"])
        p["cum"].append(p["total_tx"])

        grp = group_of(row["node"], row["peer"], npu_count)
        gt = group_time.setdefault(grp, {})
        gt[row["time_ns"]] = gt.get(row["time_ns"], 0.0) + row["tx_bytes"]
        group_total[grp] = group_total.get(grp, 0.0) + row["tx_bytes"]
    return per_port, group_time, group_total


# --------------------------------------------------------------------------- #
# Network figures
# --------------------------------------------------------------------------- #


def fig_01_util_heatmap(ports, out_dir):
    _, plt, np = import_mpl()
    all_times = sorted({t for e in ports.values() for t in e["t"]})
    tms = [t / NS_PER_MS for t in all_times]
    tms, _ = decimate(tms, list(range(len(all_times))), 400)
    keep = [int(t * NS_PER_MS) for t in tms]

    labels, matrix = [], []
    for key in sorted(ports):
        node, port_id = key
        e = ports[key]
        cap_gbps = e["cap"] / 1e9
        util = {t: tx / cap_gbps * 100 if cap_gbps else 0.0
                for t, tx in zip(e["t"], e["tx"])}
        labels.append(f"{node}.{port_id}->{e['peer']}")
        matrix.append([util.get(t, 0.0) for t in keep])

    mat = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(13, max(4, len(labels) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="inferno", vmin=0)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    step = max(1, len(tms) // 8)
    ax.set_xticks(list(range(0, len(tms), step)))
    ax.set_xticklabels([f"{tms[i]:.0f}" for i in range(0, len(tms), step)])
    ax.set_xlabel("Simulation time (ms)")
    ax.set_title("01  Per-port TX utilization heatmap (%)")
    fig.colorbar(im, ax=ax, label="Utilization (%)")
    _save(plt, out_dir / "figures" / "01_util_heatmap.png")


def fig_03_slowdown_vs(fct, ideal, sizes, starts, out_dir):
    _, plt, np = import_mpl()
    f = np.asarray(fct, dtype=float)
    i = np.asarray(ideal, dtype=float)
    sz = np.asarray(sizes, dtype=float)
    st = np.asarray(starts, dtype=float)
    m = (f > 0) & (i > 0)
    slow = f[m] / i[m]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.scatter(sz[m], slow, s=3, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Message size (B)")
    ax1.set_ylabel("Slowdown")
    ax1.set_title("03a  Slowdown vs size")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.scatter(st[m] / NS_PER_MS, slow, s=3, alpha=0.3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Flow start time (ms)")
    ax2.set_ylabel("Slowdown")
    ax2.set_title("03b  Slowdown vs start time")
    ax2.grid(True, which="both", alpha=0.3)
    _save(plt, out_dir / "figures" / "03_fct_slowdown_vs.png")


def fig_05_msg_size(sizes, out_dir):
    _, plt, np = import_mpl()
    arr = np.asarray(sizes, dtype=float)
    arr = arr[arr > 0]
    bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 40)
    counts, edges = np.histogram(arr, bins=bins)
    widths = np.diff(edges)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.hist(arr, bins=bins, color="#2563eb")
    ax1.set_yscale("log")
    ax1.set_ylabel("Flow count")
    ax1.set_title("05  Message-size spectrum")
    ax2.bar(edges[:-1], counts * edges[:-1], width=widths, color="#e76f22")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Message size (B)")
    ax2.set_ylabel("Total bytes per bin")
    ax1.grid(True, which="both", alpha=0.3)
    ax2.grid(True, which="both", alpha=0.3)
    _save(plt, out_dir / "figures" / "05_msg_size_hist.png")


def fig_06_matrix_frames(fct_path, npu_count, out_dir, frames=8):
    _, plt, np = import_mpl()
    flows, tmin, tmax = [], None, None
    for row in iter_fct_rows(fct_path):
        if row["src"] < npu_count and row["dst"] < npu_count:
            flows.append((row["start_ns"], row["src"], row["dst"], row["size"]))
            tmin = row["start_ns"] if tmin is None else min(tmin, row["start_ns"])
            tmax = row["start_ns"] if tmax is None else max(tmax, row["start_ns"])
    if not flows or tmin == tmax:
        print("  06 skipped: no NPU-pair flows")
        return
    edges = np.linspace(tmin, tmax, frames + 1)
    mats = [np.zeros((npu_count, npu_count)) for _ in range(frames)]
    for t, src, dst, size in flows:
        idx = min(frames - 1, int(np.searchsorted(edges, t, side="right") - 1))
        mats[idx][src, dst] += size
    vmax = max(m.max() for m in mats) / BYTES_PER_GB
    cols = 4
    rows = math.ceil(frames / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.1))
    axes = np.atleast_1d(axes).reshape(-1)
    for i in range(frames):
        ax = axes[i]
        im = ax.imshow(mats[i] / BYTES_PER_GB, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{(edges[i] / NS_PER_MS):.0f}-{(edges[i + 1] / NS_PER_MS):.0f} ms",
                     fontsize=8)
        ax.set_xticks(range(npu_count), [str(j) for j in range(npu_count)], fontsize=7)
        ax.set_yticks(range(npu_count), [str(j) for j in range(npu_count)], fontsize=7)
    for i in range(frames, len(axes)):
        axes[i].axis("off")
    fig.suptitle("06  GPU-pair communication volume (GB) over time")
    fig.tight_layout()
    _save(plt, out_dir / "figures" / "06_pair_matrix_frames.png")


def fig_07_cum_bytes(per_port, group_time, npu_count, out_dir):
    _, plt, np = import_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for grp, label, color in (("gpu-gpu", "NVLink direct", "#2563eb"),
                              ("fabric", "fabric", "#e76f22")):
        gt = group_time.get(grp)
        if not gt:
            continue
        times = sorted(gt)
        cum, run = [], 0.0
        for t in times:
            run += gt[t]
            cum.append(run / BYTES_PER_GB)
        ax1.plot([t / NS_PER_MS for t in times], cum, label=label, color=color)
    ax1.set_xlabel("Simulation time (ms)")
    ax1.set_ylabel("Cumulative TX (GB)")
    ax1.set_title("07a  Cumulative bytes by group")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    direct, fabric = "#2563eb", "#e76f22"
    ends = []
    for key in sorted(per_port):
        p = per_port[key]
        if not p["t"]:
            continue
        is_direct = group_of(key[0], p["peer"], npu_count) == "gpu-gpu"
        xs = [t / NS_PER_MS for t in p["t"]]
        ys = [v / BYTES_PER_GB for v in p["cum"]]
        # Fabric is drawn first: per-hop counting makes every NVLink-direct
        # curve coincide with a fabric twin, so direct lines need the top layer.
        ax2.plot(xs, ys, color=direct if is_direct else fabric,
                 linewidth=1.6 if is_direct else 1.0,
                 alpha=0.95 if is_direct else 0.65,
                 zorder=3 if is_direct else 1)
        ends.append((xs[-1], ys[-1], f"{key[0]}.{key[1]}->{p['peer']}",
                     is_direct))

    ax2.plot([], [], color=direct, linewidth=1.6, label="NVLink direct")
    ax2.plot([], [], color=fabric, linewidth=1.0, label="switch fabric")
    ax2.legend(loc="upper left")
    ax2.set_xlabel("Simulation time (ms)")
    ax2.set_ylabel("Cumulative TX (GB)")
    ax2.set_title("07b  Cumulative bytes per link")
    ax2.grid(True, alpha=0.3)

    # Coincident curves collapse into value bands, so annotate one grouped
    # label per band (NVLink members in blue, fabric members in orange).
    x_end = max(e[0] for e in ends)
    y_max = max(e[1] for e in ends) or 1.0
    ax2.set_xlim(right=x_end * 1.45)
    ax2.set_ylim(top=y_max * 1.12)
    bands = []
    for e in sorted(ends, key=lambda e: e[1]):
        if bands and abs(e[1] - bands[-1][-1][1]) <= y_max * 0.03:
            bands[-1].append(e)
        else:
            bands.append([e])
    label_x = x_end * 1.02
    text_kw = dict(fontsize=6, va="center", ha="left", clip_on=False,
                   bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                             edgecolor="none", alpha=0.75))
    for band in bands:
        y_mid = sum(b[1] for b in band) / len(band)
        lo, hi = band[0][1], band[-1][1]
        value = f"{lo:.1f} GB" if hi - lo < 0.5 else f"{lo:.1f}-{hi:.1f} GB"
        # Stack lines top-to-bottom, NVLink members above fabric ones; wrap
        # into rows of two so long bands stay inside the reserved margin.
        lines = []
        for is_direct in (True, False):
            members = [b[2] for b in band if b[3] == is_direct]
            for i in range(0, len(members), 2):
                lines.append((is_direct, ", ".join(members[i:i + 2])))
        dy = y_max * 0.024
        y0 = y_mid + (len(lines) - 1) * dy / 2
        for i, (is_direct, text) in enumerate(lines):
            if i == 0:
                text = f"{value}  {text}"
            ax2.text(label_x, y0 - i * dy, text,
                     color=direct if is_direct else "#e76f22", **text_kw)
    _save(plt, out_dir / "figures" / "07_cum_bytes.png")


def fig_08_traffic_split(per_port, npu_count, out_dir):
    _, plt, np = import_mpl()
    direct = fabric = 0.0
    for (node, _), p in per_port.items():
        total = p["total_tx"] / BYTES_PER_GB
        if group_of(node, p["peer"], npu_count) == "gpu-gpu":
            direct += total
        else:
            fabric += total
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie([direct, fabric], labels=["NVLink direct", "Switch fabric"],
           autopct="%1.1f%%", startangle=90, colors=["#2563eb", "#e76f22"])
    ax.set_title("08  TX volume split")
    ax.text(0, -1.25, f"direct={direct:.2f} GB   fabric={fabric:.2f} GB",
            ha="center", va="center", fontsize=9)
    _save(plt, out_dir / "figures" / "08_traffic_split.png")


def fig_09_port_rank(per_port, npu_count, out_dir):
    _, plt, np = import_mpl()
    rows = sorted(((p["total_tx"] / BYTES_PER_GB, node, port, p["peer"],
                    group_of(node, p["peer"], npu_count))
                   for (node, port), p in per_port.items()), reverse=True)
    labels = [f"{r[1]}.{r[2]}->{r[3]}" for r in rows]
    vals = [r[0] for r in rows]
    cols = ["#2563eb" if r[4] == "gpu-gpu" else "#e76f22" for r in rows]
    fig, ax = plt.subplots(figsize=(7, max(4, len(vals) * 0.35)))
    ax.barh(range(len(vals)), vals, color=cols)
    ax.set_yticks(range(len(vals)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cumulative TX (GB)")
    ax.set_title("09  Per-link traffic ranking")
    ax.grid(True, axis="x", alpha=0.3)
    _save(plt, out_dir / "figures" / "09_port_rank.png")


# --------------------------------------------------------------------------- #
# Serving figure
# --------------------------------------------------------------------------- #


def fig_10_ttft_tpot(serving, out_dir):
    _, plt, np = import_mpl()
    arrivals, _, ttfts, tpots = serving
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(arrivals, ttfts, s=6, alpha=0.6, label="TTFT")
    ax.scatter(arrivals, tpots, s=6, alpha=0.6, label="TPOT")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("10  Request TTFT/TPOT over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(plt, out_dir / "figures" / "10_ttft_tpot.png")


# --------------------------------------------------------------------------- #
# Serving comparison figure
# --------------------------------------------------------------------------- #


def fig_14_serving_compare(serving_self, serving_other, label_self, label_other, out_dir):
    _, plt, np = import_mpl()
    _, lat_self, ttft_self, _ = serving_self
    _, lat_other, ttft_other, _ = serving_other
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for arr, label in ((lat_self, label_self), (lat_other, label_other)):
        a = np.sort(np.asarray([v for v in arr if v > 0], dtype=float))
        if a.size:
            ax1.plot(a, np.linspace(0, 1, a.size), label=label)
    ax1.set_xlabel("End-to-end latency (ms)")
    ax1.set_ylabel("CDF")
    ax1.set_title("14  Latency CDF")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for arr, label in ((ttft_self, label_self), (ttft_other, label_other)):
        a = np.sort(np.asarray([v for v in arr if v > 0], dtype=float))
        if a.size:
            ax2.plot(a, np.linspace(0, 1, a.size), label=label)
    ax2.set_xlabel("TTFT (ms)")
    ax2.set_ylabel("CDF")
    ax2.set_title("TTFT CDF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _save(plt, out_dir / "figures" / "14_serving_compare.png")


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #

DESCRIPTIONS = {
    "01_util_heatmap.png": "全链路端口×时间利用率热力图",
    "03_fct_slowdown_vs.png": "慢化比 vs 消息大小 / vs 开始时间",
    "05_msg_size_hist.png": "消息大小谱",
    "06_pair_matrix_frames.png": "GPU 对通信矩阵随时间演化",
    "07_cum_bytes.png": "累计字节随时间(分组/每链路)",
    "08_traffic_split.png": "直连 vs 交换 fabric 流量占比",
    "09_port_rank.png": "每链路流量排行",
    "10_ttft_tpot.png": "请求 TTFT/TPOT 随时间",
    "14_serving_compare.png": "latency/TTFT 对比",
}


def write_gallery(output_dir, produced):
    cards = []
    for name in sorted(DESCRIPTIONS):
        if name not in produced:
            continue
        cards.append(
            f'<div class="card"><h2>{name}</h2>'
            f'<p>{DESCRIPTIONS[name]}</p>'
            f'<img src="figures/{name}" style="max-width:100%"></div>'
        )
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Communication plots (extra)</title>"
            "<style>body{font-family:system-ui;margin:24px;background:#f5f7fb}"
            "h1{font-size:24px}.card{background:#fff;padding:16px;margin:16px 0;"
            "border:1px solid #dde3ee;border-radius:12px}img{border:1px solid #eee}"
            "</style></head><body><h1>Communication plots (extra)</h1>" +
            "\n".join(cards) + "</body></html>")
    (output_dir / "gallery_extra.html").write_text(html, encoding="utf-8")
    print(f"  wrote {output_dir / 'gallery_extra.html'}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Additional communication plots (WIP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", type=Path,
                        help="ns3 output dir (port_bandwidth.csv / qp_rate.csv / fct.txt)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: INPUT_DIR/plots)")
    parser.add_argument("--serving-csv", type=Path, default=None,
                        help="Serving metrics CSV for figure 10")
    parser.add_argument("--cmp-serving-csv", type=Path, default=None,
                        help="Second serving CSV for figure 14")
    parser.add_argument("--label-self", type=str, default="self")
    parser.add_argument("--label-cmp-serving", type=str, default="other",
                        help="Legend label for the serving comparison (figure 14)")
    args = parser.parse_args()

    _base._reader_limit()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "plots"
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    npu_count, switches = load_manifest(input_dir)
    if npu_count is None:
        seen = set()
        for row in iter_qp_rows(input_dir / "qp_rate.csv"):
            seen.add(row["src"])
            seen.add(row["dst"])
            if len(seen) > 1024:
                break
        if seen:
            npu_count = max(seen) + 1
    print(f"NPU endpoints: {npu_count}, switches: {switches}")

    port_path = input_dir / "port_bandwidth.csv"
    fct_path = input_dir / "fct.txt"

    ports = None
    if port_path.exists():
        ports, _, _ = collect_ports(port_path, npu_count)
        print(f"  loaded {len(ports)} port series")
    per_port, group_time, group_total = None, None, None
    if port_path.exists():
        per_port, group_time, group_total = collect_byte_data(port_path, npu_count)

    fct, ideal, sizes, starts = [], [], [], []
    if fct_path.exists():
        for row in iter_fct_rows(fct_path):
            fct.append(row["fct_ns"])
            ideal.append(row["ideal_ns"])
            sizes.append(row["size"])
            starts.append(row["start_ns"])
        print(f"  loaded {len(fct)} flows")

    produced = set()

    if ports:
        fig_01_util_heatmap(ports, output_dir); produced.add("01_util_heatmap.png")
    if per_port:
        fig_07_cum_bytes(per_port, group_time, npu_count, output_dir); produced.add("07_cum_bytes.png")
        fig_08_traffic_split(per_port, npu_count, output_dir); produced.add("08_traffic_split.png")
        fig_09_port_rank(per_port, npu_count, output_dir); produced.add("09_port_rank.png")
    if fct:
        fig_03_slowdown_vs(fct, ideal, sizes, starts, output_dir); produced.add("03_fct_slowdown_vs.png")
        fig_05_msg_size(sizes, output_dir); produced.add("05_msg_size_hist.png")
        if npu_count:
            fig_06_matrix_frames(fct_path, npu_count, output_dir); produced.add("06_pair_matrix_frames.png")

    serving = None
    if args.serving_csv and args.serving_csv.exists():
        serving = load_serving(args.serving_csv)
        print(f"  loaded serving csv: {len(serving[0])} requests")
    if serving:
        fig_10_ttft_tpot(serving, output_dir); produced.add("10_ttft_tpot.png")

    if args.cmp_serving_csv and args.cmp_serving_csv.exists() and serving:
        cmp_serving = load_serving(args.cmp_serving_csv)
        fig_14_serving_compare(serving, cmp_serving,
                               args.label_self, args.label_cmp_serving, output_dir)
        produced.add("14_serving_compare.png")

    write_gallery(output_dir, produced)
    missing = [n for n in sorted(DESCRIPTIONS) if n not in produced]
    print(f"Generated {len(produced)}/{len(DESCRIPTIONS)} figures in {output_dir / 'figures'}")
    if missing:
        print("Skipped (missing inputs): " + ", ".join(missing))


if __name__ == "__main__":
    main()
