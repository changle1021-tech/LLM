#!/usr/bin/env python3
"""Convert this project's little-endian ns-3 mix.tr packet log to TSV text."""

import argparse
from functools import lru_cache
from ipaddress import IPv4Address
import os
from pathlib import Path
import struct


# SimSetting writes a port count, packed (node, interface, rate) entries,
# and a window value. TraceFormat then writes native x86-64 records.
PORT_COUNT = struct.Struct("<I")
PORT = struct.Struct("<HBQ")
WINDOW = struct.Struct("<I")
RECORD = struct.Struct("<QHBBIIIHBBBB2x24s")
DATA = struct.Struct("<HHIQHH")
ACK = struct.Struct("<HHHHI4xQ")
PFC = struct.Struct("<IIB")
CNP = struct.Struct("<HBBHH")
PORTS = struct.Struct("<HH")
EVENTS = {0: "recv", 1: "enqueue", 2: "dequeue", 3: "drop"}
PROTOCOLS = {0x06: "TCP", 0x11: "UDP", 0xFC: "ACK", 0xFD: "NACK", 0xFE: "PFC", 0xFF: "CNP"}
COLUMNS = (
    "time_ns", "node", "interface", "event", "protocol", "src_ip", "dst_ip",
    "packet_bytes", "queue_index", "queue_bytes", "ecn", "node_type", "details",
)


@lru_cache(maxsize=4096)
def address(value):
    return str(IPv4Address(value))


def packet_details(protocol, data):
    if protocol == 0x11:
        src_port, dst_port, seq, stamp, pg, payload = DATA.unpack_from(data)
        return f"sport={src_port} dport={dst_port} seq={seq} ts={stamp} pg={pg} payload_bytes={payload}"
    if protocol in (0xFC, 0xFD):
        src_port, dst_port, flags, pg, seq, stamp = ACK.unpack_from(data)
        return f"sport={src_port} dport={dst_port} seq={seq} ts={stamp} pg={pg} flags={flags:#x}"
    if protocol == 0xFE:
        pause_time, queue_bytes, queue = PFC.unpack_from(data)
        return f"pause_time={pause_time} queue_bytes={queue_bytes} queue_index={queue}"
    if protocol == 0xFF:
        flow_id, queue, ecn, feedback, total = CNP.unpack_from(data)
        return f"flow_id={flow_id} queue_index={queue} ecn={ecn} feedback={feedback} total={total}"
    if protocol == 0x06:
        src_port, dst_port = PORTS.unpack_from(data)
        return f"sport={src_port} dport={dst_port}"
    return ""


def convert(source_path, output_path, limit):
    with source_path.open("rb") as source:
        raw_count = source.read(PORT_COUNT.size)
        if len(raw_count) != PORT_COUNT.size:
            raise ValueError("Missing ns-3 trace header")
        port_count, = PORT_COUNT.unpack(raw_count)
        header_bytes = PORT_COUNT.size + port_count * PORT.size + WINDOW.size
        file_bytes = os.fstat(source.fileno()).st_size
        if header_bytes > file_bytes or (file_bytes - header_bytes) % RECORD.size:
            raise ValueError("File does not match the ns-3 trace layout or has a partial record")
        source.seek(header_bytes)

        total = (file_bytes - header_bytes) // RECORD.size
        count = total if limit is None else min(limit, total)
        with output_path.open("x", encoding="utf-8", buffering=1024 * 1024) as output:
            output.write("\t".join(COLUMNS) + "\n")
            remaining = count
            while remaining:
                batch_count = min(remaining, 16_384)
                batch_bytes = batch_count * RECORD.size
                batch = source.read(batch_bytes)
                if len(batch) != batch_bytes:
                    raise ValueError("Trace ended while reading a packet record")
                for record in RECORD.iter_unpack(batch):
                    (time, node, interface, queue, queue_bytes, src, dst,
                     size, protocol, event, ecn, node_type, data) = record
                    event_name = EVENTS.get(event, str(event))
                    protocol_name = PROTOCOLS.get(protocol) or f"0x{protocol:02x}"
                    details = packet_details(protocol, data)
                    output.write(
                        f"{time}\t{node}\t{interface}\t{event_name}\t{protocol_name}\t"
                        f"{address(src)}\t{address(dst)}\t{size}\t{queue}\t{queue_bytes}\t"
                        f"{ecn}\t{node_type}\t{details}\n"
                    )
                remaining -= batch_count
    return count, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Binary mix.tr from a completed run")
    parser.add_argument("output", type=Path, help="New text file (tab-separated)")
    parser.add_argument("--limit", type=int, help="Convert only the first N events")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    try:
        count, total = convert(args.input, args.output, args.limit)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Wrote {count:,} of {total:,} packet events to {args.output}")


if __name__ == "__main__":
    main()
