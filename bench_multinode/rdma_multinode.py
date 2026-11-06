"""Select host RDMA character devices nearest the explicitly selected GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass


_PCI = re.compile(r'^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$')
_DISTANCE = {'PIX': 0, 'PXB': 1, 'PHB': 2, 'NODE': 3, 'SYS': 4}
_LOCAL_ENV = ('NCCL_IB_HCA', 'UCX_NET_DEVICES', 'NCCL_SOCKET_IFNAME',
              'GLOO_SOCKET_IFNAME', 'NCCL_IB_GID_INDEX')


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    pci: str


@dataclass(frozen=True)
class Port:
    hca: str
    number: int
    pci_path: Path
    numa: int
    devices: tuple[str, ...]

    @property
    def name(self):
        return f'{self.hca}:{self.number}'


def _pci_id(value: str) -> str:
    value = value.strip()
    if not _PCI.fullmatch(value):
        raise ValueError(f'Invalid PCI address: {value!r}')
    domain, bus, device = value.lower().split(':')
    return f'{int(domain, 16):04x}:{bus}:{device}'


def _command(args: list[str]) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=15).stdout


def gpu_inventory() -> list[GPU]:
    output = _command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                       '--format=csv,noheader,nounits'])
    return [GPU(int(index.strip()), uuid.strip(), _pci_id(pci))
            for index, uuid, pci in csv.reader(output.splitlines())]


def select_gpus(spec: str, inventory: list[GPU]) -> list[GPU]:
    """Resolve Docker GPU indices/UUIDs, never guess which GPUs a count selects."""
    spec = spec.strip()
    if spec == 'all':
        selected = inventory
    else:
        fields = next(csv.reader([spec]))
        if spec.startswith('device='):
            identifiers = spec[len('device='):].split(',')
        elif len(fields) == 1 and fields[0].startswith('device='):
            identifiers = fields[0][len('device='):].split(',')
        else:
            raise ValueError('Automatic RDMA needs VLLM_GPUS=all or an explicit '
                             'Docker device list, e.g. VLLM_GPUS=\'"device=0,1"\'')
        selected = []
        for identifier in identifiers:
            identifier = identifier.strip()
            matches = [gpu for gpu in inventory
                       if str(gpu.index) == identifier
                       or (identifier.startswith('GPU-') and gpu.uuid.startswith(identifier))]
            if len(matches) != 1:
                raise ValueError(f'GPU selector {identifier!r} is unknown or ambiguous; '
                                 'use physical GPU indices or unique GPU UUIDs')
            if matches[0] not in selected:
                selected.append(matches[0])
    if not selected:
        raise ValueError('No GPUs selected')
    return selected


def _numa(pci_path: Path) -> int:
    try:
        return int((pci_path / 'numa_node').read_text().strip())
    except (OSError, ValueError):
        return -1


def rdma_ports(sys_root: Path, dev_root: Path) -> list[Port]:
    result = []
    for hca in sorted((sys_root / 'class/infiniband').glob('*')):
        pci_path = (hca / 'device').resolve(strict=True)
        if not _PCI.fullmatch(pci_path.name):
            continue  # Software RDMA has no physical GPU/NIC affinity.
        devices = []
        for verbs in (pci_path / 'infiniband_verbs').glob('uverbs*'):
            device = dev_root / 'infiniband' / verbs.name
            if device.is_char_device():
                devices.append(str(device))
        if not devices:
            continue
        for port in sorted((hca / 'ports').glob('*')):
            try:
                active = (port / 'state').read_text().strip().split(':')[0] == '4'
                number = int(port.name)
            except (OSError, ValueError):
                continue
            if active:
                result.append(Port(hca.name, number, pci_path, _numa(pci_path),
                                   tuple(sorted(devices))))
    if not result:
        raise ValueError('No active physical RDMA port with an accessible uverbs device; '
                         'check /sys/class/infiniband and /dev/infiniband')
    return result


def topology_distances(output: str, hcas: set[str]) -> dict[tuple[int, str], int]:
    """Read both direct mlx5 columns and NIC-number columns with a NIC legend."""
    aliases = {name: name for name in hcas}
    for line in output.splitlines():
        match = re.match(r'\s*(NIC\d+):\s*(.*)', line)
        if match:
            for name in hcas:
                if re.search(r'(?<![\w])' + re.escape(name) + r'(?![\w])', match[2]):
                    aliases[match[1]] = name
    header = None
    distances = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if header is None and fields[0].startswith('GPU') and 'GPU0' in fields:
            if fields[0] == 'GPU0' and len(fields) > 1 and fields[1] not in _DISTANCE and fields[1] != 'X':
                header = fields
            continue
        if header is None or not re.fullmatch(r'GPU\d+', fields[0]):
            continue
        for column, distance in zip(header, fields[1:]):
            if column in aliases and distance in _DISTANCE:
                distances[(int(fields[0][3:]), aliases[column])] = _DISTANCE[distance]
    return distances


def _pci_distance(gpu_path: Path, port: Port) -> tuple[int, int] | None:
    left, right = gpu_path.parts, port.pci_path.parts
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    hops = len(left) + len(right) - 2 * common
    common_parts = left[:common]
    if any(_PCI.fullmatch(part) for part in common_parts):
        return (0 if hops <= 2 else 1), hops
    if any(re.fullmatch(r'pci[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}', part) for part in common_parts):
        return 2, hops
    numa = _numa(gpu_path)
    if numa >= 0 and port.numa >= 0:
        return (3 if numa == port.numa else 4), hops
    return None


def make_plan(gpus: list[GPU], ports: list[Port], distances: dict,
              sys_root: Path = Path('/sys'), dev_root: Path = Path('/dev')) -> dict:
    chosen = {}
    assignments = []
    for gpu in gpus:
        path = (sys_root / 'bus/pci/devices' / gpu.pci).resolve(strict=True)
        scores = []
        # Use a single scoring method per GPU so an omitted topology entry cannot
        # appear artificially closer than a measured one.
        native_complete = all((gpu.index, port.hca) in distances for port in ports)
        for port in ports:
            fallback = _pci_distance(path, port)
            if native_complete:
                score = (distances[(gpu.index, port.hca)],)
            elif fallback is not None:
                score = fallback
            else:
                continue
            scores.append((score, port))
        if not scores:
            # NCCL keeps its own selection over every port mapped below.
            print(f'No PCI/NUMA affinity for GPU {gpu.index}; leaving its NIC to NCCL',
                  file=sys.stderr)
            nearest = ports
            assignments.append(dict(gpu_index=gpu.index, gpu_uuid=gpu.uuid, pci=gpu.pci,
                                    ports=[port.name for port in nearest], method='unbound'))
            for port in nearest:
                chosen[port.name] = port
            continue
        best = min(score for score, _ in scores)
        nearest = [port for score, port in scores if score == best]
        for port in nearest:
            chosen[port.name] = port
        assignments.append(dict(gpu_index=gpu.index, gpu_uuid=gpu.uuid, pci=gpu.pci,
                                ports=[port.name for port in nearest],
                                method='nvidia_smi_topology' if native_complete else 'sysfs_pci_numa'))
    names = sorted(chosen)
    devices = sorted({device for port in chosen.values() for device in port.devices})
    rdma_cm = dev_root / 'infiniband/rdma_cm'
    if rdma_cm.is_char_device():
        devices.append(str(rdma_cm))
    return dict(assignments=assignments, devices=devices, environment={
        'NCCL_IB_HCA': '=' + ','.join(names),
        'UCX_NET_DEVICES': ','.join(names),
    })


def docker_arguments(plan: dict) -> list[str]:
    args = []
    for name, value in plan['environment'].items():
        if name in os.environ and os.environ[name] != value:
            # A user selection wins; the launcher forwards the host value.
            print(f'Using user {name}={os.environ[name]} instead of automatic {value}',
                  file=sys.stderr)
            continue
        args.extend(('-e', f'{name}={value}'))
    for device in plan['devices']:
        args.extend(('--device', f'{device}:{device}'))
    args.extend(('--cap-add', 'IPC_LOCK', '--ulimit', 'memlock=-1:-1'))
    return args


def configure_worker_environment() -> None:
    """Keep each Ray node's NIC selection instead of copying the driver's names."""
    from vllm.ray.ray_env import RAY_NON_CARRY_OVER_ENV_VARS_FILE

    path = Path(RAY_NON_CARRY_OVER_ENV_VARS_FILE)
    existing = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(existing, list) or any(not isinstance(name, str) for name in existing):
        raise ValueError(f'Invalid Ray environment exclusion list: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(sorted(set(existing) | set(_LOCAL_ENV))) + '\n')
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default=os.environ.get('VLLM_GPUS', 'all'))
    parser.add_argument('--format', choices=('json', 'docker-args'), default='json')
    parser.add_argument('--configure-worker-env', action='store_true',
                        help='Inside the container, preserve node-local NIC variables for Ray workers')
    args = parser.parse_args()
    try:
        if args.configure_worker_env:
            configure_worker_environment()
            return 0
        gpus = select_gpus(args.gpus, gpu_inventory())
        ports = rdma_ports(Path('/sys'), Path('/dev'))
        try:
            topology = _command(['nvidia-smi', 'topo', '-m'])
        except (OSError, subprocess.SubprocessError):
            topology = ''
        plan = make_plan(gpus, ports, topology_distances(topology, {p.hca for p in ports}))
        for assignment in plan['assignments']:
            print(f"GPU {assignment['gpu_index']} ({assignment['pci']}) -> "
                  f"{', '.join(assignment['ports'])} [{assignment['method']}]", file=sys.stderr)
        if args.format == 'docker-args':
            print('\n'.join(docker_arguments(plan)))
        else:
            print(json.dumps(plan, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'RDMA selection failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
