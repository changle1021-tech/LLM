"""Distributed copy of the local hardware acquisition, with unchanged fits."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
import uuid

from .core import hardware_multinode as hardware
from .core import logger as log


_NETWORK_ENV = (
    'NCCL_IB_HCA', 'UCX_NET_DEVICES', 'NCCL_SOCKET_IFNAME',
    'GLOO_SOCKET_IFNAME', 'NCCL_IB_GID_INDEX', 'NCCL_IB_DISABLE',
    'NCCL_NET', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_CROSS_NIC',
    'NCCL_PXN_DISABLE', 'NCCL_NET_GDR_LEVEL',
)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _layout(npus):
    names = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
             'MASTER_ADDR', 'MASTER_PORT')
    if any(name not in os.environ for name in names):
        raise ValueError('Launch this entry point with torchrun on every node')
    rank, local, world, local_world = (int(os.environ[key]) for key in names[:4])
    if (world != npus or world < 2 or local_world < 1 or world % local_world
            or not 0 <= rank < world or not 0 <= local < local_world):
        raise ValueError('Invalid torchrun layout or --npus differs from WORLD_SIZE')
    return rank, local, world, local_world


def _rdma_reference(path):
    environment = {key: os.environ.get(key) for key in _NETWORK_ENV}
    devices = sorted(str(p) for p in Path('/dev/infiniband').glob('uverbs*')
                     if p.is_char_device())
    if path is None:
        return dict(reference=None, environment=environment, devices=devices)
    reference = json.loads(path.read_text())
    for key, value in reference['environment'].items():
        if key not in _NETWORK_ENV:
            raise ValueError(f'Unsupported RDMA reference environment key: {key}')
        if os.environ.get(key) != value:
            raise ValueError(f'{key} differs from the vLLM RDMA reference')
    expected = sorted(p for p in reference['devices']
                      if Path(p).name.startswith('uverbs'))
    if not expected:
        raise ValueError('The vLLM RDMA reference must admit a uverbs device')
    if environment['NCCL_IB_DISABLE'] == '1' or environment['NCCL_NET'] == 'Socket':
        raise ValueError('TCP-only NCCL settings conflict with the RDMA reference')
    if devices != expected:
        raise ValueError('Mounted uverbs devices differ from the vLLM RDMA reference')
    for device in reference['devices']:
        if not Path(device).is_char_device():
            raise ValueError(f'Referenced RDMA device is unavailable: {device}')
    return dict(reference=str(path), reference_sha256=hashlib.sha256(
        path.read_bytes()).hexdigest(), mapping=reference,
        environment=environment, devices=devices)


def _nic_snapshot(reference):
    # Use all ports admitted by the reference, not every NIC on the host.
    ports = set()
    for assignment in reference.get('mapping', {}).get('assignments', []):
        ports.update(assignment['ports'])
    for entry in (reference['environment'].get('UCX_NET_DEVICES') or '').split(','):
        if ':' in entry and entry.rsplit(':', 1)[1].isdigit():
            ports.add(entry)
    if not ports:
        for device in reference['devices']:
            ibdev = Path('/sys/class/infiniband_verbs') / Path(device).name / 'ibdev'
            if ibdev.is_dir():
                for hca in ibdev.iterdir():
                    for port in (Path('/sys/class/infiniband') / hca.name / 'ports').glob('*'):
                        ports.add(f'{hca.name}:{port.name}')
    result = {}
    for entry in sorted(ports):
        hca, number = entry.split(':')
        port = Path('/sys/class/infiniband') / hca / 'ports' / number
        facts = {}
        for key in ('state', 'rate', 'counters/port_xmit_data',
                    'counters/port_rcv_data', 'counters/port_rcv_errors',
                    'counters/port_xmit_discards'):
            try:
                facts[key] = (port / key).read_text().strip()
            except OSError as exc:
                facts[key] = {'unavailable': str(exc)}
        result[entry] = facts
    return dict(at=hardware._dt(), ports=result, data_counter_unit='4-byte words')


def _combined(records, memory_bw):
    combined = dict(records[0])
    samples = [dict(row, repeat=index + 1) for index, record in enumerate(records)
               for row in record['samples']]
    fit = hardware._fit(samples, npus=combined['npus'],
                        local_mem_bw_gbps=memory_bw)
    combined.update(samples=samples, fit=fit,
        collective_fits=hardware._fit_collectives(samples,
            npus=combined['npus'], local_mem_bw_gbps=memory_bw,
            fixed_latency_ns=fit['latency_ns']))
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', required=True)
    parser.add_argument('--npus', required=True, type=int)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rdma-reference', type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    rank, local, world, local_world = _layout(args.npus)
    import torch
    import torch.distributed as dist

    output = args.output_dir.resolve()
    created = False
    try:
        # Finish every node's preflight before creating a shared or local path.
        dist.init_process_group('gloo', rank=rank, world_size=world,
                                timeout=datetime.timedelta(seconds=180))
        participant = dict(rank=rank, local_rank=local, hostname=socket.gethostname(),
            local_world_size=local_world, output=str(output), error=None,
            repeats=args.repeats, hardware=args.hardware,
            run_id=uuid.uuid4().hex if rank == 0 else None)
        try:
            if output.exists():
                raise ValueError(f'Output directory must not exist: {output}')
            if local >= torch.cuda.device_count():
                raise ValueError('LOCAL_RANK does not name a visible CUDA device')
            participant['rdma'] = _rdma_reference(args.rdma_reference)
            torch.cuda.set_device(local)
            gpu = torch.cuda.get_device_properties(local)
            participant['gpu'] = dict(name=gpu.name, uuid=str(getattr(gpu, 'uuid', 'unknown')))
        except Exception as exc:
            participant['error'] = f'{type(exc).__name__}: {exc}'
        participants = [None] * world
        dist.all_gather_object(participants, participant)
        errors = [row for row in participants if row['error']]
        if errors:
            raise ValueError(f'Cluster preflight failed: {errors}')
        if any((row['repeats'], row['hardware'], row['local_world_size'])
               != (args.repeats, args.hardware, local_world) for row in participants):
            raise ValueError('All ranks must use the same acquisition settings')
        # Torchrun owns the TCP store. Unique prefixes prevent repeated groups
        # from consuming a previous group's NCCL bootstrap keys.
        run_store = dist.PrefixStore(
            f"hardware_multinode/{participants[0]['run_id']}/",
            dist.TCPStore(os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']),
                          is_master=False,
                          timeout=datetime.timedelta(seconds=180)))
        output.mkdir(parents=True, exist_ok=True)
        created = True
        with (output / f'rank_{rank:03d}_start.json').open('x') as stream:
            json.dump(participant, stream, indent=2)
        if local == 0:
            _write(output / f'node_{rank:03d}_nic_before.json',
                   _nic_snapshot(participant['rdma']))
        if rank == 0:
            _write(output / 'provenance.json', dict(
                entry='python -m profiler.hardware_multinode', participants=participants,
                npus=world, repeats=args.repeats,
                protocol='original global-rank-zero CUDA-graph and isolated timings',
                fit='original shared joint latency; per-operation bandwidth at that latency',
                torch=str(torch.__version__), cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),
                sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__), Path(hardware.__file__))}))
        dist.barrier()
        dist.destroy_process_group()
        log.configure('INFO')
        spec = hardware.probe_spec()
        memory_bw = int(spec['memory_bw_gbps'])
        records = []
        for index in range(args.repeats):
            if rank == 0:
                log.info('Collective acquisition %d/%d across %d ranks',
                         index + 1, args.repeats, world)
            samples = hardware._worker(rank, world, hardware._SIZES_BYTES,
                                       None, local_rank=local,
                                       store=dist.PrefixStore(f'repeat_{index}/', run_store))
            if rank != 0:
                continue
            fit = hardware._fit(samples, npus=world, local_mem_bw_gbps=memory_bw)
            record = dict(npus=world, collectives=list(hardware._COLLECTIVES),
                dtype='bfloat16', samples=samples, fit=fit,
                launch=dict(kind='torchrun', nodes=world // local_world,
                            local_world_size=local_world),
                scope='effective calibration for the complete measured group',
                sample_bytes='AllReduce input; AllGather local input; ReduceScatter local output',
                collective_fits=hardware._fit_collectives(samples, npus=world,
                    local_mem_bw_gbps=memory_bw, fixed_latency_ns=fit['latency_ns']))
            repeat_dir = output / f'repeat_{index + 1:03d}'
            hardware.write_hardware_yaml(repeat_dir / 'hardware.yaml',
                                          args.hardware, spec, record)
            records.append(record)
        if local == 0:
            _write(output / f'node_{rank:03d}_nic_after.json',
                   _nic_snapshot(participant['rdma']))
        dist.init_process_group('gloo', rank=rank, world_size=world,
                                store=dist.PrefixStore('completion/', run_store),
                                timeout=datetime.timedelta(seconds=180))
        dist.barrier()
        if rank == 0:
            combined = _combined(records, memory_bw)
            _write(output / 'raw_samples.json', combined['samples'])
            hardware.write_hardware_yaml(output / 'hardware.yaml',
                                          args.hardware, spec, combined)
            _write(output / 'COMPLETED.json', dict(status='completed',
                repeats=args.repeats, samples=len(combined['samples']),
                hardware_yaml_sha256=hashlib.sha256(
                    (output / 'hardware.yaml').read_bytes()).hexdigest()))
            log.success('Wrote %s', output / 'hardware.yaml')
        dist.barrier()
        return 0
    except BaseException as exc:
        if created:
            _write(output / f'ABORTED_rank_{rank:03d}.json',
                   dict(status='aborted', error=traceback.format_exc()))
            if rank == 0:
                (output / 'COMPLETED.json').unlink(missing_ok=True)
                _write(output / 'ABORTED.json', dict(status='aborted',
                                                   error=str(exc)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
