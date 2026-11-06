"""Derive operation-specific effective links from a completed hardware sweep.

This is an explicit alternative to the acquisition's shared-latency model.
Both bandwidth and latency are fitted independently for each collective using
all its samples and the original relative-least-squares Ring fitter. Serving
results are never inputs. The acquisition and its original fits stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .core import hardware_multinode as hardware


def derive(document):
    interconnect = document['measured']['interconnect']
    samples = interconnect['samples']
    if not samples or any(sample.get('us_graphed') is None for sample in samples):
        raise ValueError('This calibration requires CUDA-graph timings for every sample')
    if any(sample['us'] != sample['us_graphed'] for sample in samples):
        raise ValueError('The fit target must be the recorded CUDA-graph timing')
    npus = interconnect['npus']
    memory_bw = interconnect['fit']['assumptions']['local_mem_bw_gbps']
    fits = {}
    for operation in hardware._COLLECTIVES:
        selected = [sample for sample in samples if sample['collective'] == operation]
        fits[operation] = hardware._fit(selected, npus=npus,
                                        local_mem_bw_gbps=memory_bw)
    joint = interconnect['fit']
    links = dict(link_bw=joint['bandwidth_gbps'], link_latency=joint['latency_ns'],
                 collective_links={operation: dict(link_bw=fit['bandwidth_gbps'],
                     link_latency=fit['latency_ns']) for operation, fit in fits.items()})
    return dict(method='independent per-collective relative least squares; '
                       'all recorded graphed samples; no fixed latency',
                npus=npus, samples=len(samples), bandwidth_unit='GiB/s',
                latency_unit='ns', fits=fits, cluster_links=links)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware-yaml', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('--output-dir must not exist')
    import yaml

    source = args.hardware_yaml.read_bytes()
    result = derive(yaml.safe_load(source))
    result.update(input=str(args.hardware_yaml.resolve()),
                  input_sha256=hashlib.sha256(source).hexdigest(),
                  source_sha256={str(path.resolve()): hashlib.sha256(
                      path.read_bytes()).hexdigest()
                      for path in (Path(__file__), Path(hardware.__file__))})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'fit.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'cluster_links.json').write_text(
        json.dumps(result['cluster_links'], indent=2) + '\n')
    print(args.output_dir / 'cluster_links.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
