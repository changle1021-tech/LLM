#!/usr/bin/env python3
"""Replay collective primitives through the serving ns-3 backend.

Sizes are per-rank chunk bytes for AllGather/ReduceScatter, full tensor bytes
for AllReduce, matching profiler.hardware_multinode's recorded samples.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.core.config_builder import build_cluster_config
from serving.core.ns3_config import prepare_ns3_inputs
from serving.core.run_paths import build_run_paths
from chakra.src.converter.llm_converter import LLMConverter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cluster-config',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--sizes',type=int,nargs='+',default=[10240,40960,139264,327680,1048576,4194304])
parser.add_argument('--bandwidth',type=float)
parser.add_argument('--latency',type=float)
parser.add_argument('--payload',type=int,default=4096)
args=parser.parse_args()
output=args.output_dir.resolve()
output.mkdir(exist_ok=False,parents=True)
config=json.loads(args.cluster_config.read_text())
config['collective_links']={}
if args.bandwidth is not None:config['link_bw']=args.bandwidth
if args.latency is not None:config['link_latency']=args.latency
options=config.get('ns3',{})
if not isinstance(options,dict):
    raise ValueError('ns3 must be an object')
options=dict(options)
if options.get('links') is not None and (args.bandwidth is not None or args.latency is not None):
    raise ValueError('Edit explicit ns3.links to change their bandwidth or latency')
overrides=dict(options.get('config',{}))
overrides.setdefault('L2_CHUNK_SIZE',args.payload*4)
options.update(packet_payload_size=args.payload,config=overrides,host_dispatch_latency_ns=0)
config['ns3']=options
config_path=output/'cluster.json'
config_path.write_text(json.dumps(config,indent=2)+'\n')
astra=ROOT/'astra-sim'
paths=build_run_paths(str(astra),'probe',str(output/'inputs'))
cluster=build_cluster_config(str(astra),str(config_path),inputs_root=paths.inputs_root)
network,logical=prepare_ns3_inputs(str(astra),paths,cluster,config['ns3'])
count=cluster['total_npu']
binary=astra/'extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-optimized'
pattern=re.compile(r'sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles')
results=[]
started=time.time()
for operation in ['all_reduce','all_gather','reduce_scatter']:
    for size in args.sizes:
        if size<=0:raise ValueError('Sizes must be positive')
        folder=output/f'{operation}_{size}'
        folder.mkdir()
        workload=folder/'llm'
        row=['collective_0','1','REMOTE','0','LOCAL','0','REMOTE','0',operation.upper().replace('_',''),str(size*count if operation=='reduce_scatter' else size),'NONE']
        LLMConverter(str(folder/'trace.txt'),str(workload),count,0,False).convert_rows(
            'DECODE model_parallel_NPU_group: 1',[row, ['finish_1','1','REMOTE','0','LOCAL','0','REMOTE','0','NONE','0','NONE']])
        command=[str(binary),'--workload-configuration='+str(workload),
                 '--system-configuration='+paths.system_config,'--network-configuration='+network,
                 '--memory-configuration='+paths.memory_config,'--logical-topology-configuration='+logical,
                 '--start-npu-ids='+','.join(map(str,range(count)))]
        (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        p=subprocess.Popen(command,cwd=folder,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True)
        clocks={}
        with (folder/'backend.log').open('w') as log:
            for line in p.stdout:
                log.write(line)
                match=pattern.search(line)
                if match:clocks[int(match[1])]=int(match[3])
                if 'Waiting' in line:
                    p.stdin.write('exit\n' if len(clocks)==count else 'done\n')
                    p.stdin.flush()
        if p.wait()!=0 or len(clocks)!=count:
            raise RuntimeError(f'Primitive failed: {folder}; exit={p.returncode}')
        results.append(dict(collective=operation,bytes=size,us=max(clocks.values())/1000,rank_cycles=clocks))
        (output/'samples.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
(output/'completed.json').write_text(json.dumps(dict(state='complete',started=started,finished=time.time(),npus=count,samples=len(results)),indent=2)+'\n')
