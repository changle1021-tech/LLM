"""Generate run-isolated packet-network and logical-rank inputs for ns-3."""
import json
import math
from pathlib import Path

import yaml


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return value


def _uniform(value, name):
    values = value if isinstance(value, list) else [value]
    if not values or any(v != values[0] for v in values):
        raise ValueError(f'ns3 requires uniform {name}, or explicit ns3.links')
    return _number(values[0], name, name == 'link_bw')


def prepare_ns3_inputs(astra_sim, run_paths, cluster, options=None):
    """Keep logical collectives separate from physical packet links.

    Common cluster bandwidth is GiB/s; explicit packet links use decimal Gbit/s.
    An explicit switch list uses contiguous IDs following all NPU endpoint IDs.
    Analytical per-operation timing overrides have no packet-level equivalent.
    """
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError('ns3 must be an object')
    unknown = set(options) - {'links', 'switches', 'packet_payload_size', 'config', 'host_dispatch_latency_ns'}
    if unknown:
        raise ValueError(f'Unknown ns3 settings: {sorted(unknown)}')
    host_delay = options.get('host_dispatch_latency_ns', 0)
    if type(host_delay) is not int or not 0 <= host_delay <= 2**63-1:
        raise ValueError('ns3.host_dispatch_latency_ns must be a nonnegative int64')
    count = cluster['total_npu']
    directory = Path(run_paths.inputs_root) / 'ns3'
    output = directory / 'output'
    output.mkdir(parents=True, exist_ok=True)
    network = yaml.safe_load(Path(run_paths.network_config).read_text())
    dimensions = network['npus_count']
    if math.prod(dimensions) != count:
        raise ValueError('Logical ns3 dimensions do not cover all NPUs')
    logical = directory / 'logical_topology.json'
    logical.write_text(json.dumps({'logical-dims': [str(v) for v in dimensions]}, indent=2)+'\n')
    switches = options.get('switches', [])
    if not isinstance(switches, list) or any(type(v) is not int for v in switches) or switches != list(range(count, count+len(switches))):
        raise ValueError('ns3.switches must be consecutive IDs following the NPU ranks')
    node_count = count + len(switches)
    supplied = options.get('links')
    if supplied is None:
        if switches:
            raise ValueError('Explicit ns3 switches require explicit links')
        bw = _uniform(cluster['link_bw'], 'link_bw') * 8 * 1024**3 / 1e9
        latency = _uniform(cluster['link_latency'], 'link_latency')
        supplied = [dict(src=a,dst=b,bandwidth_gbps=bw,latency_ns=latency)
                    for a in range(count) for b in range(a+1,count)]
    if not isinstance(supplied, list):
        raise ValueError('ns3.links must be a list')
    links, edges, neighbors = [], {}, {n:set() for n in range(node_count)}
    for entry in supplied:
        if not isinstance(entry, dict) or set(entry)-{'src','dst','bandwidth_gbps','latency_ns','error_rate'}:
            raise ValueError('Invalid ns3 link fields')
        a,b = entry.get('src'),entry.get('dst')
        if type(a) is not int or type(b) is not int or not 0 <= a < node_count or not 0 <= b < node_count or a == b:
            raise ValueError('Invalid ns3 link endpoints')
        edge = tuple(sorted((a,b)))
        rate = int(round(_number(entry['bandwidth_gbps'],'bandwidth_gbps',True)*1e9))
        latency = _number(entry['latency_ns'],'latency_ns')
        error = _number(entry.get('error_rate',0),'error_rate')
        if rate < 1 or error > 1:
            raise ValueError('Invalid ns3 link rate or error probability')
        properties = (rate, latency, error)
        if edge in edges:
            if edges[edge] != properties:
                raise ValueError('Parallel ns3 links must have identical rate, latency and error rate')
        edges[edge] = properties
        neighbors[a].add(b);neighbors[b].add(a)
        links.append((a,b,rate,latency,error))
    # Servers are endpoints, not transit switches. Check every NPU pair using
    # only switches as intermediate vertices before the backend starts.
    for source in range(count):
        seen={source};todo=[source]
        while todo:
            node=todo.pop()
            for nxt in neighbors[node]-seen:
                seen.add(nxt)
                if nxt in switches:todo.append(nxt)
        if not set(range(count)) <= seen:
            raise ValueError('ns3 topology cannot connect all NPU endpoints through switches')
    topology = directory / 'topology.txt'
    lines=[f'{node_count} {len(switches)} {len(links)}',' '.join(map(str,switches))]
    lines += [f'{a} {b} {rate}bps {delay:.12g}ns {error:.12g}' for a,b,rate,delay,error in links]
    topology.write_text('\n'.join(lines)+'\n')
    payload=options.get('packet_payload_size',1000)
    if type(payload) is not int or not 64 <= payload <= 65500:
        raise ValueError('ns3.packet_payload_size must be an integer between 64 and 65500')
    replacements={'TOPOLOGY_FILE':str(topology),'PACKET_PAYLOAD_SIZE':str(payload)}
    for key,name in [('FLOW_FILE','flow.txt'),('TRACE_FILE','trace.txt'),('TRACE_OUTPUT_FILE','mix.tr'),
                     ('FCT_OUTPUT_FILE','fct.txt'),('PFC_OUTPUT_FILE','pfc.txt'),('QLEN_MON_FILE','qlen.txt'),
                     ('BW_MON_FILE','port_bandwidth.csv'),('RATE_MON_FILE','qp_rate.csv'),
                     ('NETWORK_MONITOR_SUMMARY_FILE','network_stats.json')]:
        replacements[key]=str(output/name)
    for key in ('FLOW_FILE','TRACE_FILE'):
        Path(replacements[key]).write_text('0\n')
    rates=sorted({row[2] for row in links})
    if rates:
        for key,base in [('KMAX_MAP',3200),('KMIN_MAP',800),('PMAX_MAP',0.2)]:
            pairs=[]
            for rate in rates:
                value=max(1,int(round(base*rate/400e9))) if key!='PMAX_MAP' else base
                pairs += [str(rate),str(value)]
            replacements[key]=str(len(rates))+' '+' '.join(pairs)
    template=Path(astra_sim)/'extern/network_backend/ns-3/scratch/config/config.txt'
    settings={}
    for line in template.read_text().splitlines():
        parts=line.split(maxsplit=1)
        if parts:settings[parts[0]]=parts[1] if len(parts)>1 else ''
    settings['ENABLE_TRACE']='0'
    # Frontend-owned monitor keys do not require changes to the ns-3 template.
    # Unlike packet tracing, periodic counter output is small enough for normal runs.
    settings.update(ENABLE_NETWORK_MONITOR='1', BW_MON_INTERVAL_NS='1000000',
                    QP_MON_INTERVAL_NS='10000')
    overrides=options.get('config',{})
    if not isinstance(overrides,dict) or set(overrides)-set(settings):
        raise ValueError('ns3.config must contain supported template keys')
    if set(overrides)&{'TOPOLOGY_FILE','FLOW_FILE','TRACE_FILE','TRACE_OUTPUT_FILE','FCT_OUTPUT_FILE','PFC_OUTPUT_FILE','QLEN_MON_FILE',
                       'BW_MON_FILE','RATE_MON_FILE','NETWORK_MONITOR_SUMMARY_FILE'}:
        raise ValueError('ns3 output and topology paths are generated per run')
    if 'PACKET_PAYLOAD_SIZE' in overrides:
        raise ValueError('Use ns3.packet_payload_size to set the packet payload')
    for key,value in overrides.items():
        if key == 'ENABLE_NETWORK_MONITOR' and (type(value) not in (bool, int) or value not in (0, 1)):
            raise ValueError('ns3.config.ENABLE_NETWORK_MONITOR must be 0 or 1')
        if key in {'BW_MON_INTERVAL_NS', 'QP_MON_INTERVAL_NS'} and (
                type(value) is not int or not 1 <= value <= (2**63 - 1) // 2):
            raise ValueError(f'ns3.config.{key} must be a positive integer in ns')
        if isinstance(value,(list,dict)) or '\n' in str(value) or '\r' in str(value):
            raise ValueError(f'Invalid ns3 config value for {key}')
        settings[key]=str(int(value) if isinstance(value, bool) else value)
    settings.update({key:value for key,value in replacements.items()
                     if key not in {'KMAX_MAP','KMIN_MAP','PMAX_MAP'} or key not in overrides})
    if settings["ENABLE_TRACE"] == "1":
        node_ids = " ".join(str(node) for node in range(node_count))
        Path(settings["TRACE_FILE"]).write_text(f"{node_count}\n{node_ids}\n")
    config=directory/'config.txt'
    config.write_text(''.join(f'{key} {value}\n' for key,value in settings.items()))
    (directory/'network_manifest.json').write_text(json.dumps(dict(
        backend='ns3',logical_dims=dimensions,npu_endpoints=count,switches=switches,
        links=[dict(src=a,dst=b,bandwidth_bps=rate,latency_ns=delay,error_rate=error)
               for a,b,rate,delay,error in links],packet_payload_size=payload,
        common_bandwidth_unit='GiB/s',explicit_bandwidth_unit='Gbit/s',latency_unit='ns',
        host_dispatch_latency_ns=host_delay),indent=2)+'\n')
    return str(config),str(logical)
