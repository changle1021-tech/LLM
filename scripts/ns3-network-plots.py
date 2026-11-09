#!/usr/bin/env python3
"""Render ns-3 port and QP monitor CSVs as interactive, offline HTML charts."""

import argparse
import csv
import html
import json
import math
from pathlib import Path


PORT_COLUMNS = (
    'time_ns', 'node_id', 'port_id', 'peer_node_id', 'link_bandwidth_bps',
    'interval_ns', 'tx_bytes', 'rx_bytes', 'tx_gbps', 'rx_gbps',
)
QP_COLUMNS = (
    'time_ns', 'src_node_id', 'dst_node_id', 'src_port', 'dst_port',
    'size_bytes', 'rate_bps',
)


class Envelope:
    """Bound plotted points while retaining local extrema and endpoint samples."""

    def __init__(self, limit):
        self.limit = limit
        self.points = []
        self.samples = 0

    def add(self, time_ns, value):
        self.samples += 1
        self.points.append([time_ns / 1e6, value])
        if len(self.points) > self.limit:
            points = self.points
            kept = [points[0]]
            for start in range(1, len(points) - 1, 4):
                block = points[start:min(start + 4, len(points) - 1)]
                indices = {min(range(len(block)), key=lambda i: block[i][1]),
                           max(range(len(block)), key=lambda i: block[i][1])}
                kept.extend(block[i] for i in sorted(indices))
            kept.append(points[-1])
            self.points = kept


def rows(path, required):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(required) <= set(reader.fieldnames):
            raise ValueError(f'{path}: expected columns {", ".join(required)}')
        for row in reader:
            try:
                values = {key: float(row[key]) if key.endswith('_gbps') else int(row[key])
                          for key in required}
                if any(not math.isfinite(value) or value < 0
                       for key, value in values.items() if key != 'peer_node_id'):
                    raise ValueError('values must be finite and nonnegative')
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'{path}:{reader.line_num}: {error}') from error
            yield values


def read_ports(path, limit):
    ports = {}
    for row in rows(path, PORT_COLUMNS):
        key = row['node_id'], row['port_id']
        baseline = (key not in ports and row['time_ns'] == 0 and row['interval_ns'] == 0
                    and all(row[field] == 0 for field in ('tx_bytes', 'rx_bytes', 'tx_gbps', 'rx_gbps')))
        if row['link_bandwidth_bps'] <= 0 or (row['interval_ns'] <= 0 and not baseline):
            raise ValueError(f'{path}: interval_ns must be positive except for a zero initial baseline; '
                             'link_bandwidth_bps must be positive')
        if key not in ports:
            ports[key] = dict(peer=row['peer_node_id'], capacity=row['link_bandwidth_bps'],
                              tx=Envelope(limit), rx=Envelope(limit), tx_bytes=0,
                              rx_bytes=0, last_time=-1, peak_tx=0.0, peak_rx=0.0)
        port = ports[key]
        if (port['peer'], port['capacity']) != (row['peer_node_id'], row['link_bandwidth_bps']):
            raise ValueError(f'{path}: link metadata changed for port {key}')
        if row['time_ns'] < port['last_time']:
            raise ValueError(f'{path}: timestamps moved backwards for port {key}')
        port['last_time'] = row['time_ns']
        for direction in ('tx', 'rx'):
            port[direction].add(row['time_ns'], row[f'{direction}_gbps'])
            port[f'{direction}_bytes'] += row[f'{direction}_bytes']
            port[f'peak_{direction}'] = max(port[f'peak_{direction}'], row[f'{direction}_gbps'])
    return ports


def read_qps(path, limit):
    pairs = {}
    active = {}
    timestamp = None
    touched = set()
    event_count = 0

    def emit():
        for pair in touched:
            data = pairs[pair]
            data['rate'].add(timestamp, data['rate_bps'] / 1e9)
            data['count'].add(timestamp, data['active'])

    for row in rows(path, QP_COLUMNS):
        time = row['time_ns']
        if timestamp is not None and time < timestamp:
            raise ValueError(f'{path}: timestamps moved backwards')
        if timestamp is not None and time != timestamp:
            emit()
            touched.clear()
        timestamp = time
        pair = row['src_node_id'], row['dst_node_id']
        if pair not in pairs:
            pairs[pair] = dict(rate=Envelope(limit), count=Envelope(limit), rate_bps=0,
                               active=0, events=0)
        key = pair + (row['src_port'], row['dst_port'], row['size_bytes'])
        previous = active.get(key, 0)
        rate = row['rate_bps']
        data = pairs[pair]
        data['rate_bps'] += rate - previous
        data['active'] += int(rate > 0) - int(previous > 0)
        data['events'] += 1
        if rate:
            active[key] = rate
        else:
            active.pop(key, None)
        touched.add(pair)
        event_count += 1
    if timestamp is not None:
        emit()
    return pairs, event_count, len(active)


STYLE = '''
body{font:15px system-ui,sans-serif;color:#243044;margin:24px;background:#f5f7fb}
main{max-width:1400px;margin:auto}h1{font-size:26px}h2{font-size:20px}
a{color:#245ac4}p{line-height:1.6}.card{background:white;padding:20px;margin:20px 0;
border:1px solid #dde3ee;border-radius:12px}.chart{height:340px;width:100%;display:block;
touch-action:none;cursor:crosshair}.legend{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.legend label{font-size:13px;cursor:pointer}.hint{font-size:13px;color:#53647e}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;
border-bottom:1px solid #e5e9f1}th:first-child,td:first-child{text-align:left}
button{border:1px solid #cbd5e5;border-radius:5px;background:white;padding:5px 10px;cursor:pointer}
'''


CHART_SCRIPT = r'''
const colors=['#2563eb','#e76f22','#0b9875','#d13065','#8656c9','#8a6d3b','#1097b2','#63758d'];
const fmt=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:4});
for(const [idx,spec] of charts.entries()){
 const box=document.createElement('section');box.className='card';
 const title=document.createElement('h2');title.textContent=spec.title;box.append(title);
 const legend=document.createElement('div');legend.className='legend';box.append(legend);
 const canvas=document.createElement('canvas');canvas.className='chart';box.append(canvas);
 const hint=document.createElement('p');hint.className='hint';
 hint.textContent='Hover for values. Scroll to zoom; drag to pan. ';box.append(hint);
 const reset=document.createElement('button');reset.textContent='Reset view';hint.append(reset);
 document.getElementById('charts').append(box);
 let xmin=Infinity,xmax=-Infinity,view=null,hover=null,drag=null;
 for(const s of spec.series){s.visible=true;if(s.points.length){xmin=Math.min(xmin,s.points[0][0]);
  xmax=Math.max(xmax,s.points[s.points.length-1][0]);}}
 if(!isFinite(xmin)){hint.prepend('No samples. ');continue;}
 if(xmax===xmin)xmax=xmin+0.001;view=[xmin,xmax];
 spec.series.forEach((s,i)=>{const label=document.createElement('label');
  const check=document.createElement('input');check.type='checkbox';check.checked=true;
  check.onchange=()=>{s.visible=check.checked;draw()};label.append(check);
  const text=document.createElement('span');text.style.color=colors[i%colors.length];
  text.textContent=s.name;label.append(text);legend.append(label);});
 function nearest(points,x){let lo=0,hi=points.length;while(lo<hi){let m=(lo+hi)>>1;
  if(points[m][0]<=x)lo=m+1;else hi=m;}
  if(spec.step==='before')return points[Math.min(lo,points.length-1)];
  if(spec.step==='after')return points[Math.max(0,lo-1)];
  if(!lo)return points[0];if(lo===points.length)return points[lo-1];
  return x-points[lo-1][0]<points[lo][0]-x?points[lo-1]:points[lo];}
 function draw(){const ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
  canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);
  const c=canvas.getContext('2d');c.scale(ratio,ratio);c.clearRect(0,0,w,h);
  const l=80,r=w-24,t=18,b=h-48,span=view[1]-view[0];let ymax=0;
  for(const s of spec.series)if(s.visible){for(const p of s.points)if(p[0]>=view[0]&&p[0]<=view[1])ymax=Math.max(ymax,p[1]);
   const p=nearest(s.points,view[0]);if(p)ymax=Math.max(ymax,p[1]);}
  ymax=ymax?ymax*1.12:1;const px=x=>l+(x-view[0])/span*(r-l),py=y=>b-y/ymax*(b-t);
  c.font='12px system-ui';c.lineWidth=1;
  for(let i=0;i<=5;i++){const y=b-i*(b-t)/5;c.strokeStyle='#e5e9f1';c.beginPath();c.moveTo(l,y);c.lineTo(r,y);c.stroke();
   c.fillStyle='#53647e';c.textAlign='right';c.fillText(fmt(i*ymax/5),l-8,y+4);
   const x=l+i*(r-l)/5;c.textAlign='center';c.fillText(fmt(view[0]+i*span/5),x,b+20);}
  c.fillStyle='#354660';c.textAlign='left';c.fillText(spec.unit,8,13);
  c.textAlign='center';c.fillText('Simulation time (ms)',(l+r)/2,h-5);
  c.save();c.beginPath();c.rect(l,t,r-l,b-t);c.clip();
  spec.series.forEach((s,i)=>{if(!s.visible||!s.points.length)return;
   c.strokeStyle=colors[i%colors.length];c.lineWidth=1.5;c.beginPath();let prev=null;
   for(const p of s.points){
    if(prev&&spec.step==='before')c.lineTo(px(prev[0]),py(p[1]));
    if(prev&&spec.step==='after')c.lineTo(px(p[0]),py(prev[1]));
    if(prev)c.lineTo(px(p[0]),py(p[1]));else c.moveTo(px(p[0]),py(p[1]));prev=p;}
   c.stroke();if(s.points.length===1){c.beginPath();c.arc(px(prev[0]),py(prev[1]),3,0,Math.PI*2);c.fillStyle=c.strokeStyle;c.fill();}});
  if(hover!==null){c.strokeStyle='#99a5b7';c.beginPath();c.moveTo(hover,t);c.lineTo(hover,b);c.stroke();}c.restore();
  if(hover!==null){const x=view[0]+(hover-l)/(r-l)*span;
   const lines=['Time: '+fmt(x)+' ms'];for(const s of spec.series)if(s.visible){const p=nearest(s.points,x);
    if(p)lines.push(s.name+': '+fmt(p[1])+' '+spec.unit+' @ '+fmt(p[0])+' ms');}
   c.font='12px system-ui';const width=Math.min(w-20,Math.max(...lines.map(a=>c.measureText(a).width))+20);
   const tx=Math.max(5,Math.min(hover+12,w-width-5));const ty=25;
   c.fillStyle='rgba(255,255,255,.96)';c.fillRect(tx,ty,width,lines.length*17+12);
   c.strokeStyle='#cad3e1';c.strokeRect(tx,ty,width,lines.length*17+12);
   c.fillStyle='#243044';c.textAlign='left';lines.forEach((line,i)=>c.fillText(line,tx+10,ty+18+i*17));}}
 canvas.onpointermove=e=>{const x=e.clientX-canvas.getBoundingClientRect().left;
  if(drag){const delta=(x-drag.x)/(canvas.clientWidth-104)*(drag.view[1]-drag.view[0]);view=[drag.view[0]-delta,drag.view[1]-delta];}
  hover=x;draw();};canvas.onpointerleave=()=>{hover=null;if(!drag)draw();};
 canvas.onpointerdown=e=>{drag={x:e.clientX-canvas.getBoundingClientRect().left,view:[...view]};canvas.setPointerCapture(e.pointerId);};
 canvas.onpointerup=e=>{drag=null;canvas.releasePointerCapture(e.pointerId);};
 canvas.addEventListener('wheel',e=>{e.preventDefault();const f=Math.max(0,Math.min(1,(e.offsetX-80)/(canvas.clientWidth-104)));
  const anchor=view[0]+f*(view[1]-view[0]),factor=e.deltaY>0?1.25:0.8;
  view=[anchor-(anchor-view[0])*factor,anchor+(view[1]-anchor)*factor];draw();},{passive:false});
 reset.onclick=()=>{view=[xmin,xmax];draw();};new ResizeObserver(draw).observe(canvas);draw();
}
'''


def write_page(path, title, description, charts, table=''):
    payload = json.dumps(charts, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{STYLE}</style><main>'
        f'<p><a href="index.html">Network monitor overview</a></p><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(description)}</p>{table}<div id="charts"></div></main>'
        f'<script>const charts={payload};\n{CHART_SCRIPT}</script></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path, help='Directory containing port_bandwidth.csv and qp_rate.csv')
    parser.add_argument('--output-dir', type=Path, help='HTML output directory (default: INPUT_DIR/plots)')
    parser.add_argument('--max-points', type=int, default=2048,
                        help='Bound displayed points per series; retain raw CSVs unchanged (default: 2048)')
    args = parser.parse_args()
    if args.max_points < 16:
        parser.error('--max-points must be at least 16')
    output = args.output_dir or args.input_dir / 'plots'
    try:
        ports = read_ports(args.input_dir / 'port_bandwidth.csv', args.max_points)
        pairs, qp_events, active_qps = read_qps(args.input_dir / 'qp_rate.csv', args.max_points)
        output.mkdir(parents=True, exist_ok=True)
        node_ids = sorted({node for node, _ in ports})
        table = ['<section class="card"><h2>Physical port samples</h2><table><tr>'
                 '<th>Node / port → peer</th><th>Link capacity (Gbps)</th><th>TX bytes</th>'
                 '<th>RX bytes</th><th>Peak TX (Gbps)</th><th>Peak RX (Gbps)</th></tr>']
        for (node, port_id), data in sorted(ports.items()):
            table.append(f'<tr><td><a href="node_{node}.html">{node} / {port_id} → {data["peer"]}</a></td>'
                         f'<td>{data["capacity"] / 1e9:g}</td><td>{data["tx_bytes"]}</td>'
                         f'<td>{data["rx_bytes"]}</td><td>{data["peak_tx"]:.6g}</td>'
                         f'<td>{data["peak_rx"]:.6g}</td></tr>')
        table.append('</table></section><section class="card"><h2>QP configured send rates</h2><ul>')
        for src, dst in sorted(pairs):
            table.append(f'<li><a href="qp_{src}_{dst}.html">Node {src} → node {dst}</a></li>')
        if not pairs:
            table.append('<li>No QP rate events were recorded.</li>')
        table.append('</ul></section>')
        description = ('Each physical port direction is counted separately; no network-wide sum is shown. '
                       'Rates are interval averages drawn over the interval ending at each sample timestamp, '
                       'including packet headers. Utilization divides the rate by '
                       'the configured physical link capacity. QP rates are configured send limits, not achieved '
                       'throughput. Charts retain sampled extrema when compacting long series; CSVs are unchanged.')
        write_page(output / 'index.html', 'ns-3 network monitor', description, [], ''.join(table))
        for node in node_ids:
            rates, utilization = [], []
            for (port_node, port_id), data in sorted(ports.items()):
                if port_node != node:
                    continue
                for direction in ('tx', 'rx'):
                    name = f'Port {port_id} → {data["peer"]}: {direction.upper()}'
                    rates.append(dict(name=name, points=data[direction].points))
                    utilization.append(dict(name=name, points=[
                        [time, value * 1e11 / data['capacity']] for time, value in data[direction].points]))
            write_page(output / f'node_{node}.html', f'Node {node}: physical ports', description, [
                dict(title='Port throughput', unit='Gbps', step='before', series=rates),
                dict(title='Physical link utilization', unit='%', step='before', series=utilization)])
        for (src, dst), data in sorted(pairs.items()):
            write_page(output / f'qp_{src}_{dst}.html', f'Node {src} → node {dst}: QP send limits',
                       'Events update the sum of active QPs after all changes at the same timestamp. '
                       'These are congestion-control send limits; their sum may exceed physical link bandwidth.', [
                           dict(title='Sum of configured QP rates', unit='Gbps', step='after',
                                series=[dict(name='Configured rate sum', points=data['rate'].points)]),
                           dict(title='Active QPs', unit='QPs', step='after',
                                series=[dict(name='Active QP count', points=data['count'].points)])])
        summary = dict(port_series=len(ports), port_samples=sum(p['tx'].samples for p in ports.values()),
                       qp_pairs=len(pairs), qp_events=qp_events, active_qps_at_end=active_qps,
                       max_points=args.max_points, html_files=1 + len(node_ids) + len(pairs))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    print(f'Wrote {summary["html_files"]} offline HTML files to {output.resolve()}')
    print(f'Overview: {(output / "index.html").resolve()}')


if __name__ == '__main__':
    main()
