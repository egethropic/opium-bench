#!/usr/bin/env python3
"""Manually preview, launch, inspect, resume and analyze frozen v2 protocols.

Local HTTP commands use the same service/worker path as the browser. No model is
loaded and no generation is performed by list or dry-run. An unchanged expansion
hash is required for launch and resume. Existing evidence is never overwritten.
"""
import argparse
import json
import math
from pathlib import Path
import time
from urllib.parse import quote

from lab.protocol_library import dry_run, list_protocols, load_protocol
from lab.storage import atomic_json
from run_lab_experiments import Client


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(value,path=None):
    if path:
        path=Path(path)
        if path.exists():raise ValueError('Output exists; choose a new path to preserve earlier evidence')
        atomic_json(path,value)
    else:print(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8766')
    sub=p.add_subparsers(dest='operation',required=True)
    listing=sub.add_parser('list',help='List installed protocol templates without contacting the app');listing.add_argument('--output',type=Path)
    preview=sub.add_parser('dry-run',help='Freeze an exact matrix without model loading or generation')
    preview.add_argument('--protocol');preview.add_argument('--document',type=Path);preview.add_argument('--mode',choices=['smoke','full'],default='smoke')
    preview.add_argument('--seeds',type=int,nargs='+');preview.add_argument('--order-seed',type=int,default=1729);preview.add_argument('--output',type=Path)
    launch=sub.add_parser('launch',help='Queue the exact frozen preview through the lab service')
    launch.add_argument('--preview',type=Path,required=True);launch.add_argument('--document',type=Path);launch.add_argument('--calibration',required=True);launch.add_argument('--bindings',type=Path);launch.add_argument('--output',type=Path)
    poll=sub.add_parser('poll',help='Read durable job/stage receipts');poll.add_argument('job_id');poll.add_argument('--wait',action='store_true');poll.add_argument('--timeout',type=float,default=3600);poll.add_argument('--interval',type=float,default=2);poll.add_argument('--output',type=Path)
    resume=sub.add_parser('resume',help='Resume queued work; retries need a separate explicit flag')
    resume.add_argument('job_id');resume.add_argument('--expansion-sha256',required=True);resume.add_argument('--retry-failed',action='store_true');resume.add_argument('--output',type=Path)
    analysis=sub.add_parser('analysis',help='Report planned/observed denominators and an optional paired contrast')
    analysis.add_argument('job_id');analysis.add_argument('--endpoint');analysis.add_argument('--arm-a');analysis.add_argument('--arm-b');analysis.add_argument('--attempt-policy',choices=['first','latest'],default='first');analysis.add_argument('--output',type=Path)
    return p


def main(argv=None,client_factory=Client):
    args=parser().parse_args(argv)
    if args.output and args.output.exists():raise ValueError('Output exists; choose a new path to preserve earlier evidence')
    if args.operation=='list':
        return _write([{'id':d['id'],'title':d['title'],'version':d['version'],'content_sha256':d['content_sha256']} for d in list_protocols()],args.output)
    if args.operation=='dry-run':
        if bool(args.protocol)==bool(args.document):raise ValueError('Choose exactly one protocol ID or document')
        document=_read(args.document) if args.document else load_protocol(args.protocol)
        return _write(dry_run(document,args.mode,seeds=args.seeds,order_seed=args.order_seed),args.output)
    if args.operation=='launch':
        preview=_read(args.preview);document=_read(args.document) if args.document else load_protocol(preview['protocol_id'])
        expected=dry_run(document,preview['mode'],seeds=preview['seeds'],order_seed=preview['order_seed'])
        if expected!=preview:raise ValueError('Frozen preview differs from the supplied/current protocol; intentionally regenerate it first')
        payload={'document':document,'mode':preview['mode'],'seeds':preview['seeds'],'order_seed':preview['order_seed'],
            'expansion_sha256':preview['expansion_sha256'],'calibration_id':args.calibration,'bindings':_read(args.bindings) if args.bindings else {}}
        return _write(client_factory(args.url).command('start_protocol',payload),args.output)
    client=client_factory(args.url)
    if args.operation=='resume':
        return _write(client.command('resume_protocol',{'research_job_id':args.job_id,'expansion_sha256':args.expansion_sha256,'retry_failed':args.retry_failed}),args.output)
    if args.operation=='analysis':
        payload={'research_job_id':args.job_id,'attempt_policy':args.attempt_policy}
        for key in ('endpoint','arm_a','arm_b'):
            if getattr(args,key) is not None:payload[key]=getattr(args,key)
        return _write(client.command('analyze_protocol',payload),args.output)
    if not math.isfinite(args.timeout) or not math.isfinite(args.interval) or args.timeout<=0 or not .1<=args.interval<=60:raise ValueError('Polling requires a positive timeout and interval from 0.1 to 60 seconds')
    started=time.monotonic();previous=None
    while True:
        result=client.get('/api/research/'+quote(args.job_id,safe=''))
        if not args.wait or result.get('settled') or result.get('receipt',{}).get('status') in {'complete','partial','stopped','failed','resource_stopped','preparation_failed'}:
            return _write(result,args.output)
        if time.monotonic()-started>=args.timeout:raise TimeoutError('Polling timed out; the job continues in the app')
        state=(result.get('status_counts'),result.get('receipt',{}).get('status'))
        if state!=previous:print(json.dumps({'job_id':args.job_id,'status_counts':state[0],'status':state[1]}),flush=True);previous=state
        time.sleep(args.interval)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,TimeoutError) as error:raise SystemExit(str(error))
