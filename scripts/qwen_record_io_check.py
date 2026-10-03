"""Windows CPU-only sharing check. New --output directory required; no GPU imports."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from genai_lab.qwen_record_io import (write_json, publish_progress, latest_progress,
    write_terminal, finalize_stopped_run, JSONReplaceError)


def hold_without_delete_share(path, ready, seconds):
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32', use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                     wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle; close.argtypes=[wintypes.HANDLE]; close.restype=wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ|WRITE, deliberately NO FILE_SHARE_DELETE.
    handle=create(str(path),0x80000000,3,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        ready.write_text('ready',encoding='utf-8')
        time.sleep(seconds)
    finally: close(handle)


def start_holder(path, ready, seconds):
    child=subprocess.Popen([sys.executable,'-m','scripts.qwen_record_io_check','--hold',str(path),
        '--ready',str(ready),'--seconds',str(seconds)], cwd=str(Path(__file__).resolve().parents[1]),
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline=time.monotonic()+15
    while not ready.exists():
        if child.poll() is not None or time.monotonic()>deadline:
            child.terminate(); child.wait(timeout=10)
            raise RuntimeError('lock holder failed to start')
        time.sleep(.01)
    return child


def stop_holder(child):
    try: child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill(); child.wait(); raise
    if child.returncode: raise RuntimeError('holder failed')


def snapshot_worker(directory):
    record={'status':'starting','request_sha256':'a'*64,'steps':[]}
    write_json(directory/'run.json',record)
    for index in range(40):
        record['steps'].append({'index':index})
        publish_progress(directory,record)
        time.sleep(.01)
    record['status']='completed'
    write_terminal(directory,record)


def check(directory):
    if os.name!='nt': raise SystemExit('Windows only; cross-platform injection tests are in pytest')
    directory.mkdir(parents=True,exist_ok=False)
    report={'gpu_executed':False,'models_loaded':False,'directory':str(directory)}
    target=directory/'locked.json'; write_json(target,{'old':True})
    ready=directory/'holder.ready'
    child=start_holder(target,ready,.7)
    try:
        oldtmp=directory/'legacy.tmp'; oldtmp.write_text('{"new":true}',encoding='utf-8')
        try:
            oldtmp.replace(target)
            raise AssertionError('expected legacy sharing collision was not reproduced')
        except PermissionError as error:
            report['legacy_replace_error']={'type':type(error).__name__,'winerror':getattr(error,'winerror',None)}
        start=time.monotonic()
        write_json(target,{'new':True})
        report['retry_seconds']=time.monotonic()-start
        assert json.loads(target.read_text(encoding='utf-8'))=={'new':True}
        report['bounded_retry_succeeded']=True
    finally: stop_holder(child)
    # Terminal file blocked longer than the entire budget. Backup and raw survive.
    terminal=directory/'terminal'; terminal.mkdir()
    write_json(terminal/'run.json',{'status':'starting'})
    raw=terminal/'raw.png'; raw.write_bytes(b'CPU preservation sentinel; not a generated image')
    before=raw.read_bytes()
    child=start_holder(terminal/'run.json',directory/'terminal.ready',3.0)
    final={'status':'completed','request_sha256':'a'*64,'steps':list(range(40))}
    try:
        try:
            write_terminal(terminal,final)
            raise AssertionError('expected bounded final publication failure')
        except JSONReplaceError as error:
            report['permanent_lock_error']=str(error)
            assert json.loads((terminal/'run.final.json').read_text(encoding='utf-8'))==final
            assert raw.read_bytes()==before
    finally: stop_holder(child)
    finalize_stopped_run(terminal,'a'*64,cancelled=False,error='lock released',returncode=1)
    assert json.loads((terminal/'run.json').read_text(encoding='utf-8'))==final
    report['terminal_recovered_raw_unchanged']=raw.read_bytes()==before
    # Real concurrent writer/reader processes: reader deliberately holds snapshots.
    live=directory/'concurrent'; live.mkdir()
    child=subprocess.Popen([sys.executable,'-m','scripts.qwen_record_io_check','--snapshots',str(live)],
        cwd=str(Path(__file__).resolve().parents[1]),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    seen=[]; deadline=time.monotonic()+30
    try:
        while child.poll() is None:
            assert time.monotonic()<deadline,'snapshot worker timeout'
            snapshot=latest_progress(live)
            if snapshot:
                seen.append(snapshot[0])
                with (live/'progress'/snapshot[0]).open('r',encoding='utf-8') as stream:
                    json.load(stream); time.sleep(.02)
            else: time.sleep(.005)
    finally:
        if child.poll() is None: child.terminate()
        stop_holder(child)
    a=json.loads((live/'run.json').read_text(encoding='utf-8'))
    b=json.loads((live/'run.final.json').read_text(encoding='utf-8'))
    assert a==b and a['status']=='completed' and len(a['steps'])==40 and seen
    report.update(snapshot_files=len(list((live/'progress').glob('*.json'))),
        distinct_snapshots_read=len(set(seen)),final_records_equal=True,status='passed')
    write_json(directory/'result.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output'); parser.add_argument('--hold'); parser.add_argument('--ready')
    parser.add_argument('--seconds',type=float,default=.7); parser.add_argument('--snapshots')
    args=parser.parse_args()
    if args.hold: hold_without_delete_share(Path(args.hold),Path(args.ready),args.seconds)
    elif args.snapshots: snapshot_worker(Path(args.snapshots))
    elif args.output: check(Path(args.output).resolve())
    else: parser.error('--output is required')


if __name__=='__main__': main()
