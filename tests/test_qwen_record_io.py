"""CPU-only Windows sharing failures are injected; also runnable on Linux."""
import json
from pathlib import Path
import threading

import pytest

from genai_lab import qwen_record_io as records


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    waits = []
    def sleep(seconds):
        waits.append(seconds); now[0] += seconds
    monkeypatch.setattr(records.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(records.time, "sleep", sleep)
    return now, waits


def test_permission_error_retries_then_publishes_complete_json(tmp_path, monkeypatch, clock):
    target = tmp_path/'run.json'; target.write_text('{"old":true}', encoding='utf-8')
    original = Path.replace
    calls = []
    def locked(path, destination):
        calls.append(1)
        if len(calls) <= 3:
            assert read(target) == {"old": True}
            raise PermissionError("WinError 5")
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", locked)
    records.write_json(target, {"status":"completed", "text":"기록"})
    assert len(calls) == 4 and clock[1] == [.02,.04,.08]
    assert read(target) == {"status":"completed", "text":"기록"}
    assert list(tmp_path.glob('*.tmp')) == []


def test_permission_error_exhaustion_is_bounded_and_retains_pending(tmp_path, monkeypatch, clock):
    target=tmp_path/'run.json'; target.write_text('{"old":true}', encoding='utf-8')
    def blocked(*args): raise PermissionError("locked")
    monkeypatch.setattr(Path, 'replace', blocked)
    with pytest.raises(records.JSONReplaceError) as caught:
        records.write_json(target, {"status":"completed"})
    assert sum(clock[1]) <= 2.0 and len(clock[1]) <= 11
    assert read(target)=={"old":True}
    assert read(caught.value.pending_path)=={"status":"completed"}


def test_other_io_error_is_not_retried(tmp_path, monkeypatch, clock):
    def disk(*args): raise OSError(28, 'disk full')
    monkeypatch.setattr(Path, 'replace', disk)
    with pytest.raises(OSError): records.write_json(tmp_path/'data.json', {'a':1})
    assert clock[1] == []


def test_progress_is_immutable_and_incomplete_snapshot_is_ignored(tmp_path):
    p1=records.publish_progress(tmp_path, {'steps':[0]})
    before=p1.read_bytes()
    p2=records.publish_progress(tmp_path, {'steps':[0,1]})
    (p2.parent/'000000003.json').write_text('{partial',encoding='utf-8')
    (p2.parent/'000000004.json.pending.tmp').write_text('{}',encoding='utf-8')
    assert records.latest_progress(tmp_path)==(p2.name, {'steps':[0,1]})
    assert p1.read_bytes()==before and not (tmp_path/'run.json').exists()


def test_simultaneous_reader_and_snapshot_writer_terminal_complete(tmp_path):
    observed=[]; stop=threading.Event(); errors=[]
    def reader():
        try:
            while not stop.is_set():
                latest=records.latest_progress(tmp_path)
                if latest: observed.append(latest[1]['index'])
                stop.wait(.001)
        except Exception as error: errors.append(error)
    thread=threading.Thread(target=reader); thread.start()
    try:
        for i in range(40): records.publish_progress(tmp_path, {'index':i,'steps':list(range(i+1))})
        final={'status':'completed','steps':list(range(40)),'request_sha256':'a'*64}
        records.write_terminal(tmp_path, final)
    finally:
        stop.set(); thread.join(timeout=10)
    assert not thread.is_alive() and not errors and observed
    assert read(tmp_path/'run.json')==read(tmp_path/'run.final.json')==final
    assert len(list((tmp_path/'progress').glob('*.json')))==40


def test_terminal_backup_survives_persistent_lock_raw_unchanged(tmp_path, monkeypatch, clock):
    raw=tmp_path/'raw.png'; raw.write_bytes(b'existing result bytes')
    canonical=tmp_path/'run.json'; records.write_json(canonical, {'status':'starting'})
    original=Path.replace
    def blocked(path, target):
        if Path(target)==canonical: raise PermissionError('locked final')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',blocked)
    final={'status':'completed','request_sha256':'a'*64,'all_fields':list(range(40))}
    with pytest.raises(records.JSONReplaceError): records.write_terminal(tmp_path,final)
    assert read(tmp_path/'run.final.json')==final
    assert raw.read_bytes()==b'existing result bytes'
    monkeypatch.setattr(Path,'replace',original)
    records.finalize_stopped_run(tmp_path,'a'*64,cancelled=False,error='lock',returncode=1)
    assert read(canonical)==final
    assert raw.read_bytes()==b'existing result bytes'


@pytest.mark.parametrize('cancelled,status', [(True,'cancelled'),(False,'failed')])
def test_forced_exit_recovers_last_snapshot_and_raw(tmp_path,cancelled,status):
    records.publish_progress(tmp_path,{'request_sha256':'a'*64,'status':'starting','steps':[1,2],'versions':{'x':'y'}})
    (tmp_path/'raw.png').write_bytes(b'completed before exit')
    final=records.finalize_stopped_run(tmp_path,'a'*64,cancelled=cancelled,error='stopped',returncode=-1)
    assert final['status']==status and final['steps']==[1,2] and final['versions']=={'x':'y'}
    assert final['raw_preserved'] and final['metrics_incomplete']
    assert read(tmp_path/'run.json')==read(tmp_path/'run.final.json')


@pytest.mark.parametrize('outcome', ['completed','failed','cancelled'])
def test_worker_finally_writes_terminal_without_models(tmp_path,monkeypatch,outcome):
    import genai_lab.qwen_pose_worker as worker
    records.write_json(tmp_path/'request.json',{'cpu_only':True})
    def fake_generate(request,directory,record):
        record['steps']=list(range(40))
        records.publish_progress(directory,record)
        if outcome=='failed': raise ValueError('fake failure')
        if outcome=='cancelled': raise worker.WorkerCancelled('user cancel')
        (directory/'raw.png').write_bytes(b'fake result')
    monkeypatch.setattr(worker,'generate',fake_generate)
    assert worker.main([str(tmp_path/'request.json')])==(0 if outcome=='completed' else 1)
    final=read(tmp_path/'run.json')
    assert final['status']==outcome and final['steps']==list(range(40))
    assert final==read(tmp_path/'run.final.json')
