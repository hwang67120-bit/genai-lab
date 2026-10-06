"""CPU-only advisory contracts: frozen metrics, request isolation, GUI and cancellation."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QObject, Signal, QProcess
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from genai_lab import tail_complexity as complexity
from genai_lab.qwen_preservation import file_sha, json_sha
from genai_lab.qwen_tail_edit import make_tail_request
from genai_lab.qwen_tail_gui import TailInputDialog
from genai_lab.tail_complexity_gui import TailComplexityRunner
from genai_lab.tail_complexity_worker import analyze_image
from test_qwen_tail import assets, spec_at, runtime


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


class FakeRunner(QObject):
    completed = Signal(dict)
    def __init__(self):
        super().__init__()
        self.starts = []
        self.cancels = 0
    def start(self, *args):
        self.starts.append(args)
    def cancel(self):
        self.cancels += 1


def measured(box, sha):
    return dict(status='completed', advisory_only=True, box=list(box), source_sha256=sha,
                color_complex=True, color_families=[{}, {}], deep_valleys=4,
                turn_deg=160, thickness_px=20, pattern=True, curvy=True,
                display_deferred=False, deferral_reason=None)


def wait_until(predicate, timeout=5000):
    end = time.monotonic() + timeout/1000
    while not predicate() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(.02)
    assert predicate()


def test_color_families_keep_v3_thresholds():
    rgb = np.full((100, 100, 3), 180, np.uint8)
    mask = np.ones((100, 100), bool)
    assert len(complexity.color_families(rgb, mask)) == 1
    rgb[:, :30] = (255, 0, 0)
    assert len(complexity.color_families(rgb, mask)) == 2
    rgb[:] = (0, 0, 0)
    assert complexity.color_families(rgb, mask) == []  # v3 excludes dark achromatic pixels.


def test_mask_clips_box_and_preserves_five_percent_components():
    rgb = np.full((100, 100, 3), 180, np.uint8)
    mask = np.zeros((100, 100), bool)
    mask[10:70, 10:20] = True
    mask[80:85, 10:20] = True
    mask[90:92, 10:12] = True
    mask[:, 95:] = True
    record, kept = complexity.measure_tail(rgb, mask, (0, 0, 90, 100))
    assert record['mask_px'] == 650
    assert record['main_px'] == 600
    assert not kept[:, 95:].any()
    assert not kept[90:92].any()


@pytest.mark.parametrize('turn,deep,curvy,pattern', [(119.9, 3, False, True), (120, 2, True, False), (120, 3, True, True)])
def test_frozen_pattern_and_curve_thresholds(monkeypatch, turn, deep, curvy, pattern):
    rgb = np.full((100, 100, 3), 180, np.uint8)
    monkeypatch.setattr(complexity, 'centerline_v2', lambda mask: dict(t=10, length=100, step=10, win=5, turn=turn, fine=[]))
    prominence = np.array([20.0]*deep + [19.9])
    monkeypatch.setattr(complexity, 'profile', lambda *a: (None, None, list(range(len(prominence))), prominence))
    record, _ = complexity.measure_tail(rgb, np.ones((100, 100), bool), (10, 10, 90, 90))
    assert record['curvy'] == curvy and record['pattern'] == pattern
    assert record['deep_valleys'] == deep


def test_round_blob_defers_display_without_altering_measurements(monkeypatch):
    rgb = np.full((100, 100, 3), 180, np.uint8)
    monkeypatch.setattr(complexity, 'centerline_v2', lambda m: dict(t=30, length=60, step=15, win=5, turn=180, fine=[]))
    monkeypatch.setattr(complexity, 'profile', lambda *a: (None, None, [0,1,2], np.array([21,22,23])))
    record, _ = complexity.measure_tail(rgb, np.ones((100, 100), bool), (10,10,90,90))
    assert record['pattern'] and record['curvy']
    assert record['display_deferred'] and record['deferral_validated'] is False
    text = complexity.advisory_text(record)
    assert '판단 보류' in text and '검증되지 않았습니다' in text
    assert '반복 무늬 있음' not in text and '꼬리 고치기를 권장' not in text


def test_empty_mask_is_failure():
    with pytest.raises(ValueError, match='마스크'):
        complexity.measure_tail(np.zeros((100,100,3), np.uint8), np.zeros((100,100), bool), (10,10,90,90))


def test_analysis_does_not_modify_source_or_reference(tmp_path):
    source, _ = assets(tmp_path)
    sha = file_sha(source)
    class Segmenter:
        snapshot = 'local/revision'
        def segment(self, image, box):
            assert image.size == (936,2048)
            mask = np.zeros((image.height,image.width), bool)
            mask[1150:1650,650:740] = True
            return mask, .8
    record = analyze_image(Segmenter(), source, sha, (560,1100,936,1700), tmp_path/'output')
    assert file_sha(source) == sha
    assert record['device'] == 'cpu' and record['advisory_only']
    assert Path(record['overlay_path']).is_file()
    with pytest.raises(ValueError, match='변경'):
        analyze_image(Segmenter(), source, 'a'*64, (560,1100,936,1700), tmp_path/'bad')


def test_advisory_on_off_keeps_entire_request_and_prompt_identical(tmp_path, app):
    spec = spec_at(tmp_path)
    settings = runtime(tmp_path)
    runner = FakeRunner()
    dialog = TailInputDialog(spec.source_path, spec.source_sha256, settings_path='runtime.json', complexity_runner=runner)
    try:
        dialog.canvas.box = spec.box
        dialog.pattern.setCurrentIndex(1)
        dialog.confirm.setChecked(True)
        request_before = make_tail_request(spec, settings=settings)
        dialog.complexity_finished(measured(spec.box, spec.source_sha256))
        saved = complexity.persist_advisory(tmp_path/'complexity-on', spec, dialog.complexity_result)
        request_after = make_tail_request(spec, settings=settings)
        assert saved['advisory_only']
        assert json.dumps(request_before, sort_keys=True).encode() == json.dumps(request_after, sort_keys=True).encode()
        assert json_sha(request_before) == json_sha(request_after)
        assert request_before['prompt']['positive'].encode() == request_after['prompt']['positive'].encode()
        assert request_before['prompt']['positive_sha256'] == request_after['prompt']['positive_sha256']
        assert dialog.pattern.currentData() is True and dialog.tip.text() == ''
        assert not dialog.recognition_enabled.isChecked()
        assert dialog.confirm.isChecked() and dialog.run.isEnabled()
        dialog.complexity_enabled.setChecked(False)
        complexity.persist_advisory(tmp_path/'complexity-off', spec, dialog.complexity_result)
        assert make_tail_request(spec, settings=settings) == request_before
    finally:
        dialog.close()


def test_analysis_keeps_unselected_pattern_and_empty_tip(tmp_path, app):
    source, _ = assets(tmp_path)
    runner = FakeRunner()
    dialog = TailInputDialog(source, file_sha(source), complexity_runner=runner)
    try:
        dialog.canvas.box = (560,1100,936,1700)
        dialog.complexity_finished(measured(dialog.canvas.box, file_sha(source)))
        assert dialog.pattern.currentData() is None
        assert dialog.tip.text() == ''
        assert not dialog.run.isEnabled()
    finally:
        dialog.close()


@pytest.mark.parametrize('status', ['failed', 'timeout'])
def test_analysis_failure_does_not_block_existing_edit_controls(tmp_path, app, status):
    source, _ = assets(tmp_path)
    dialog = TailInputDialog(source, file_sha(source), settings_path='runtime.json', complexity_runner=FakeRunner())
    try:
        dialog.canvas.box = (560,1100,936,1700)
        dialog.pattern.setCurrentIndex(2)
        dialog.confirm.setChecked(True)
        dialog.complexity_finished(dict(status=status, advisory_only=True, error='model missing'))
        assert '분석 불가' in dialog.complexity_hint.text()
        assert dialog.run.isEnabled() and dialog.confirm.isChecked()
    finally:
        dialog.close()


def test_new_selection_rejects_old_result_and_closing_cancels(tmp_path, app):
    source, _ = assets(tmp_path)
    runner = FakeRunner()
    dialog = TailInputDialog(source, file_sha(source), complexity_runner=runner)
    dialog.canvas.box = (560,1100,936,1700)
    old = measured(dialog.canvas.box, file_sha(source))
    dialog.canvas.selectionChanged.emit()
    wait_until(lambda: len(runner.starts) == 1)
    dialog.canvas.box = (600,1150,900,1650)
    dialog.canvas.selectionChanged.emit()
    dialog.complexity_finished(old)
    assert dialog.complexity_result['status'] == 'pending'
    dialog.close()
    assert not dialog.complexity_timer.isActive()
    assert runner.cancels >= 3


def test_stale_advisory_not_attached_to_edit(tmp_path):
    spec = spec_at(tmp_path)
    wrong = measured((0,0,10,10), spec.source_sha256)
    record = complexity.persist_advisory(tmp_path/'advisory', spec, wrong)
    assert record['status'] == 'unavailable'


def test_parent_run_records_advisory_without_changing_worker_final_or_request(tmp_path):
    directory = tmp_path/'run'; directory.mkdir()
    (directory/'launcher.json').write_text(json.dumps({'status':'awaiting_review','request_sha256':'a'*64}))
    for file in ['request.json', 'run.json', 'run.final.json']:
        (directory/file).write_text('{}')
    before = {name: (directory/name).read_bytes() for name in ['request.json','run.json','run.final.json']}
    complexity.annotate_finished_run(directory, {'status':'completed','advisory_only':True})
    assert json.loads((directory/'launcher.json').read_text())['tail_complexity']['advisory_only']
    assert all((directory/name).read_bytes() == value for name,value in before.items())


def test_subprocess_is_offline_cpu_and_gui_remains_responsive(tmp_path, app):
    script = tmp_path/'worker.py'
    script.write_text("""import os,sys,json,time,hashlib
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads(p.read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['HF_HUB_OFFLINE']=='1'
time.sleep(.15)
sha=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
# Use the actual project hash function to retain its canonical serialization.
from genai_lab.qwen_preservation import json_sha
(p.parent/'result.json').write_text(json.dumps({'status':'completed','request_sha256':json_sha(r),'advisory_only':True}))
""", encoding='utf-8')
    # A -c shim keeps the repository import path available, unlike running a temp script directly.
    runner = TailComplexityRunner(program=sys.executable, arguments=['-c', 'exec(open('+repr(str(script))+', encoding="utf-8").read())'])
    results = []
    runner.completed.connect(results.append)
    try:
        started = time.monotonic()
        runner.start('synthetic.png', 'a'*64, (0,0,10,10))
        assert time.monotonic()-started < 1
        wait_until(lambda: bool(results))
        assert results[0]['status'] == 'completed'
        assert runner.process is None
    finally:
        runner.close()


@pytest.mark.parametrize('operation', ['timeout', 'cancel'])
def test_running_cpu_process_is_stopped(tmp_path, app, operation):
    runner = TailComplexityRunner(timeout_ms=200, program=sys.executable,
                                  arguments=['-c', 'import time; time.sleep(30)'])
    results = []
    runner.completed.connect(results.append)
    try:
        runner.start('synthetic.png','a'*64,(0,0,10,10))
        process = runner.process
        wait_until(lambda: process.state() == QProcess.ProcessState.Running)
        if operation == 'cancel':
            runner.cancel()
            end = time.monotonic() + .25
            while time.monotonic() < end:
                QApplication.processEvents()
                time.sleep(.02)
            assert not results
        else:
            wait_until(lambda: bool(results))
            assert results[0]['status'] == 'timeout'
        assert runner.process is None
    finally:
        runner.close()


def test_missing_python_is_advisory_failure(tmp_path, app):
    runner = TailComplexityRunner(program=str(tmp_path/'missing-python'))
    results = []
    runner.completed.connect(results.append)
    try:
        runner.start('synthetic.png','a'*64,(0,0,10,10))
        wait_until(lambda: bool(results))
        assert results[0]['status'] == 'failed'
    finally:
        runner.close()


def test_controller_keeps_same_request_with_or_without_advisory(tmp_path, monkeypatch, app):
    from gui_main import GenAILabWindow
    import genai_lab.qwen_tail_gui as gui
    import genai_lab.qwen_tail_edit as edit
    import genai_lab.studio_controller as controller
    from test_qwen_tail import simple_batch
    source, _ = assets(tmp_path/'source')
    window = GenAILabWindow()
    window.studio.analysis = {'source':str(source), 'references':{'character':{'sha256':file_sha(source)}},
                              'groups':{'fixed':['tail']}}
    window.studio.generated(simple_batch(tmp_path/'batch'))
    stable_spec = spec_at(tmp_path/'stable', source=source, basis=window.studio.results.candidate.path)
    class Input:
        settings_path = 'runtime.json'
        settings = runtime(tmp_path)
        canvas = SimpleNamespace(box=stable_spec.box)
        pattern = SimpleNamespace(currentData=lambda:True)
        tip = SimpleNamespace(text=lambda:'')
        confirm = SimpleNamespace(isChecked=lambda:True)
        recognition_enabled = SimpleNamespace(isChecked=lambda:False)
        complexity_result = {'status':'disabled', 'advisory_only':True}
        def __init__(self, *a, **kw): pass
        def exec(self): return QDialog.DialogCode.Accepted
    monkeypatch.setattr(gui, 'TailInputDialog', Input)
    # Same approved snapshot isolates advisory ON/OFF from intentionally unique run paths.
    monkeypatch.setattr(edit, 'prepare_tail_spec', lambda *a, **k: stable_spec)
    requests = []
    monkeypatch.setattr(window.studio, 'launch_tail_edit', lambda spec, settings: requests.append(make_tail_request(spec, settings=settings)))
    try:
        window.studio.edit_tail()
        assert window.studio.tail_context['tail_complexity']['status'] == 'disabled'
        Input.complexity_result = measured(stable_spec.box, stable_spec.source_sha256)
        window.studio.edit_tail()
        assert window.studio.tail_context['tail_complexity']['status'] == 'completed'
        assert len(requests) == 2
        assert json_sha(requests[0]) == json_sha(requests[1])
        assert requests[0]['prompt'] == requests[1]['prompt']
        assert 'tail_complexity' not in requests[1] and 'tail_complexity' not in requests[1]['tail_spec']
    finally:
        window.close()


def test_missing_cached_model_fails_offline_without_download(monkeypatch):
    import os
    import huggingface_hub
    from genai_lab.tail_complexity_worker import CpuTailSegmenter
    def local_snapshot(model_id, **kwargs):
        assert model_id == 'facebook/sam2.1-hiera-tiny'
        assert kwargs['local_files_only'] is True
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
        raise FileNotFoundError('no local model')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', local_snapshot)
    with pytest.raises(FileNotFoundError, match='no local model'):
        CpuTailSegmenter('missing-cache')


def test_missing_advisory_overlay_does_not_block_edit(tmp_path):
    spec = spec_at(tmp_path)
    report = measured(spec.box, spec.source_sha256)
    report['overlay_path'] = str(tmp_path/'absent.png')
    record = complexity.persist_advisory(tmp_path/'advisory', spec, report)
    assert 'overlay' in record['artifact_errors']
    assert record['advisory_only'] is True
