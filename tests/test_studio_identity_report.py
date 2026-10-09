import json
from types import SimpleNamespace
from pathlib import Path
from PySide6.QtWidgets import QApplication
from genai_lab.studio_generation import StudioResults, digest
from genai_lab.identity_report import compare_measurements
from genai_lab import studio_identity_report as gui
from test_studio_generation import batch_at, analysis_at
from gui_main import GenAILabWindow


class PendingTask:
    def __init__(self,*args):
        self.result=None;self.error=None
        self.stop=SimpleNamespace(set=lambda:None)
        self.finished=SimpleNamespace(connect=lambda f:None)
    def start(self):pass
    def deleteLater(self):pass
    def wait(self,n):return True


def test_pending_measurement_never_blocks_approval_export(tmp_path,monkeypatch):
    monkeypatch.setattr(gui,'MeasurementTask',PendingTask)
    app=QApplication.instance() or QApplication([]);w=GenAILabWindow()
    try:
        s=w.studio;s.runtime=object();s.analysis=analysis_at(tmp_path/'inputs')
        s.results=StudioResults(batch_at(tmp_path/'case'))
        s.controls()
        assert s.identity_report_controller.task is not None
        assert w.approve_candidate_button.isEnabled()
        s.results.approve({'character':True,'garment':True,'exposure':True,'appendages':True});s.controls()
        assert w.save_candidate_button.isEnabled()
        s.results.export(tmp_path/'saved.png')
        info=json.loads((tmp_path/'saved.review.json').read_text())
        assert info['identity_report']['status']=='pending_or_unavailable'
    finally:w.close()


def test_late_report_is_bound_to_its_image_not_new_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(gui,'MeasurementTask',PendingTask)
    app=QApplication.instance() or QApplication([]);w=GenAILabWindow()
    try:
        s=w.studio;s.runtime=object();s.analysis=analysis_at(tmp_path/'inputs')
        s.results=StudioResults(batch_at(tmp_path/'case'));s.controls()
        controller=s.identity_report_controller;old_sha=digest(s.results.current_path)
        path=tmp_path/'identity-report.json';path.write_text('{}')
        controller.task.result={'path':str(path),'sha256':digest(path),'result_sha256':old_sha,
            'report':compare_measurements({'B2':1},{'B2':1.2})}
        s.select(1);controller.finished()
        assert s.results.identity_report_record['status']=='pending_or_unavailable'
        assert not w.identity_report_button.isEnabled()
        s.select(0)
        assert '20.0%' in w.identity_report_label.text()
        assert w.identity_report_button.isEnabled()
        s.results.approve({'character':True,'garment':True,'exposure':True,'appendages':True})
        s.results.export(tmp_path/'measured.png')
        assert json.loads((tmp_path/'measured.review.json').read_text())['identity_report']['sha256']==digest(path)
        path.write_text('changed')
        assert s.results.identity_report_record['status']=='changed'
    finally:w.close()
