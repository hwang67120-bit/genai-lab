"""생성·승인과 분리해 참조 측정을 실행한다. 주 작업을 막지 않는다."""
from pathlib import Path
import threading
import uuid
from PySide6.QtCore import QObject, QThread, QEvent
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QTableWidget, QTableWidgetItem
from genai_lab.identity_report import summary
from genai_lab.identity_measure_service import measure_pair
from genai_lab.proportion_inputs import sha


class MeasurementTask(QThread):
    def __init__(self, original, result, directory, runtime, parent):
        super().__init__(parent)
        self.args=(original,result,directory,runtime);self.stop=threading.Event()
        self.result=None;self.error=None

    def run(self):
        try:self.result=measure_pair(*self.args,cancelled=self.stop.is_set)
        except Exception as error:self.error=str(error)


class IdentityReportController(QObject):
    def __init__(self, owner):
        super().__init__(owner.window)
        self.owner=owner;self.task=None;self.pending=None;self.active=None;self.reports={}
        owner.window.installEventFilter(self)
        owner.window.identity_report_button.clicked.connect(self.details)

    def refresh(self):
        owner=self.owner;w=owner.window;results=owner.results
        visible=results is not None
        w.identity_report_label.setVisible(visible);w.identity_report_button.setVisible(visible)
        w.identity_report_button.setEnabled(False)
        if not visible:
            self.pending=None
            if self.task:self.task.stop.set()
            return
        if owner.runtime is None or not owner.analysis or 'directory' not in owner.analysis:
            w.identity_report_label.setText('측정 준비 안 됨 · 승인·저장은 가능합니다.');return
        original=Path(owner.analysis['directory'])/'character.png';result=results.current_path
        try:key=(str(results.batch.directory),sha(original),sha(result))
        except OSError:
            w.identity_report_label.setText('측정할 이미지를 찾지 못했습니다.');return
        if key in self.reports:
            info=self.reports[key]
            w.identity_report_label.setText(summary(info['report']) if 'report' in info else '측정 불가 · '+info['error'])
            w.identity_report_button.setEnabled('report' in info)
            return
        w.identity_report_label.setText('체형·머리색 측정 중 · 기다리지 않고 승인·저장할 수 있습니다.')
        self.pending=(key,original,result,results,owner.runtime)
        if self.task is None:self.start_pending()

    def start_pending(self):
        if self.pending is None:return
        self.active=self.pending;self.pending=None
        key,original,result,results,runtime=self.active
        folder=results.batch.directory/'identity-measurements'/uuid.uuid4().hex
        task=MeasurementTask(original,result,folder,runtime,self)
        self.task=task;task.finished.connect(self.finished);task.start()

    def finished(self):
        task=self.task;key,original,result,results,runtime=self.active
        if task.error:info={'error':task.error}
        else:info=task.result
        self.reports[key]=info
        if info and 'report' in info:
            results.identity_reports[key[2]]={k:info[k] for k in ('path','sha256','result_sha256')}
            # 현재 선택한 원본만 저장한다. 이미 내보낸 부속 기록을 덮어쓰지 않는다.
            if self.owner.results is results:
                try:results.persist()
                except OSError:pass
        self.task=None;self.active=None;task.deleteLater()
        self.refresh()

    def details(self):
        results=self.owner.results
        if results is None:return
        current=sha(results.current_path)
        info=next((v for k,v in self.reports.items() if k[0]==str(results.batch.directory) and k[2]==current and 'report' in v),None)
        if info is None:return
        dialog=QDialog(self.owner.window);dialog.setWindowTitle('체형·머리색 측정 · 참고용');dialog.resize(900,720)
        layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel('옷 포함 폭은 실제 몸 두께가 아닙니다. 숫자로 자동 승인하거나 결과를 숨기지 않습니다.'))
        table=QTableWidget(9,4);table.setHorizontalHeaderLabels(['항목','원본','결과','차이 / 측정 불가 이유'])
        for i,(key,row) in enumerate(info['report']['items'].items()):
            reason_names={'exposure_mismatch':'맨살과 옷 포함 조건이 다름','measurement_unavailable':'관절 또는 영역을 확인하지 못함',
                'no_valid_side':'팔다리 영역을 나누지 못함','merge':'다른 부위와 겹침','arm_touch':'측정 불가(팔과 겹침)', 'arm_merge':'팔과 몸통이 붙음',
                'hair_skin_inseparable':'머리카락과 피부를 구분하지 못함','hair_measurement_unavailable':'머리색 표본 부족'}
            difference=reason_names.get(row['reason'],'측정 불가: '+str(row['reason'])) if row['reason'] else (
                f"{row['difference']*100:.1f}%" if key!='H5' else str(row['difference']))
            label=row['label'] + (' (맨살)' if row.get('original_bare') is True else ' (옷 포함)' if row.get('original_bare') is False else '')
            vals=[label,row['original'],row['result'],difference]
            if key=='H5':vals[1:3]=['색 분포','색 분포']
            for j,v in enumerate(vals):table.setItem(i,j,QTableWidgetItem(str(v) if v is not None else '측정 불가'))
        layout.addWidget(table)
        scroll=QScrollArea();label=QLabel();label.setPixmap(QPixmap(info['report']['overlay']['path']))
        scroll.setWidget(label);layout.addWidget(scroll);dialog.exec()

    def eventFilter(self, obj, event):
        if event.type()==QEvent.Type.Close and self.task is not None:
            self.task.stop.set()
            if not self.task.wait(2500):
                event.ignore();return True
        return super().eventFilter(obj,event)
