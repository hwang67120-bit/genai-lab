"""작업실 사용자 화면 시작(생성 없음)에서 열리는 파일 기록. 화면 없이 띄우고 3초 뒤 닫는다."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, r"Z:\genai-lab")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.pop("GENAI_STUDIO_DEVELOPER", None)
with open(os.environ["GENAI_OPEN_TRACE"], "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(stage="S_startup")) + "\n")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
import gui_main  # noqa: E402

app = QApplication(sys.argv)
window = gui_main.GenAILabWindow()
window.show()
print("developer_mode", window.developer_mode, "product", window.studio.product is not None, flush=True)
QTimer.singleShot(3000, app.quit)
app.exec()
with open(os.environ["GENAI_OPEN_TRACE"], "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(stage="end")) + "\n")
