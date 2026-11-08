"""PyQt5 desktop demo for one-image PatchCore inspection.

依赖 PyQt5（pyproject 的 demo extra 与之对应）。PyQt6 的 API 差异（exec/exec_、枚举作用域）
本文件没有兼容层，装 PyQt6 是跑不起来的。
"""
import os
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace
from pathlib import Path

import PyQt5

# Keep the demo self-contained even when another Python/Qt installation is on
# PATH (this machine also has Anaconda's Qt DLLs).
_qt_root = Path(PyQt5.__file__).parent / "Qt5"
_qt_bin = _qt_root / "bin"
os.environ["QT_PLUGIN_PATH"] = str(_qt_root / "plugins")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_root / "plugins" / "platforms")
os.environ["PATH"] = str(_qt_bin) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_qt_bin))

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

ALIGN_CENTER = Qt.AlignCenter
KEEP_ASPECT = Qt.KeepAspectRatio
SMOOTH = Qt.SmoothTransformation


class InferWorker(QObject):
    done = pyqtSignal(str, float, str, str)
    failed = pyqtSignal(str)

    def __init__(self, bank_path, image_path, weights_path, tmpdir):
        super().__init__()
        self.bank_path, self.image_path, self.weights_path = bank_path, image_path, weights_path
        self.tmpdir = tmpdir

    def run(self):
        try:
            import torch
            import classify
            classify._need_torch()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bank, meta = classify.load_bank(self.bank_path, device)
            weights = classify._pick_weights(SimpleNamespace(weights=self.weights_path or None), meta)
            model = classify._pcdemo.Features(weights_path=weights,
                                               imagenet_norm=bool(meta.get("imagenet_norm", False)))
            model = model.to(device).eval()
            # 与 classify.cmd_infer 同口径：预处理/裁剪/平滑/邻域填充都要从 meta 取，
            # 否则非默认记忆库（letterbox / aspect_crop / 非 2.0 sigma）会静默错分。
            patch, score = classify.image_patch_scores(
                model, bank, self.image_path, int(meta["size"]), device,
                bool(meta.get("neighbors", False)), meta.get("score_agg", "max"),
                meta.get("preprocess", "stretch"), float(meta.get("aspect_crop", 0.0)),
                float(meta.get("blur_sigma", 2.0)), meta.get("neighbor_pad", "zero"))
            verdict = classify.verdict(score, float(meta["thr_ok"]), float(meta["thr_ng"]))
            # 固定文件名写在本次会话的临时目录里，退出时整目录删掉；
            # 原来每次点「开始检测」都 mkstemp 一个新文件、从不删除，会在 %TEMP% 里越积越多。
            heat_path = os.path.join(self.tmpdir, "heatmap.png")
            classify.save_heatmap(patch, heat_path, image_path=self.image_path,
                                  size=int(meta["size"]))
            self.done.emit(verdict, score, heat_path,
                           f"OK ≤ {meta['thr_ok']:.3f}  |  NG ≥ {meta['thr_ng']:.3f}")
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore 工业缺陷检测演示")
        self.resize(1050, 720)
        self.thread = None
        self.worker = None
        self._tmpdir = tempfile.mkdtemp(prefix="patchcore_demo_")
        self._make_ui()

    def _make_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        form = QFormLayout()
        self.bank = QLineEdit()
        self.image = QLineEdit()
        self.weights = QLineEdit()
        form.addRow("记忆库 (.npz)", self._browse_row(self.bank, "bank"))
        form.addRow("待检测图片", self._browse_row(self.image, "image"))
        form.addRow("权重文件（可留空，使用记忆库记录的路径）", self._browse_row(self.weights, "weights"))
        layout.addLayout(form)
        self.run_button = QPushButton("开始检测")
        self.run_button.clicked.connect(self.infer)
        layout.addWidget(self.run_button)
        self.result = QLabel("选择记忆库和图片后开始检测")
        self.result.setAlignment(ALIGN_CENTER)
        self.result.setStyleSheet("font-size: 22px; padding: 12px")
        layout.addWidget(self.result)
        images = QHBoxLayout()
        self.original = QLabel("原图")
        self.heatmap = QLabel("异常热力图")
        for label in (self.original, self.heatmap):
            label.setAlignment(ALIGN_CENTER)
            label.setMinimumSize(480, 420)
            label.setStyleSheet("border: 1px solid #b8c2cc; background: #f4f6f8")
            images.addWidget(label)
        layout.addLayout(images)
        self.setCentralWidget(root)

    @staticmethod
    def _browse_row(edit, kind):
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton("浏览…")
        button.clicked.connect(lambda: MainWindow._choose(edit, kind))
        row.addWidget(button)
        return holder

    @staticmethod
    def _choose(edit, kind):
        filters = {"bank": "Bank (*.npz)", "image": "Images (*.png *.jpg *.jpeg *.bmp)",
                   "weights": "Weights (*.pth *.pt *.bin)"}
        path, _ = QFileDialog.getOpenFileName(None, "选择文件", edit.text(), filters[kind])
        if path:
            edit.setText(path)

    def infer(self):
        bank, image = self.bank.text().strip(), self.image.text().strip()
        if not os.path.isfile(bank) or not os.path.isfile(image):
            QMessageBox.warning(self, "缺少文件", "请选择有效的记忆库和待检测图片。")
            return
        self.run_button.setEnabled(False)
        self.result.setText("正在提取特征并检索记忆库…")
        self.original.setPixmap(QPixmap(image).scaled(self.original.size(), KEEP_ASPECT, SMOOTH))
        self.thread = QThread(self)
        worker = InferWorker(bank, image, self.weights.text().strip(), self._tmpdir)
        self.worker = worker
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.done.connect(self._done)
        worker.failed.connect(self._failed)
        worker.done.connect(self.thread.quit)
        worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self._clear_worker)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _clear_worker(self):
        self.worker = None
        self.thread = None

    def _done(self, verdict, score, path, thresholds):
        colors = {"OK": "#16803c", "REVIEW": "#b7791f", "NG": "#c53030"}
        self.result.setText(f"{verdict}    异常分数 {score:.3f}    {thresholds}")
        self.result.setStyleSheet(f"font-size: 22px; padding: 12px; color: {colors.get(verdict, '#222')}")
        self.heatmap.setPixmap(QPixmap(path).scaled(self.heatmap.size(), KEEP_ASPECT, SMOOTH))
        self.run_button.setEnabled(True)

    def closeEvent(self, event):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().closeEvent(event)

    def _failed(self, message):
        self.run_button.setEnabled(True)
        self.result.setText("检测失败，请检查权重、记忆库配置和依赖。")
        QMessageBox.critical(self, "推理错误", message[-3000:])


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
