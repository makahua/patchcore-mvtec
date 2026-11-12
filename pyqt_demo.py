"""PyQt5 desktop demo for one-image PatchCore inspection.

功能：选择记忆库 / 待检测图片 / 权重 → 异步推理 → 结果横幅（OK/REVIEW/NG）
     + 原图与热力图并排 + 结果明细表（可导出 CSV）+ 运行日志面板（同时写 logs/ 文件）
     + NG 声音告警与计数。

依赖 PyQt5（pyproject 的 demo extra 与之对应）。PyQt6 的 API 差异（exec/exec_、枚举作用域）
本文件没有兼容层，装 PyQt6 是跑不起来的。

用法：
    python pyqt_demo.py                     # 正常启动
    python pyqt_demo.py --selftest          # 离屏自检（不打开窗口，渲染一张截图后退出）
    python pyqt_demo.py --selftest --shot out.png --bank <npz> --image <png>
"""
import csv
import datetime as _dt
import logging
import os
import shutil
import sys
import tempfile
import time
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
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ALIGN_CENTER = Qt.AlignCenter
KEEP_ASPECT = Qt.KeepAspectRatio
SMOOTH = Qt.SmoothTransformation

LOG_DIR = Path(__file__).resolve().parent / "logs"
VERDICT_COLORS = {"OK": "#16803c", "REVIEW": "#b7791f", "NG": "#c53030"}
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"


def _setup_logging():
    """进程级日志：写文件 + 由界面上的 handler 转进日志面板。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("patchcore.demo")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(LOG_DIR / f"demo_{_dt.date.today():%Y%m%d}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


LOG = _setup_logging()


class _QtLogHandler(logging.Handler):
    """把 logging 记录转成 Qt 信号，避免工作线程直接碰控件。"""

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:  # 日志本身不能把程序带崩
            pass


class LogBridge(QObject):
    message = pyqtSignal(str)


class InferWorker(QObject):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, bank_path, image_path, weights_path, tmpdir, ng_ratio, play_alarm):
        super().__init__()
        self.bank_path, self.image_path, self.weights_path = bank_path, image_path, weights_path
        self.tmpdir, self.ng_ratio, self.play_alarm = tmpdir, float(ng_ratio), bool(play_alarm)

    def run(self):
        t0 = time.perf_counter()
        LOG.info("开始检测：%s", os.path.basename(self.image_path))
        try:
            import torch
            import classify
            classify._need_torch()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            LOG.info("设备：%s", device)

            t = time.perf_counter()
            bank, meta = classify.load_bank(self.bank_path, device)
            LOG.info("记忆库已载入 %.0f ms：layers=%s size=%s blur=%s neighbors=%s", (time.perf_counter() - t) * 1000,
                     meta.get("layers"), meta.get("size"), meta.get("blur_sigma"), meta.get("neighbors"))
            thr_ok = float(meta["thr_ok"])
            thr_ng = thr_ok * self.ng_ratio

            weights = classify._pick_weights(SimpleNamespace(weights=self.weights_path or None), meta)
            model = classify._pcdemo.Features(weights_path=weights,
                                              imagenet_norm=bool(meta.get("imagenet_norm", False)))
            model = model.to(device).eval()
            LOG.info("权重：%s", weights or "（内置默认）")

            # 与 classify.cmd_infer 同口径：预处理/裁剪/平滑/邻域填充都要从 meta 取，
            # 否则非默认记忆库（letterbox / aspect_crop / 非 2.0 sigma）会静默错分。
            t = time.perf_counter()
            patch, score = classify.image_patch_scores(
                model, bank, self.image_path, int(meta["size"]), device,
                bool(meta.get("neighbors", False)), meta.get("score_agg", "max"),
                meta.get("preprocess", "stretch"), float(meta.get("aspect_crop", 0.0)),
                float(meta.get("blur_sigma", 2.0)), meta.get("neighbor_pad", "zero"))
            infer_ms = (time.perf_counter() - t) * 1000
            verdict = classify.verdict(score, thr_ok, thr_ng)

            # 固定文件名写在本次会话的临时目录里，退出时整目录删掉；
            # 原来每次点「开始检测」都 mkstemp 一个新文件、从不删除，会在 %TEMP% 里越积越多。
            heat_path = os.path.join(self.tmpdir, "heatmap.png")
            classify.save_heatmap(patch, heat_path, image_path=self.image_path, size=int(meta["size"]))

            total_ms = (time.perf_counter() - t0) * 1000
            LOG.info("结果：%s  分数 %.3f  阈值 ok=%.3f ng=%.3f  推理 %.1f ms  总耗时 %.1f ms",
                     verdict, score, thr_ok, thr_ng, infer_ms, total_ms)
            if verdict == "NG":
                LOG.warning("检出 NG：%s", os.path.basename(self.image_path))
            self.done.emit({
                "verdict": verdict, "score": float(score), "thr_ok": thr_ok, "thr_ng": thr_ng,
                "heatmap": heat_path, "image": self.image_path, "infer_ms": infer_ms,
                "total_ms": total_ms, "play_alarm": self.play_alarm,
                "meta": {k: meta.get(k) for k in ("layers", "size", "blur_sigma", "neighbors", "thr_ok")},
            })
        except Exception:
            tb = traceback.format_exc()
            LOG.error("检测失败：%s", tb.rstrip().splitlines()[-1])
            self.failed.emit(tb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore 工业缺陷检测演示")
        self.resize(1180, 900)
        self.thread = None
        self.worker = None
        self._tmpdir = tempfile.mkdtemp(prefix="patchcore_demo_")
        self._rows = []
        self._counts = {"OK": 0, "REVIEW": 0, "NG": 0}

        self._bridge = LogBridge()
        self._bridge.message.connect(self._append_log)
        self._log_handler = _QtLogHandler(self._bridge)
        LOG.addHandler(self._log_handler)

        self._make_ui()
        LOG.info("界面就绪；日志目录 %s", LOG_DIR)

    # ---------- UI ----------
    def _make_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        form = QFormLayout()
        self.bank = QLineEdit()
        self.image = QLineEdit()
        self.weights = QLineEdit()
        form.addRow("记忆库 (.npz)", self._browse_row(self.bank, "bank"))
        form.addRow("待检测图片", self._browse_row(self.image, "image"))
        form.addRow("权重文件（留空则用记忆库记录的路径，再退回内置 timm 权重）", self._browse_row(self.weights, "weights"))
        default_weights = Path(__file__).resolve().parent / "torch_cache" / "wide_resnet50_2_timm.pth"
        if default_weights.is_file():
            self.weights.setText(str(default_weights))
        layout.addLayout(form)

        params = QHBoxLayout()
        params.addWidget(QLabel("灰区倍数"))
        self.ng_ratio = QDoubleSpinBox()
        self.ng_ratio.setRange(0.8, 3.0)
        self.ng_ratio.setSingleStep(0.1)
        self.ng_ratio.setValue(1.2)
        self.ng_ratio.setDecimals(2)
        self.ng_ratio.setToolTip("NG 阈值 = OK 阈值 × 该倍数；[OK, NG) 之间判为复核")
        params.addWidget(self.ng_ratio)
        self.alarm_sound = QCheckBox("NG 声音告警")
        self.alarm_sound.setChecked(True)
        params.addWidget(self.alarm_sound)
        params.addStretch(1)
        self.meta_label = QLabel("记忆库配置：—")
        self.meta_label.setStyleSheet("color:#486581")
        params.addWidget(self.meta_label)
        layout.addLayout(params)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("开始检测")
        self.run_button.clicked.connect(self.infer)
        buttons.addWidget(self.run_button)
        self.export_button = QPushButton("导出结果 CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        buttons.addWidget(self.export_button)
        buttons.addStretch(1)
        self.counter_label = QLabel("本次会话：OK 0 · REVIEW 0 · NG 0")
        self.counter_label.setStyleSheet("color:#486581")
        buttons.addWidget(self.counter_label)
        layout.addLayout(buttons)

        self.result = QLabel("选择记忆库和图片后开始检测")
        self.result.setAlignment(ALIGN_CENTER)
        self.result.setStyleSheet("font-size: 22px; padding: 12px; font-weight: 600")
        layout.addWidget(self.result)

        images = QHBoxLayout()
        self.original = QLabel("原图")
        self.heatmap = QLabel("异常热力图")
        for label in (self.original, self.heatmap):
            label.setAlignment(ALIGN_CENTER)
            label.setMinimumSize(520, 340)
            label.setMaximumHeight(400)
            label.setStyleSheet("border: 1px solid #b8c2cc; background: #f4f6f8")
            images.addWidget(label)
        layout.addLayout(images)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["时间", "文件", "判定", "分数", "OK 阈值", "NG 阈值", "耗时 ms"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setFixedHeight(168)
        layout.addWidget(self.table)

        log_title = QLabel("运行日志")
        log_title.setStyleSheet("color:#243b53; font-weight:600")
        layout.addWidget(log_title)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFixedHeight(150)
        self.log_view.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_view)

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

    # ---------- 行为 ----------
    def infer(self):
        bank, image = self.bank.text().strip(), self.image.text().strip()
        if not os.path.isfile(bank) or not os.path.isfile(image):
            QMessageBox.warning(self, "缺少文件", "请选择有效的记忆库和待检测图片。")
            return
        self.run_button.setEnabled(False)
        self.result.setStyleSheet("font-size: 18px; padding: 12px; color:#334e68")
        self.result.setText("正在提取特征并检索记忆库…")
        self.original.setPixmap(QPixmap(image).scaled(self.original.size(), KEEP_ASPECT, SMOOTH))
        self.thread = QThread(self)
        worker = InferWorker(bank, image, self.weights.text().strip(), self._tmpdir,
                             self.ng_ratio.value(), self.alarm_sound.isChecked())
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

    def _done(self, payload):
        verdict = payload["verdict"]
        color = VERDICT_COLORS.get(verdict, "#222")
        self.result.setText(
            f"{verdict}    异常分数 {payload['score']:.3f}    "
            f"OK ≤ {payload['thr_ok']:.3f}  |  NG ≥ {payload['thr_ng']:.3f}    "
            f"（推理 {payload['infer_ms']:.1f} ms）")
        self.result.setStyleSheet(f"font-size: 22px; padding: 12px; font-weight: 600; color: {color}")
        self.heatmap.setPixmap(QPixmap(payload["heatmap"]).scaled(self.heatmap.size(), KEEP_ASPECT, SMOOTH))

        meta = payload.get("meta") or {}
        self.meta_label.setText("记忆库配置：layers={} size={} blur={} neighbors={}".format(
            meta.get("layers"), meta.get("size"), meta.get("blur_sigma"), meta.get("neighbors")))

        self._counts[verdict] = self._counts.get(verdict, 0) + 1
        self.counter_label.setText("本次会话：OK {} · REVIEW {} · NG {}".format(
            self._counts["OK"], self._counts["REVIEW"], self._counts["NG"]))

        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        row = [stamp, os.path.basename(payload["image"]), verdict, f"{payload['score']:.3f}",
               f"{payload['thr_ok']:.3f}", f"{payload['thr_ng']:.3f}", f"{payload['total_ms']:.0f}"]
        self._append_row(row, color)
        self.export_button.setEnabled(True)

        if verdict == "NG":
            self.setWindowTitle("⚠ NG 检出 — PatchCore 工业缺陷检测演示")
            if payload.get("play_alarm"):
                QApplication.beep()
        else:
            self.setWindowTitle("PatchCore 工业缺陷检测演示")
        if verdict == "REVIEW":
            self.setWindowTitle("复核 — PatchCore 工业缺陷检测演示")
        self.run_button.setEnabled(True)

    def _append_row(self, row, color):
        self._rows.append(row)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, value in enumerate(row):
            item = QTableWidgetItem(value)
            item.setTextAlignment(ALIGN_CENTER if c != 1 else (Qt.AlignVCenter | Qt.AlignLeft))
            if c == 2:
                item.setForeground(QColor(color))
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.table.setItem(r, c, item)
        # 表格高度跟随行数增长，避免只有一行时留一大片空白
        self.table.setFixedHeight(min(320, 40 + (r + 1) * 30))
        self.table.scrollToBottom()

    def _append_log(self, text):
        self.log_view.appendPlainText(text)

    def export_csv(self):
        if not self._rows:
            return
        default = _dt.datetime.now().strftime("inspection_results_%Y%m%d_%H%M.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", default, "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["时间", "文件", "判定", "分数", "OK 阈值", "NG 阈值", "耗时 ms"])
                writer.writerows(self._rows)
            LOG.info("已导出 %d 条结果 → %s", len(self._rows), path)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)
        LOG.removeHandler(self._log_handler)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        super().closeEvent(event)

    def _failed(self, message):
        self.run_button.setEnabled(True)
        self.result.setStyleSheet("font-size: 20px; padding: 12px; color:#c53030")
        self.result.setText("检测失败，请检查权重、记忆库配置和依赖。")
        QMessageBox.critical(self, "推理错误", message[-3000:])


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    selftest = "--selftest" in argv
    shot = None
    if "--shot" in argv:
        shot = argv[argv.index("--shot") + 1]
    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(argv)
    window = MainWindow()
    window.show()

    if selftest:
        for flag, attr in (("--bank", "bank"), ("--image", "image"), ("--weights", "weights")):
            if flag in argv:
                getattr(window, attr).setText(argv[argv.index(flag) + 1])
        if window.bank.text() and window.image.text():
            window.infer()
            deadline = time.time() + 300
            while time.time() < deadline:
                app.processEvents()
                if window.thread is None:
                    break
                time.sleep(0.05)
        if shot:
            app.processEvents()
            window.repaint()
            app.processEvents()
            ok = window.grab().save(os.path.abspath(shot))
            LOG.info("自检截图%s：%s", "已保存" if ok else "保存失败", os.path.abspath(shot))
        app.processEvents()
        window.close()
        return 0

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
