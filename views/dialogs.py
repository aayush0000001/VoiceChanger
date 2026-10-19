"""Modal dialogs: the problem report form, a plain confirmation and the preset name box. Presentation only: they
collect input, and the caller sends it to the EngineBridge through signals."""

from __future__ import annotations

import re
from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    MessageBoxBase,
    PlainTextEdit,
    ProgressBar,
    PushButton,
    ScrollArea,
    StrongBodyLabel,
    SubtitleLabel,
    TextEdit,
)

import preset_names
import reporting
from styles import GAP, GAP_SM, danger_color
from .widgets import muted


class FastMessageBox(MessageBoxBase):
    """
    MessageBoxBase without the library's fade-in/fade-out and blurred drop shadow. Those are offscreen-rendered
    effects that repaint the whole window for every animation frame (and the blur on every keystroke), which can
    delay a live audio block; a modal dialog should be free to open while the engine is running.
    """

    _closed = False

    def setShadowEffect(self, *args, **kwargs) -> None:  # called by MessageBoxBase.__init__
        pass

    def showEvent(self, e) -> None:
        self._closed = False
        QDialog.showEvent(self, e)

    def done(self, code: int) -> None:
        if self._closed:  # Enter can reach both the box and the default button: report the answer only once
            return
        self._closed = True
        QDialog.done(self, code)


class ConfirmDialog(FastMessageBox):
    """Title, a few lines of text, an optional check box, and a confirm / cancel pair."""

    def __init__(
        self,
        parent: QWidget,
        title: str,
        content: str,
        confirm_text: str = "OK",
        checkbox_text: Optional[str] = None,
    ):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel(title))
        body = BodyLabel(content)
        body.setWordWrap(True)
        self.viewLayout.addWidget(body)
        self.checkBox: Optional[CheckBox] = None
        if checkbox_text:
            self.checkBox = CheckBox(checkbox_text)
            self.viewLayout.addWidget(self.checkBox)
        self.yesButton.setText(confirm_text)
        self.widget.setMinimumWidth(420)
        self.widget.setMaximumWidth(520)

    def checked(self) -> bool:
        return bool(self.checkBox and self.checkBox.isChecked())


class NameDialog(FastMessageBox):
    """
    Asks for a custom preset's name (Save as preset, Rename). The confirm button stays off until the name can be used,
    and the line under the box says why not: empty, too long, a built-in's name, or already taken. The same rules
    (``preset_names``) are enforced again by the engine bridge.
    """

    def __init__(
        self,
        parent: QWidget,
        title: str,
        confirm_text: str,
        existing: Iterable[str],
        reserved: Iterable[str],
        initial: str = "",
        ignore: Optional[str] = None,
        adding: bool = True,
        caption: str = "",
    ):
        super().__init__(parent)
        self._existing = list(existing)
        self._reserved = list(reserved)
        self._ignore = ignore
        self._adding = adding
        self._problem: Optional[str] = "Give the preset a name."

        self.viewLayout.addWidget(SubtitleLabel(title))
        if caption:
            hint = muted(CaptionLabel(caption))
            hint.setWordWrap(True)
            self.viewLayout.addWidget(hint)
        self.edit = LineEdit()
        self.edit.setMaxLength(preset_names.NAME_MAX)
        self.edit.setClearButtonEnabled(True)
        self.edit.setAccessibleName("Preset name")
        self.edit.setText(initial)
        self.counter = muted(CaptionLabel(""))
        self.errorLabel = CaptionLabel("")
        color = danger_color()
        self.errorLabel.setTextColor(color, color)
        self.errorLabel.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.errorLabel, 1)
        row.addWidget(self.counter, 0, Qt.AlignmentFlag.AlignTop)
        self.viewLayout.addWidget(self.edit)
        self.viewLayout.addLayout(row)

        self.yesButton.setText(confirm_text)
        self.widget.setMinimumWidth(420)
        self.widget.setMaximumWidth(480)
        self.edit.textChanged.connect(self._on_changed)
        self.edit.returnPressed.connect(self._submit)
        self._on_changed()
        self.edit.selectAll()
        self.edit.setFocus()

    def name(self) -> str:
        return preset_names.normalize_name(self.edit.text())

    def problem(self) -> Optional[str]:
        return self._problem

    def _on_changed(self, *_args) -> None:
        name = self.name()
        self.counter.setText(f"{len(self.edit.text())}/{preset_names.NAME_MAX}")
        self._problem = preset_names.check_name(name, self._existing, self._reserved, self._ignore, self._adding)
        self.errorLabel.setText(self._problem if name and self._problem else "")
        self.yesButton.setEnabled(self._problem is None)

    def validate(self) -> bool:  # MessageBoxBase calls this before accepting, whichever way Save was triggered
        return self._problem is None

    def _submit(self) -> None:
        if self.yesButton.isEnabled():
            self.yesButton.click()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.edit.setFocus()


# ---------------------------------------------------------------------------------------------- preset files
PRESET_FILE_FILTER = "V-COMMS preset (*.json);;All files (*)"


def ask_preset_import_path(parent: QWidget) -> str:
    path, _ = QFileDialog.getOpenFileName(parent, "Import voice preset", "", PRESET_FILE_FILTER)
    return path


def ask_preset_export_path(parent: QWidget, name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "preset"
    path, _ = QFileDialog.getSaveFileName(parent, "Export voice preset", f"{safe}.json", PRESET_FILE_FILTER)
    return path


class ReportDialog(FastMessageBox):
    """
    Report a problem: type, one-line summary, description, optional reply-to email, an "Include diagnostics" check
    box and a read-only preview of exactly what will be sent. Send stays disabled-in-effect until the form is valid:
    pressing it with a problem shows the message and keeps the dialog open.
    """

    def __init__(self, parent: QWidget, diagnostics: str):
        super().__init__(parent)
        self._diagnostics = diagnostics

        self.titleLabel = SubtitleLabel("Report a problem")
        intro = muted(CaptionLabel("Nothing leaves your PC until you press Send. No audio and no files are included."))
        self.typeBox = ComboBox()
        for name in reporting.REPORT_TYPES:
            self.typeBox.addItem(name)
        self.typeBox.setFixedWidth(180)
        self.replyEdit = LineEdit()
        self.replyEdit.setPlaceholderText("Optional, so I can reply")
        self.replyEdit.setMaxLength(reporting.REPLY_TO_MAX)
        self.replyEdit.setClearButtonEnabled(True)
        self.summaryEdit = LineEdit()
        self.summaryEdit.setPlaceholderText("One line, e.g. Voice cuts out when I switch presets")
        self.summaryEdit.setMaxLength(reporting.SUMMARY_MAX)
        self.summaryCount = muted(CaptionLabel(f"0/{reporting.SUMMARY_MAX}"))
        self.descriptionEdit = TextEdit()
        self.descriptionEdit.setPlaceholderText("What happened, what you expected, and how to make it happen again.")
        self.descriptionEdit.setFixedHeight(96)
        self.diagBox = CheckBox("Include diagnostics")
        self.diagBox.setChecked(True)
        self.preview = PlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFixedHeight(130)
        mono = QFont("Cascadia Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPixelSize(12)
        self.preview.setFont(mono)
        self.errorLabel = CaptionLabel("")
        color = danger_color()
        self.errorLabel.setTextColor(color, color)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 14, 0)  # room for the scroll bar, which floats over the right edge
        column.setSpacing(GAP_SM)
        column.addWidget(self.titleLabel)
        column.addWidget(intro)

        top = QHBoxLayout()
        top.setSpacing(GAP)
        type_col = QVBoxLayout()
        type_col.setSpacing(4)
        type_col.addWidget(StrongBodyLabel("Type"))
        type_col.addWidget(self.typeBox)
        reply_col = QVBoxLayout()
        reply_col.setSpacing(4)
        reply_col.addWidget(StrongBodyLabel("Your email"))
        reply_col.addWidget(self.replyEdit)
        top.addLayout(type_col)
        top.addLayout(reply_col, 1)
        column.addLayout(top)

        summary_head = QHBoxLayout()
        summary_head.addWidget(StrongBodyLabel("Summary"))
        summary_head.addStretch(1)
        summary_head.addWidget(self.summaryCount)
        column.addLayout(summary_head)
        column.addWidget(self.summaryEdit)
        column.addWidget(StrongBodyLabel("Description"))
        column.addWidget(self.descriptionEdit)
        column.addWidget(self.diagBox)
        column.addWidget(StrongBodyLabel("Exactly what will be sent"))
        column.addWidget(self.preview)
        column.addWidget(self.errorLabel)

        body = QWidget()
        body.setObjectName("reportBody")
        body.setStyleSheet("#reportBody { background: transparent; }")
        body.setLayout(column)
        scroll = ScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.enableTransparentBackground()
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        room = max(parent.height() - 240, 280)
        scroll.setMaximumHeight(room)
        scroll.setMinimumHeight(min(room, 520))
        self.viewLayout.addWidget(scroll)

        self.copyButton = PushButton("Copy")
        self.copyButton.setToolTip("Copy the report text, for pasting into an email")
        self.copyButton.clicked.connect(self._copy)
        self.buttonLayout.insertWidget(0, self.copyButton, 0)
        self.yesButton.setText("Send report")
        self.widget.setMinimumWidth(620)
        self.widget.setMaximumWidth(700)

        for edit in (self.summaryEdit, self.replyEdit):
            edit.textChanged.connect(self._on_changed)
        self.descriptionEdit.textChanged.connect(self._on_changed)
        self.typeBox.currentIndexChanged.connect(self._on_changed)
        self.diagBox.stateChanged.connect(self._on_changed)
        self._on_changed()

    # ------------------------------------------------------------------ what the report will contain
    def report_type(self) -> str:
        return self.typeBox.currentText()

    def title_text(self) -> str:
        summary = self.summaryEdit.text().strip() or "<summary>"
        return reporting.build_title(self.report_type(), summary)

    def body_text(self) -> str:
        description = self.descriptionEdit.toPlainText().strip() or "<description>"
        diagnostics = self._diagnostics if self.diagBox.isChecked() else None
        return reporting.build_body(description, self.replyEdit.text(), diagnostics)

    def payload(self) -> dict:
        """What the window hands to the EngineBridge when the dialog was accepted."""
        return {
            "title": self.title_text(),
            "body": self.body_text(),
            "reply_to": self.replyEdit.text().strip(),
        }

    def _on_changed(self, *_args) -> None:
        count = len(self.summaryEdit.text())
        self.summaryCount.setText(f"{count}/{reporting.SUMMARY_MAX}")
        self.preview.setPlainText(reporting.report_text(self.title_text(), self.body_text()))
        self.errorLabel.setText("")

    def _copy(self) -> None:
        QApplication.clipboard().setText(reporting.report_text(self.title_text(), self.body_text()))
        self.errorLabel.setText("")

    # ------------------------------------------------------------------ Send
    def validate(self) -> bool:
        errors = reporting.validate(
            self.report_type(),
            self.summaryEdit.text(),
            self.descriptionEdit.toPlainText(),
            self.replyEdit.text(),
        )
        if not errors:
            return True
        field = next(f for f in ("summary", "description", "reply_to", "type") if f in errors)
        self.errorLabel.setText(errors[field])
        {"summary": self.summaryEdit, "description": self.descriptionEdit, "reply_to": self.replyEdit}.get(
            field, self.typeBox
        ).setFocus()
        return False


class HuBERTDownloadDialog(FastMessageBox):
    """Prompts the user to download the HuBERT ContentVec base model required for RVC voice models."""

    def __init__(self, parent: QWidget, target_model_name: str = ""):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel("AI Neural Voice Base Model Required"))
        desc = (
            f"The selected model ('{target_model_name}') is an RVC neural voice model.\n\n"
            "It requires the HuBERT ContentVec neural base model (~360 MB) for hardware-accelerated "
            "RTX / DirectML GPU conversion.\n\n"
            "Would you like V-COMMS to download it automatically now?"
        )
        self.bodyLabel = BodyLabel(desc)
        self.bodyLabel.setWordWrap(True)
        self.viewLayout.addWidget(self.bodyLabel)

        self.progressBar = ProgressBar(self)
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.viewLayout.addWidget(self.progressBar)

        self.statusLabel = CaptionLabel("")
        self.statusLabel.hide()
        self.viewLayout.addWidget(self.statusLabel)

        self.yesButton.setText("Download Now")
        self.cancelButton.setText("Cancel")
        self.widget.setMinimumWidth(460)
        self.widget.setMaximumWidth(540)
        try:
            self.yesButton.disconnect()
        except Exception:
            pass

    def set_downloading(self) -> None:
        self.bodyLabel.setText("Downloading HuBERT ContentVec neural base model (vec-768-layer-12.onnx)...\nThis is a one-time download.")
        self.yesButton.setEnabled(False)
        self.progressBar.show()
        self.statusLabel.show()

    def update_progress(self, pct: int, mb: float, total_mb: float) -> None:
        self.progressBar.setValue(pct)
        if total_mb > 0:
            self.statusLabel.setText(f"{mb:.1f} MB / {total_mb:.1f} MB ({pct}%)")
        else:
            self.statusLabel.setText(f"{mb:.1f} MB downloaded...")
