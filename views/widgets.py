"""Reusable, theme-aware building blocks shared by the pages. Pure presentation: no engine, no business logic."""

from __future__ import annotations

import time
from typing import List, Optional

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QContextMenuEvent, QFont, QFontMetrics, QKeyEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CardWidget,
    ComboBox,
    DropDownPushButton,
    FluentIcon,
    GroupHeaderCardWidget,
    HeaderCardWidget,
    IconWidget,
    InfoBar,
    InfoBarIcon,
    InfoBarPosition,
    MenuAnimationType,
    RoundMenu,
    ScrollArea,
    Slider,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    TitleLabel,
    TransparentToolButton,
    getFont,
    isDarkTheme,
    qconfig,
    themeColor,
)

from styles import (
    CARD_PADDING,
    CARD_RADIUS,
    CONTENT_MAX_WIDTH,
    GAP,
    GAP_SM,
    PAGE_MARGIN,
    PageStyle,
    danger_color,
    stroke_color,
    success_color,
    text_primary,
    text_secondary,
    text_tertiary,
    track_color,
    warning_color,
)


# ======================================================================================================
# small helpers
# ======================================================================================================

def muted(label: CaptionLabel) -> CaptionLabel:
    """Secondary text colour, light/dark aware."""
    label.setTextColor(QColor(96, 96, 96), QColor(206, 206, 206))
    return label


def icon_by_name(name: str, default=FluentIcon.MICROPHONE):
    """A FluentIcon looked up by member name (the bridge sends names, not UI objects)."""
    return getattr(FluentIcon, name, default)


def set_switch(switch: SwitchButton, checked: bool) -> None:
    """Moves a switch to ``checked`` without emitting checkedChanged."""
    if switch.isChecked() != checked:
        switch.blockSignals(True)
        switch.setChecked(checked)
        switch.blockSignals(False)


def tighten_group_card(card: GroupHeaderCardWidget) -> GroupHeaderCardWidget:
    """GroupHeaderCardWidget re-spaced to the 8 px grid with the same 8 px corners as every other card."""
    card.setBorderRadius(CARD_RADIUS)
    card.headerLayout.setContentsMargins(CARD_PADDING, 0, CARD_PADDING, 0)
    card.headerView.setFixedHeight(48)
    return card


class _ElideMixin:
    """One-line label text that ends in an ellipsis instead of forcing its card (and the page) wider. The full
    text stays in the tooltip. Mix in before the label class."""

    def _start_eliding(self, text: str) -> None:
        self._full = ""
        self._mode = Qt.TextElideMode.ElideRight
        self.setText(text)

    def setElideMode(self, mode: Qt.TextElideMode) -> None:
        """ElideMiddle keeps the end of a file path (the file name) visible."""
        self._mode = mode
        self._elide()

    def fullText(self) -> str:
        return self._full

    def setText(self, text: str) -> None:
        self._full = text
        self._elide()
        self.updateGeometry()  # sizeHint follows the full text, not the elided one

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        shown = metrics.elidedText(self._full, self._mode, max(self.width() - 2, 24))
        if shown != self.text():
            QLabel.setText(self, shown)
        self.setToolTip(self._full if shown != self._full else "")

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def sizeHint(self) -> QSize:
        return QSize(QFontMetrics(self.font()).horizontalAdvance(self._full) + 4, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(40, super().minimumSizeHint().height())


class ElidedCaption(_ElideMixin, CaptionLabel):
    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        CaptionLabel.__init__(self, parent)  # (the str overload re-enters __init__ with the wrong arguments in subclasses)
        self._start_eliding(text)


class ElidedStrongLabel(_ElideMixin, StrongBodyLabel):
    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        StrongBodyLabel.__init__(self, parent)
        self._start_eliding(text)


def tidy_group_row(group) -> None:
    """Row inside a GroupHeaderCardWidget: 16 px sides, 12 px top/bottom, 16 px icon gap, captions that elide."""
    group.hBoxLayout.setContentsMargins(CARD_PADDING, GAP, CARD_PADDING, GAP)
    group.hBoxLayout.setSpacing(CARD_PADDING)
    # swap the caption for one that cannot set the minimum width of the whole page
    caption = muted(ElidedCaption(group.contentLabel.text()))
    group.textLayout.replaceWidget(group.contentLabel, caption)
    group.contentLabel.hide()
    group.contentLabel.deleteLater()
    group.contentLabel = caption
    group.titleLabel.setMinimumWidth(80)


# ======================================================================================================
# cards
# ======================================================================================================

class Card(HeaderCardWidget):
    """HeaderCardWidget on the 8 px grid; add content to ``self.body`` (a vertical layout)."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)  # (the str overload re-enters __init__ with the wrong arguments in subclasses)
        self.setTitle(title)
        self.setBorderRadius(CARD_RADIUS)
        self.headerLayout.setContentsMargins(CARD_PADDING, 0, CARD_PADDING, 0)
        self.headerView.setFixedHeight(48)
        self.viewLayout.setContentsMargins(CARD_PADDING, CARD_PADDING, CARD_PADDING, CARD_PADDING)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(GAP)
        self.viewLayout.addLayout(self.body)


class GroupCard(GroupHeaderCardWidget):
    """GroupHeaderCardWidget (icon + title + caption + control rows) with Fluent 2 spacing."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setTitle(title)
        tighten_group_card(self)

    def add_row(self, icon, title: str, caption: str, widget: QWidget, stretch: int = 0):
        group = self.addGroup(icon, title, caption, widget, stretch)
        tidy_group_row(group)
        return group


# ======================================================================================================
# page scaffold
# ======================================================================================================

class PageBase(ScrollArea):
    """A scrolling page: title + caption header (with a slot for actions on the right) over a centred column."""

    def __init__(self, object_name: str, title: str, caption: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.canvas = QWidget()
        self.canvas.setObjectName("pageContent")
        self.setWidget(self.canvas)

        outer = QHBoxLayout(self.canvas)
        outer.setContentsMargins(PAGE_MARGIN, 0, PAGE_MARGIN, 0)
        outer.setSpacing(0)
        self.column = QWidget(self.canvas)
        self.column.setMaximumWidth(CONTENT_MAX_WIDTH)
        outer.addStretch(1)
        outer.addWidget(self.column, 100)
        outer.addStretch(1)

        self.content = QVBoxLayout(self.column)
        self.content.setContentsMargins(0, GAP_SM, 0, PAGE_MARGIN)
        self.content.setSpacing(GAP)

        self.titleLabel = TitleLabel(title)
        self.captionLabel = muted(CaptionLabel(caption))
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.setContentsMargins(0, 0, 0, 0)
        titles.addWidget(self.titleLabel)
        titles.addWidget(self.captionLabel)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(GAP)
        self.actions.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, GAP_SM)
        header.setSpacing(GAP)
        header.addLayout(titles, 1)
        header.addLayout(self.actions)
        self.content.addLayout(header)

        PageStyle.PAGE.apply(self)

    def add_section(self, text: str) -> StrongBodyLabel:
        label = StrongBodyLabel(text)
        label.setContentsMargins(2, GAP_SM, 0, 0)
        self.content.addWidget(label)
        return label

    def finish(self) -> None:
        """Call after the last widget has been added so spare height goes below the content, not between cards."""
        self.content.addStretch(1)


class ReflowGrid(QWidget):
    """Equal-width columns: as many as fit at ``min_item_width`` (at most ``max_columns``), re-flowed on resize."""

    def __init__(self, min_item_width: int, max_columns: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._min_w = min_item_width
        self._max_cols = max_columns
        self._items: List[QWidget] = []
        self._cols = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(GAP)

    def add(self, widget: QWidget) -> None:
        widget.setParent(self)
        self._items.append(widget)
        self._place(force=True)

    def clear(self) -> None:
        while self._grid.count():
            self._grid.takeAt(0)
        for w in self._items:
            w.setParent(None)
            w.deleteLater()
        self._items.clear()
        self._cols = 0

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._place()

    def _place(self, force: bool = False) -> None:
        if not self._items:
            return
        width = self.width() if self.width() > 20 else 10_000
        cols = max(1, min(self._max_cols, len(self._items), (width + GAP) // (self._min_w + GAP)))
        if cols == self._cols and not force:
            return
        self._cols = cols
        while self._grid.count():
            self._grid.takeAt(0)
        for i, w in enumerate(self._items):
            self._grid.addWidget(w, i // cols, i % cols)
        for c in range(self._max_cols):
            self._grid.setColumnStretch(c, 1 if c < cols else 0)
        self.updateGeometry()


# ======================================================================================================
# notifications
# ======================================================================================================

class Toasts:
    """
    Transient InfoBars at the window's top-right, without the library's slide and fade animations. Those
    repaint a large part of the window for 200 ms per toast and measurably delay audio blocks while the
    engine is live, so a toast simply appears, stays and disappears.
    """

    TOP = 56      # just below the 48 px title bar
    MARGIN = 16

    def __init__(self, window: QWidget):
        self._window = window
        self._bars: List[InfoBar] = []

    def show(self, level: str, title: str, text: str, duration_ms: int = 4000) -> None:
        icon = {
            "success": InfoBarIcon.SUCCESS,
            "error": InfoBarIcon.ERROR,
            "warning": InfoBarIcon.WARNING,
        }.get(level, InfoBarIcon.INFORMATION)
        bar = InfoBar(icon, title, text, Qt.Orientation.Horizontal, True, -1, InfoBarPosition.NONE, self._window)
        bar.setGraphicsEffect(None)  # InfoBar keeps an opacity effect attached, which forces offscreen rendering
        bar.closedSignal.connect(lambda b=bar: self._forget(b))
        self._bars.append(bar)
        bar.show()  # sizes itself to its text
        self.relayout()
        bar.raise_()
        QTimer.singleShot(duration_ms, lambda b=bar: self._expire(b))

    def relayout(self) -> None:
        y = self.TOP
        for bar in self._bars:
            bar.move(max(self._window.width() - bar.width() - self.MARGIN, 0), y)
            y += bar.height() + GAP_SM

    def _forget(self, bar: InfoBar) -> None:
        if bar in self._bars:
            self._bars.remove(bar)
            self.relayout()

    def _expire(self, bar: InfoBar) -> None:
        if bar in self._bars:
            bar.close()


# ======================================================================================================
# painted chips
# ======================================================================================================

class Chip(QWidget):
    """Small rounded label (hotkeys, EXPERIMENTAL). ``accent`` tints it with the theme colour."""

    def __init__(self, text: str, accent: bool = False, min_width: int = 0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._accent = accent
        self._min_width = min_width
        self._font = getFont(12, QFont.Weight.DemiBold)
        self._text = ""
        self.setFixedHeight(22)
        self.setText(text)
        qconfig.themeChangedFinished.connect(self.update)

    def setText(self, text: str) -> None:
        self._text = text
        self.setFixedWidth(max(self._min_width, QFontMetrics(self._font).horizontalAdvance(text) + 16))
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._accent:
            c = themeColor()
            fill = QColor(c.red(), c.green(), c.blue(), 46)
            stroke = QColor(c.red(), c.green(), c.blue(), 120)
            fg = c
        else:
            fill = QColor(255, 255, 255, 14) if isDarkTheme() else QColor(0, 0, 0, 10)
            stroke = stroke_color()
            fg = text_secondary()
        p.setBrush(fill)
        p.setPen(QPen(stroke, 1))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(fg)
        p.setFont(self._font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)


class StatusPill(QWidget):
    """Rounded status chip: coloured dot + text. kind: ok | idle | busy | error."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._font = getFont(13, QFont.Weight.DemiBold)
        self._text = ""
        self._kind = "idle"
        self.setFixedHeight(32)
        self.set_status("Stopped", "idle")
        qconfig.themeChangedFinished.connect(self.update)

    def set_status(self, text: str, kind: str) -> None:
        if text == self._text and kind == self._kind:
            return
        self._text, self._kind = text, kind
        self.setFixedWidth(QFontMetrics(self._font).horizontalAdvance(text) + 46)
        self.update()

    def _dot(self) -> QColor:
        return {
            "ok": success_color(),
            "busy": warning_color(),
            "error": danger_color(),
        }.get(self._kind, text_tertiary())

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dot = self._dot()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(QColor(dot.red(), dot.green(), dot.blue(), 38 if self._kind != "idle" else 20))
        p.setPen(QPen(stroke_color(), 1))
        p.drawRoundedRect(r, CARD_RADIUS, CARD_RADIUS)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        p.drawEllipse(QPointF(18, self.height() / 2), 4.5, 4.5)
        p.setPen(text_primary())
        p.setFont(self._font)
        p.drawText(
            QRectF(30, 0, self.width() - 38, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._text,
        )


# ======================================================================================================
# meters
# ======================================================================================================

class MeterBar(QWidget):
    """Horizontal bar: track, fill, optional hold tick. Painted by hand; repaints only when something moved."""

    HEIGHT = 16

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fraction = 0.0
        self._zone = "ok"
        self._hold: Optional[float] = None
        qconfig.themeChangedFinished.connect(self.update)

    def set_state(self, fraction: float, zone: str = "ok", hold: Optional[float] = None) -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        if hold is not None:
            hold = min(max(hold, 0.0), 1.0)
        same_hold = (hold is None and self._hold is None) or (
            hold is not None and self._hold is not None and abs(hold - self._hold) < 0.004
        )
        if zone == self._zone and abs(fraction - self._fraction) < 0.004 and same_hold:
            return
        self._fraction, self._zone, self._hold = fraction, zone, hold
        self.update()

    def _fill(self) -> QColor:
        return {"warn": warning_color(), "danger": danger_color()}.get(self._zone, themeColor())

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = min(CARD_RADIUS, r.height() / 2)

        track = QPainterPath()
        track.addRoundedRect(r, radius, radius)
        p.setPen(Qt.PenStyle.NoPen)
        p.fillPath(track, track_color())

        if self._fraction > 0.0:
            p.save()
            p.setClipPath(track)
            p.setBrush(self._fill())
            # left cap is clipped away; only the leading edge is rounded
            p.drawRoundedRect(
                QRectF(r.left() - radius, r.top(), r.width() * self._fraction + radius, r.height()), radius, radius
            )
            p.restore()

        if self._hold is not None and self._hold > 0.01:
            x = r.left() + r.width() * self._hold
            p.setBrush(text_primary())
            p.drawRoundedRect(QRectF(x - 1.5, r.top() + 3, 3, r.height() - 6), 1.5, 1.5)


class LevelMeter(QWidget):
    """Title + dBFS readout over a MeterBar. Fast attack, ~0.2 s release (same ballistics as before), 1.2 s peak hold."""

    FLOOR_DB = -65.0
    DECAY_PER_60HZ_TICK = 0.82
    HOLD_SECONDS = 1.2
    TEXT_INTERVAL = 0.12  # the bar moves every tick; the number is only readable ~8 times a second anyway

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.titleLabel = BodyLabel(title)
        self.valueLabel = muted(CaptionLabel("-80.0 dBFS"))
        self.valueLabel.setFixedWidth(84)
        self.valueLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar = MeterBar()
        self._level = -80.0
        self._hold = -80.0
        self._hold_at = 0.0
        self._clock = time.perf_counter()
        self._text = ""
        self._text_at = 0.0

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.titleLabel)
        top.addStretch(1)
        top.addWidget(self.valueLabel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(GAP_SM - 2)
        lay.addLayout(top)
        lay.addWidget(self.bar)
        self._render()

    def set_level(self, db: float) -> None:
        now = time.perf_counter()
        dt = min(now - self._clock, 0.25)
        self._clock = now
        if db > self._level:
            self._level = db
        else:
            k = self.DECAY_PER_60HZ_TICK ** (dt * 60.0)
            self._level = self._level * k + db * (1.0 - k)
        if db >= self._hold:
            self._hold, self._hold_at = db, now
        elif now - self._hold_at > self.HOLD_SECONDS:
            self._hold = max(self._level, self._hold - 45.0 * dt)
        self._render()

    def reset(self) -> None:
        self._level = self._hold = -80.0
        self._render(force_text=True)

    def _render(self, force_text: bool = False) -> None:
        span = -self.FLOOR_DB
        fraction = (self._level - self.FLOOR_DB) / span
        hold = (self._hold - self.FLOOR_DB) / span if self._hold > self.FLOOR_DB + 3.0 else None
        zone = "danger" if self._level > -3.0 else ("warn" if self._level > -9.0 else "ok")
        self.bar.set_state(fraction, zone, hold)
        now = self._clock
        if force_text or now - self._text_at >= self.TEXT_INTERVAL:
            text = f"{max(self._level, -80.0):5.1f} dBFS"
            if text != self._text:
                self._text = text
                self._text_at = now
                self.valueLabel.setText(text)


class StatBlock(QWidget):
    """Caption over a large value with an optional detail line (latency numbers)."""

    def __init__(self, caption: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.captionLabel = muted(CaptionLabel(caption))
        self.valueLabel = SubtitleLabel("-")
        self.detailLabel = muted(CaptionLabel(""))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.captionLabel)
        lay.addWidget(self.valueLabel)
        lay.addWidget(self.detailLabel)
        self._value = "-"
        self._detail = ""

    def set(self, value: str, detail: str = "") -> None:
        if value != self._value:
            self._value = value
            self.valueLabel.setText(value)
        if detail != self._detail:
            self._detail = detail
            self.detailLabel.setText(detail)


# ======================================================================================================
# controls
# ======================================================================================================

class SliderRow(QWidget):
    """Title on the left, live value on the right, a Fluent slider underneath. Emits (name, real value)."""

    valueChanged = pyqtSignal(str, float)

    def __init__(
        self,
        name: str,
        title: str,
        minimum: float,
        maximum: float,
        scale: float,
        unit: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.name, self.scale, self.unit = name, scale, unit
        self._signed = minimum < 0.0 < maximum
        self._silent = False

        self.titleLabel = BodyLabel(title)
        self.valueLabel = muted(CaptionLabel("0"))
        self.valueLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.valueLabel.setMinimumWidth(80)
        self.slider = Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum * scale)), int(round(maximum * scale)))
        self.slider.setAccessibleName(title)
        self.slider.valueChanged.connect(self._on_slider)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.titleLabel)
        top.addStretch(1)
        top.addWidget(self.valueLabel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(GAP_SM - 2)
        lay.addLayout(top)
        lay.addWidget(self.slider)

    def _format(self, value: float) -> str:
        if self.unit == "%":
            return f"{int(round(value * 100))}%"
        if self.scale == 1.0:
            return f"{int(round(value))} {self.unit}"
        return f"{value:+.1f} {self.unit}" if self._signed else f"{value:.1f} {self.unit}"

    def set_value(self, value: float) -> None:
        """Moves the slider to a value the engine reported (does not emit)."""
        self._silent = True
        try:
            self.slider.setValue(int(round(value * self.scale)))
        finally:
            self._silent = False
        self.valueLabel.setText(self._format(value))

    def _on_slider(self, raw: int) -> None:
        value = raw / self.scale
        self.valueLabel.setText(self._format(value))
        if not self._silent:
            self.valueChanged.emit(self.name, value)


class ElidedComboBox(ComboBox):
    """ComboBox that elides a long device name in the middle ("... [Windows WASAPI]" stays visible)."""

    def __init__(self, width: int = 360, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = ""
        self.setFixedWidth(width)

    def setText(self, text: str) -> None:  # ComboBoxBase calls this whenever the current item changes
        self._full = text
        self._elide()

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        shown = metrics.elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(self.width() - 52, 40))
        QPushButton.setText(self, shown)
        self.setToolTip(self._full if shown != self._full else "")

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def sizeHint(self) -> QSize:
        return QSize(self.width(), 33)


class QuietDropDownButton(DropDownPushButton):
    """DropDownPushButton whose menu simply appears: the stock slide-down animation repaints while audio may be live."""

    def _showMenu(self) -> None:
        menu = self.menu()
        if not menu:
            return
        menu.view.setMinimumWidth(self.width())
        menu.view.adjustSize()
        menu.adjustSize()
        x = -menu.width() // 2 + menu.layout().contentsMargins().left() + self.width() // 2
        menu.exec(self.mapToGlobal(QPoint(x, self.height())), aniType=MenuAnimationType.NONE)


# ======================================================================================================
# preset tile
# ======================================================================================================

class PresetTile(CardWidget):
    """
    Clickable voice-preset card: hotkey chip, name, character tag, one-line description, 'active' check. A custom
    (player-made) tile shows a 'Custom' chip instead of a hotkey and a "..." button with Rename, Export and Delete.
    """

    chosen = pyqtSignal(str)
    renameRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)
    _selected = False

    TOP_ROW = 24          # tiles in one row line up whether or not they carry the "..." button
    MORE_SIZE = 24

    def __init__(
        self,
        key: str,
        title: str,
        tag: str,
        hotkey: str,
        blurb: str,
        icon,
        custom: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.key = key
        self.custom = custom
        self._menu: Optional[RoundMenu] = None
        self.setBorderRadius(CARD_RADIUS)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(152)
        self.setAccessibleName(f"{title}, custom voice preset" if custom else f"{title} voice preset, {hotkey}")

        self.keyChip = Chip(hotkey, accent=custom)
        self.iconWidget = IconWidget(icon)
        self.iconWidget.setFixedSize(20, 20)
        self.titleLabel = ElidedStrongLabel(title)
        self.tagLabel = muted(ElidedCaption(tag))
        self.blurbLabel = muted(CaptionLabel(blurb))
        self.blurbLabel.setWordWrap(True)

        top_host = QWidget()
        top_host.setFixedHeight(self.TOP_ROW)
        top = QHBoxLayout(top_host)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.keyChip)
        top.addStretch(1)
        self.moreButton: Optional[TransparentToolButton] = None
        if custom:
            self.moreButton = TransparentToolButton(FluentIcon.MORE, self)
            self.moreButton.setFixedSize(self.MORE_SIZE, self.MORE_SIZE)
            self.moreButton.setToolTip("Rename, export or delete")
            self.moreButton.setAccessibleName(f"{title} options")
            self.moreButton.clicked.connect(self._open_menu_at_button)
            top.addWidget(self.moreButton)
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        names.addWidget(self.titleLabel)
        names.addWidget(self.tagLabel)
        identity = QHBoxLayout()
        identity.setContentsMargins(0, 0, 0, 0)
        identity.setSpacing(GAP)
        identity.addWidget(self.iconWidget, 0, Qt.AlignmentFlag.AlignTop)
        identity.addLayout(names, 1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(CARD_PADDING, GAP, CARD_PADDING, GAP)
        lay.setSpacing(GAP_SM)
        lay.addWidget(top_host)
        lay.addLayout(identity)
        lay.addWidget(self.blurbLabel)
        lay.addStretch(1)

        self.clicked.connect(self._emit)

    # ------------------------------------------------------------------ the "..." menu of a custom tile
    def _build_menu(self) -> RoundMenu:
        if self._menu is None:
            menu = RoundMenu(parent=self)
            menu.addAction(Action(FluentIcon.EDIT, "Rename…", triggered=lambda: self.renameRequested.emit(self.key)))
            menu.addAction(Action(FluentIcon.SHARE, "Export…", triggered=lambda: self.exportRequested.emit(self.key)))
            menu.addSeparator()
            menu.addAction(Action(FluentIcon.DELETE, "Delete…", triggered=lambda: self.deleteRequested.emit(self.key)))
            self._menu = menu
        return self._menu

    def _open_menu(self, global_pos: QPoint) -> None:
        # no slide animation: it would repaint while the audio engine may be live
        self._build_menu().exec(global_pos, aniType=MenuAnimationType.NONE)

    def _open_menu_at_button(self) -> None:
        button = self.moreButton
        self._open_menu(button.mapToGlobal(QPoint(0, button.height() + 2)))

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        if self.custom:
            self._open_menu(e.globalPos())
            e.accept()
        else:
            super().contextMenuEvent(e)

    def _emit(self) -> None:
        self.chosen.emit(self.key)

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self._updateBackgroundColor()
        self.update()

    # selected tiles get an accent wash on top of the normal card fill
    def _wash(self, alpha: int) -> QColor:
        c = themeColor()
        return QColor(c.red(), c.green(), c.blue(), alpha)

    def _normalBackgroundColor(self) -> QColor:
        return self._wash(34) if self._selected else super()._normalBackgroundColor()

    def _hoverBackgroundColor(self) -> QColor:
        return self._wash(46) if self._selected else super()._hoverBackgroundColor()

    def _pressedBackgroundColor(self) -> QColor:
        return self._wash(26) if self._selected else super()._pressedBackgroundColor()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._emit()
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        accent = themeColor()
        if self._selected:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(accent, 2))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), CARD_RADIUS, CARD_RADIUS)
            # check badge, top right (left of the "..." button on a custom tile)
            shift = self.MORE_SIZE + GAP_SM if self.custom else 0
            c = QPointF(self.width() - CARD_PADDING - 10 - shift, GAP + self.TOP_ROW / 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(accent)
            p.drawEllipse(c, 10, 10)
            p.setPen(QPen(QColor(255, 255, 255) if accent.lightness() < 170 else QColor(0, 0, 0), 2,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            tick = QPainterPath()
            tick.moveTo(c.x() - 4.5, c.y() + 0.5)
            tick.lineTo(c.x() - 1.2, c.y() + 3.6)
            tick.lineTo(c.x() + 4.6, c.y() - 3.2)
            p.drawPath(tick)
        if self.hasFocus() and not self._selected:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(text_primary(), 1.5))
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), CARD_RADIUS, CARD_RADIUS)
