"""
Design tokens and the few QSS rules the Fluent pages need.

Layout follows Fluent 2 on an 8 px grid: 16 px page margins, 8-12 px gaps, 8 px corner radius on every container.
The QSS is kept in Python strings (not .qss files) so a PyInstaller build needs no extra data files.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtGui import QColor
from qfluentwidgets import StyleSheetBase, Theme, isDarkTheme

# --- spacing & shape ----------------------------------------------------------------------------------------
PAGE_MARGIN = 16          # outer margin of every page
GAP = 12                  # between cards and between rows inside a card
GAP_SM = 8                # tight pairs (label + value, icon + text)
CARD_RADIUS = 8           # every card, tile, meter track
CARD_PADDING = 16
CONTENT_MAX_WIDTH = 1040  # pages stop growing here and centre themselves on wide windows


# --- colours that must follow the theme but are not part of qfluentwidgets ------------------------------------
def text_secondary() -> QColor:
    return QColor(255, 255, 255, 200) if isDarkTheme() else QColor(0, 0, 0, 155)


def text_tertiary() -> QColor:
    return QColor(255, 255, 255, 139) if isDarkTheme() else QColor(0, 0, 0, 114)


def text_primary() -> QColor:
    return QColor(255, 255, 255) if isDarkTheme() else QColor(0, 0, 0, 228)


def track_color() -> QColor:
    return QColor(255, 255, 255, 22) if isDarkTheme() else QColor(0, 0, 0, 20)


def stroke_color() -> QColor:
    return QColor(255, 255, 255, 30) if isDarkTheme() else QColor(0, 0, 0, 30)


def success_color() -> QColor:
    return QColor("#6CCB5F") if isDarkTheme() else QColor("#107C10")


def warning_color() -> QColor:
    return QColor("#FCC000") if isDarkTheme() else QColor("#E8A100")


def danger_color() -> QColor:
    return QColor("#FF6369") if isDarkTheme() else QColor("#D13438")


# --- QSS ------------------------------------------------------------------------------------------------------
_PAGE_QSS = """
QScrollArea { background: transparent; border: none; }
#pageContent { background: transparent; }
"""


class PageStyle(StyleSheetBase, Enum):
    """Transparent page canvas, so the window's Mica backdrop shows through between the cards."""

    PAGE = "page"

    def path(self, theme=Theme.AUTO) -> str:  # the QSS lives in this module, there is no file
        return ""

    def content(self, theme=Theme.AUTO) -> str:
        return _PAGE_QSS
