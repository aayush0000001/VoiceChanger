"""One module per page. Views are presentation only: they emit pyqtSignals for user intent and expose slots
(``update_status``, ``apply_params``, ...) that the main window connects to the EngineBridge."""

from .devices_view import DevicesView
from .live_view import LiveView
from .settings_view import SettingsView
from .voice_view import VoiceView

__all__ = ["LiveView", "VoiceView", "DevicesView", "SettingsView"]
