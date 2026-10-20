"""reporting, appdata (folders and the v1.0.2 migration), diagnostics, applog, winintegration and the two model
downloaders (against a local HTTP server)."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import logging
import os
import random
import string
import sys
import threading
import time
import urllib.error
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import pytest

import appdata
import applog
import diagnostics
import engine as eng
import presets as pr
import report_config
import reporting
import winintegration as wi
from conftest import ITERATIONS, SEEDS


# ------------------------------------------------------------------------------------------------ reporting
@pytest.mark.parametrize("seed", SEEDS)
def test_validate_report_form(seed):
    rnd = random.Random(seed)
    summary = "".join(rnd.choice(string.ascii_letters + " ") for _ in range(rnd.randint(0, 100)))
    desc = "".join(rnd.choice(string.ascii_letters + " \n") for _ in range(rnd.randint(0, 4100)))
    reply = rnd.choice(["", "me@example.com", "not-an-email", "a@b", "x@y.co", " spaced@ex.org "])
    rtype = rnd.choice(list(reporting.REPORT_TYPES) + ["Other"])
    errors = reporting.validate(rtype, summary, desc, reply)
    assert ("type" in errors) == (rtype not in reporting.REPORT_TYPES)
    assert ("summary" in errors) == (not summary.strip() or len(summary.strip()) > reporting.SUMMARY_MAX)
    assert ("description" in errors) == (not desc.strip() or len(desc.strip()) > reporting.DESCRIPTION_MAX)
    assert ("reply_to" in errors) == (bool(reply.strip()) and not reporting.valid_email(reply))


@pytest.mark.parametrize("seed", SEEDS)
def test_mailto_link_always_fits_and_returns(seed):
    """Japanese / emoji text used to loop forever here (and the report thread held the GIL the whole time)."""
    rnd = random.Random(seed)
    alphabets = ["abc def ", "日本語のテキスト", "ü€ß", "😀🎮🔊", "%&?=#+ \n"]
    for _ in range(20):
        alphabet = rnd.choice(alphabets)
        title = reporting.build_title("Bug", "".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 80))))
        body = "".join(rnd.choice(rnd.choice(alphabets)) for _ in range(rnd.randint(0, 6000)))
        t0 = time.perf_counter()
        url = reporting.mailto_url("dev@example.com", title, body)
        assert time.perf_counter() - t0 < 1.0
        assert url.startswith("mailto:dev@example.com?subject=") and len(url) <= reporting.MAILTO_URL_MAX
    assert len(reporting.mailto_url("a@b.cd", "題" * 3000, "本" * 3000)) <= reporting.MAILTO_URL_MAX
    short = reporting.mailto_url("a@b.cd", "t", "short body")
    assert short.endswith("body=short%20body")


class FakeResponse:
    def __init__(self, status=200, body=b"{}"):
        self.status, self._body = status, body

    def read(self, n=-1):
        return self._body

    def getcode(self):
        return self.status

    def close(self):
        pass


@pytest.mark.parametrize("seed", SEEDS)
def test_post_report_outcomes(seed):
    rnd = random.Random(seed)
    sent = []

    def opener_for(result):
        def opener(request, timeout):
            sent.append(request)
            if isinstance(result, Exception):
                raise result
            return result
        return opener

    ok_cases = [FakeResponse(200, b'{"ok": true}'), FakeResponse(201, b""), FakeResponse(200, b"not json")]
    bad_cases = [FakeResponse(500), FakeResponse(200, b'{"ok": false}'), FakeResponse(200, b'{"errors": ["x"]}'),
                 urllib.error.HTTPError("u", 403, "no", {}, None), urllib.error.URLError("dns"), TimeoutError()]
    reply = rnd.choice(["", "me@example.com"])
    reporting.post_report("https://formspree.io/f/abc", "T", "B" * rnd.randint(0, 20000), reply,
                          opener=opener_for(rnd.choice(ok_cases)))
    request = sent[-1]
    body = request.data.decode()
    assert request.get_method() == "POST" and "_subject=T" in body and len(body) < reporting.BODY_MAX * 3 + 500
    assert ("_replyto" in body) == bool(reply)
    with pytest.raises(reporting.ReportError):
        reporting.post_report("https://formspree.io/f/abc", "T", "B", "", opener=opener_for(rnd.choice(bad_cases)))
    with pytest.raises(reporting.ReportError):
        reporting.post_report("http://insecure.example", "T", "B", "", opener=opener_for(FakeResponse()))


def test_rate_limiter_and_submit():
    now = [1000.0]
    limiter = reporting.RateLimiter(interval=60, clock=lambda: now[0])
    first = reporting.submit("T", "B", "", limiter)  # the shipped config is a placeholder: mail fallback
    assert first["status"] == "fallback" and first["reason"] == "no_key" and first["mailto"].startswith("mailto:")
    assert reporting.submit("T", "B", "", limiter) == {"status": "rate_limited", "wait_s": 60}
    now[0] += 59.5
    assert reporting.submit("T", "B", "", limiter)["wait_s"] == 1
    now[0] += 1
    sent = reporting.submit("T", "B", "", limiter, endpoint="https://formspree.io/f/x",
                            opener=lambda r, t: FakeResponse(200, b'{"ok": true}'))
    assert sent == {"status": "sent"}
    now[0] += 61

    def explode(*_a):
        raise ValueError("surprise")

    failed = reporting.submit("T", "B", "", limiter, endpoint="https://formspree.io/f/x", opener=explode)
    assert failed["status"] == "fallback" and "ValueError" in failed["detail"] and failed["text"].startswith("Subject: T")
    assert reporting.is_placeholder(report_config.FORM_ENDPOINT)
    assert reporting.is_placeholder_address(report_config.MAILTO_ADDRESS)
    assert not reporting.is_placeholder_address("dev@realmail.com")
    assert reporting._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example") is None


# ------------------------------------------------------------------------------------------------ appdata
def test_folders_live_in_appdata(isolated_home):
    assert appdata.roaming_dir() == isolated_home / "Roaming" / "V-COMMS"
    assert appdata.local_dir() == isolated_home / "Local" / "V-COMMS"
    for fn, name in ((appdata.profile_path, "tactical_voice_profile.json"), (appdata.presets_path, "custom_presets.json"),
                     (appdata.hotkeys_path, "hotkeys.json"), (appdata.settings_path, "settings.ini")):
        assert fn() == appdata.roaming_dir() / name
    assert appdata.documents_dir().is_dir()


def _legacy_library(path: Path, names, drive=0.0):
    entries = [{"name": n, "params": {"pitch_semitones": i, "drive_db": drive}} for i, n in enumerate(names)]
    path.write_text(json.dumps({"format": pr.LIBRARY_FORMAT, "version": 1, "presets": entries}))


@pytest.mark.parametrize("seed", SEEDS)
def test_migration_from_v102(isolated_home, seed):
    rnd = random.Random(seed)
    home = isolated_home / "home"
    downloads, desktop = home / "Downloads", home / "Desktop"
    downloads.mkdir()
    desktop.mkdir()
    old_profile = json.dumps(asdict_config(block_size=128))
    (downloads / appdata.PROFILE_FILENAME).write_text(old_profile)
    newer = json.dumps(asdict_config(block_size=512))
    (desktop / appdata.PROFILE_FILENAME).write_text(newer)
    os.utime(downloads / appdata.PROFILE_FILENAME, (time.time() - 100, time.time() - 100))
    names_a = [f"A{i}" for i in range(rnd.randint(0, 5))] + ["Shared"]
    names_b = [f"B{i}" for i in range(rnd.randint(0, 5))] + ["Shared"]
    _legacy_library(downloads / appdata.PRESETS_FILENAME, names_a, drive=1.0)
    _legacy_library(desktop / appdata.PRESETS_FILENAME, names_b, drive=2.0)  # "Shared" differs between the two
    _legacy_library(home / "Documents" / appdata.PRESETS_FILENAME, names_a, drive=1.0) if (home / "Documents").is_dir() else None
    report = appdata.migrate_legacy()
    assert report["profile"].endswith(str(Path("Desktop") / appdata.PROFILE_FILENAME))  # the newest one
    assert appdata.profile_path().read_text() == newer
    lib = pr.PresetLibrary(appdata.presets_path())
    lib.load()
    assert set(lib.names()) >= set(names_a) | set(names_b) - {"Shared"}
    assert "Shared (2)" in lib.names()  # same name, different settings: kept as a copy
    assert (downloads / appdata.PRESETS_FILENAME).exists()  # the old files are never deleted
    assert appdata.migrate_legacy() == {"skipped": True}
    assert "skipped" not in appdata.migrate_legacy(force=True)


def asdict_config(**kw):
    from dataclasses import asdict

    return asdict(eng.AppConfig(**kw))


def test_migration_survives_a_damaged_library(isolated_home):
    downloads = isolated_home / "home" / "Downloads"
    downloads.mkdir()
    (downloads / appdata.PRESETS_FILENAME).write_text("{ damaged")
    appdata.presets_path().write_text("{ also damaged")
    report = appdata.migrate_legacy()
    assert any(e.startswith("presets:") for e in report["errors"])
    assert appdata.presets_path().read_text() == "{ also damaged"  # never overwritten


def test_old_hotkeys_file_moves(isolated_home):
    appdata.local_dir().joinpath(appdata.HOTKEYS_FILENAME).write_text('{"hotkeys": {}}')
    report = appdata.migrate_legacy()
    assert report["hotkeys"] and appdata.hotkeys_path().exists()


# ------------------------------------------------------------------------------------------------ diagnostics & log
@pytest.mark.parametrize("seed", SEEDS)
def test_scrub_hides_the_user_name(monkeypatch, seed):
    rnd = random.Random(seed)
    name = "".join(rnd.choice(string.ascii_letters) for _ in range(rnd.randint(2, 12)))
    monkeypatch.setenv("USERNAME", name)
    text = f"C:\\Users\\{name}\\AppData\\x and /home/{name}/y and '{name}' alone and C:\\Users\\{name}"
    out = diagnostics.scrub(text)
    assert f"\\Users\\{name}\\" not in out and f"/{name}/" not in out
    assert out.count("%USERNAME%") >= 3


def test_diagnostics_text(isolated_home):
    applog.setup_logging()
    applog.get_logger().info("hello from the test")
    for h in applog.get_logger().handlers:
        h.flush()
    snapshot = {"input": "Mic [WASAPI]", "output": "CABLE Input [WASAPI]", "monitor": "", "sample_rate": 48000,
                "block_size": 256, "preset": "Deep Shadow (Omen)", "flags": {"bypass_effects": True}, "lite_mode": False,
                "stability_buffer": True, "engine": {"blocks": 10, "running": True, "mean_dsp_ms": 0.1,
                                                     "worst_dsp_ms": 0.4, "xruns": 2},
                "ai_model": {"loaded": True, "kind": "RVC", "errors": 3, "last_error": "RuntimeError: boom"},
                "qt_versions": "Qt x"}
    text = diagnostics.build_diagnostics(snapshot)
    assert "Block size: 256 samples (5.3 ms)" in text and "Bypass: on" in text and "dropouts: 2" in text
    assert "AI model: loaded (RVC)  |  3 conversion errors, last: RuntimeError: boom" in text
    assert "hello from the test" in text
    assert "not started" in diagnostics.build_diagnostics({})


def test_logging_and_excepthook(isolated_home):
    logger = applog.setup_logging()
    assert applog.setup_logging() is logger  # idempotent: still exactly one file handler of ours
    assert sum(1 for h in logger.handlers if getattr(h, "_vcomms", False)) == 1
    seen = []
    old_hook, old_thread_hook = sys.excepthook, threading.excepthook
    try:
        applog.install_excepthooks(on_error=seen.append)
        stderr = io.StringIO()
        sys.__stderr__, real = stderr, sys.__stderr__
        try:
            try:
                raise ValueError("kaboom")
            except ValueError:
                sys.excepthook(*sys.exc_info())
            t = threading.Thread(target=lambda: 1 / 0)
            t.start()
            t.join()
        finally:
            sys.__stderr__ = real
    finally:
        sys.excepthook, threading.excepthook = old_hook, old_thread_hook
    assert seen == ["ValueError: kaboom", "ZeroDivisionError: division by zero"]
    for h in logger.handlers:
        h.flush()
    tail = "\n".join(applog.tail(200))
    assert "kaboom" in tail and "ZeroDivisionError" in tail
    assert applog.log_path().is_relative_to(isolated_home)


# ------------------------------------------------------------------------------------------------ winintegration
@pytest.mark.skipif(sys.platform == "win32", reason="on Windows these write the real Run key and open Settings")
def test_windows_helpers_are_safe_elsewhere():
    assert wi.claim_instance() is True and wi.instance_running() is False
    assert wi.autostart_enabled() is False and wi.set_autostart(True) is False
    assert wi.cable_status() == {"input": "unknown", "output": "unknown"} and wi.mic_privacy() == "unknown"
    assert wi.open_uri("ms-settings:sound") is False
    cmd = wi.launch_command()
    assert cmd.endswith(wi.BACKGROUND_ARG) and "vcomms_main.py" in cmd
    assert wi.server_name().startswith("V-COMMS-")
    wi.allow_other_to_take_focus()


# ------------------------------------------------------------------------------------------------ downloaders
class _Handler(http.server.BaseHTTPRequestHandler):
    routes = {}

    def do_GET(self):  # noqa: N802
        route = self.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        body, declared = route
        self.send_response(200)
        self.send_header("Content-Length", str(declared if declared is not None else len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@contextmanager
def file_server(routes):
    _Handler.routes = routes
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("seed", SEEDS)
def test_cli_model_downloader(tmp_path, seed, monkeypatch):
    import download_rvc_models as dl

    rnd = random.Random(seed)
    good = os.urandom(rnd.randint(1000, 300000))
    digest = hashlib.sha256(good).hexdigest()
    routes = {"/good": (good, None), "/bad": (os.urandom(len(good)), None), "/short": (good[: len(good) // 2], len(good))}
    with file_server(routes) as base, redirect_stdout(io.StringIO()):
        dest = tmp_path / "model.onnx"
        assert dl.download_file([f"{base}/missing", f"{base}/bad", f"{base}/short", f"{base}/good"], dest, digest)
        assert dest.read_bytes() == good and not dest.with_suffix(".tmp").exists()
        dest.unlink()
        assert not dl.download_file([f"{base}/short"], dest, None)  # cut-off download: refused even without a hash
        assert not dl.download_file([f"{base}/bad"], dest, digest)
        assert not dest.exists() and not dest.with_suffix(".tmp").exists()
        big = tmp_path / "big.onnx"  # an existing file over 10 MB is only kept when its checksum matches
        big.write_bytes(b"\0" * 10_000_001)
        assert dl.download_file([f"{base}/good"], big, digest) and big.read_bytes() == good
    assert dl.SHA256["vec-768-layer-12.onnx"] == "b3886e7dff1495cda514f94f4680a7b1261e05d6929f5c764cdb17934b413c2a"


@pytest.mark.parametrize("seed", SEEDS)
def test_hubert_downloader(isolated_home, qapp, monkeypatch, seed):
    import hubert_downloader as hd

    rnd = random.Random(seed)
    good = os.urandom(rnd.randint(2000, 200000))
    monkeypatch.setattr(hd, "HUBERT_SHA256", hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(hd, "MIN_MODEL_BYTES", 1000)
    routes = {"/good": (good, None), "/bad": (good[::-1], None), "/tiny": (b"<html>404</html>", None)}
    with file_server(routes) as base:
        monkeypatch.setattr(hd, "HUBERT_MIRRORS", [f"{base}/missing", f"{base}/bad", f"{base}/good"])
        worker = hd.HuBERTDownloadWorker()
        results, progress = [], []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))
        worker.progress.connect(lambda *a: progress.append(a))
        worker.run()  # synchronously: the signals are delivered directly
        assert results and results[-1][0] is True, results
        assert worker.dest_path.read_bytes() == good and worker.dest_path.parent == appdata.roaming_dir() / "models"
        assert progress and progress[-1][0] == 100
        monkeypatch.setattr(hd, "HUBERT_MIRRORS", [f"{base}/tiny", f"{base}/missing"])
        results.clear()
        worker.run()
        assert results == [(False, "Could not download the base model from any mirror. Check your internet connection.")]
        assert not worker.dest_path.with_suffix(".tmp").exists()
        worker.cancel()
        results.clear()
        worker.run()
        assert results == []  # cancelled: no message at all
