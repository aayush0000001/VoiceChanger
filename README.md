# V-COMMS // Real-Time Low-Latency Tactical Voice Engine (Windows 11)

[![Download Executable](https://img.shields.io/badge/Download-Standalone%20EXE-2ea44f?style=for-the-badge&logo=windows)](https://github.com/aayush0000001/VoiceChanger/releases/latest)
![Platform Windows 11](https://img.shields.io/badge/Platform-Windows%2011%20(WASAPI)-0078D6?logo=windows)
![Latency ~19ms](https://img.shields.io/badge/Est.%20Roundtrip%20Latency-~19ms-00F5D4)
![Vanguard Safe](https://img.shields.io/badge/Riot%20Vanguard-100%25%20Compliant-FF4655)

**V-COMMS** is a lightweight, wait-free real-time DSP voice conversion desktop application engineered for tactical gaming voice chat (**VALORANT**, **Discord**, **OBS Studio**) on Windows 11.

---

## ⚡ Quick Start for Users (No Python Needed!)

If you just want to use the voice changer without installing Python or programming dependencies:

1. **Download the Executable:**
   👉 Go to **[Latest Releases](https://github.com/aayush0000001/VoiceChanger/releases/latest)** and download **`V-COMMS.exe`**.
2. **Install VB-CABLE:**
   Download and install the free [VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/) (reboot if prompted).
3. **Run the App:**
   Double-click **`V-COMMS.exe`** to start.

---

## Key Engineering Highlights

* **100% Riot Vanguard Compliant:** Never reads/writes game memory, never enumerates processes, and never injects `WH_KEYBOARD_LL` global input hooks. Global hotkeys use the native Win32 `RegisterHotKey` OS message pump (`0.00%` idle CPU).
* **Wait-Free Real-Time Audio Callback (`O(1)` Auxiliary Memory):**
  * **Read-Copy-Update (RCU) Parameter Swapping:** UI slider adjustments compile `float32` second-order section (`sos`) filter matrices off-thread and swap an immutable snapshot pointer atomically—the audio thread **never** acquires a mutex lock.
  * **Power-of-Two Bitwise Pitch Shifter:** Dual-head rotating delay line ($2^{12} = 4096$ samples) using bitwise masking (`& 4095`), single-multiply fused linear interpolation, and the complementary Hann identity $y = v_B + g_A(v_A - v_B)$ with zero heap allocations per block.
  * **Wait-Free SPSC Monitor Ring Buffer:** Lock-free monotonic ring buffer ($2^{14} = 16384$ samples) with automatic clock-drift clamping for zero-lag *"Hear Myself"* side-tone monitoring.
  * **Zero-Roll 2x Mirrored ONNX Buffer:** Eliminates $\mathcal{O}(W)$ `np.roll` array copies during streaming AI inference by maintaining a double-length ($2W$) mirrored ring buffer that exposes $\mathcal{O}(1)$ C-contiguous views.
* **Hybrid CPU Affinity Isolation:** Raises process priority to Above Normal and, on any PC with 8+ logical cores, pins the whole process to the final 2 logical cores. On Intel 12th–14th Gen hybrid CPUs these are E-Cores, leaving VALORANT the P-Cores; on AMD or non-hybrid CPUs they are ordinary cores. The 30Hz GUI telemetry timer also pauses when the window is closed to the system tray.

---

## Developer / Source Installation

If you want to run or modify the code directly from source:

### 1. Prerequisites
- **Operating System:** Windows 11 (64-bit)
- **Python:** Python 3.10 or newer (tested on 3.11–3.13)
- **Virtual Audio Device:** [VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/)

### 2. Clone & Setup Virtual Environment
```powershell
git clone https://github.com/aayush0000001/VoiceChanger.git
cd VoiceChanger

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run Self-Test & Benchmark
```powershell
python val_voice_changer.py --self-test
```

### 4. Launch the GUI
```powershell
python val_voice_changer.py
```

---

## Mandatory Windows 11 + VB-Cable 48kHz Setup (Prevents Hidden OS Latency)

To prevent the Windows 11 Audio Graph from activating its internal sample-rate converter (which adds 15–25 ms of buffer lag), match your sample rates to **48000 Hz**:

1. Install **VB-CABLE** (run installer as Administrator and reboot).
2. Press `Win + R`, type `mmsys.cpl`, and press **Enter**.
3. Under the **Playback** tab:
   * Double-click **CABLE Input (VB-Audio Virtual Cable)** -> **Advanced** tab.
   * Set Default Format to **24-bit, 48000 Hz (Studio Quality)** -> Click **Apply**.
4. Under the **Recording** tab:
   * Double-click **CABLE Output (VB-Audio Virtual Cable)** -> **Advanced** tab -> Set to **48000 Hz**.
   * Double-click your **Physical Microphone** -> **Advanced** tab -> Set to **48000 Hz**.

---

## Configuring V-COMMS & VALORANT

### Inside V-COMMS:
On the **Devices** page:
1. **Physical Mic Input:** Select your hardware microphone (`[Windows WASAPI]`).
2. **Virtual Cable Output:** Select `CABLE Input (VB-Audio Virtual Cable) [Windows WASAPI]`.
3. **Headphone Monitor:** Select your gaming headphones (`[Windows WASAPI]`). It must be a different device from the Virtual Cable Output, otherwise no monitor stream is opened. Turn on **Hear Myself** to enable it.
4. **Buffer Size (@48kHz):** Select **256 samples (5.3 ms)**.

Then on the **Live** page:

5. Click **Start audio engine**.
6. *(Optional)* Close the window (`X`) to minimize V-COMMS to the Windows System Tray—this pauses the 30 FPS UI meters for zero GPU/DWM overhead while keeping hotkeys active.

### Inside VALORANT (and Discord / OBS):
1. Open **VALORANT** -> **Settings** -> **Audio** -> **Voice Chat**.
2. Set **Input Device** to **CABLE Output (VB-Audio Virtual Cable)**.

---

## Global In-Game Hotkeys

All hotkeys work in fullscreen VALORANT with zero hook injection:

| Default hotkey | Action / Preset |
| :--- | :--- |
| `Ctrl + Alt + F1` | **Deep Shadow (Omen)** — Sub-harmonic pitch/formant drop + dark spatializer |
| `Ctrl + Alt + F2` | **Tactical Comms (Radio)** — Tight band-pass (420Hz–2.9kHz) + Pade overdrive crunch |
| `Ctrl + Alt + F3` | **Cybernetic Agent (KAY/O)** — Phase-continuous 58Hz ring modulator + vocal tract shift |
| `Ctrl + Alt + F4` | **Pitch Clean (Up / Helium-Lite)** — Clean upward pitch & formant shift |
| `Ctrl + Alt + F9` | **Bypass Toggle** — Instant passthrough to your raw natural voice |
| `Ctrl + Alt + F10` | **Hardware Mute Toggle** — Zero-fill output buffer instantly |

These are the defaults. Change any of them in **Settings > Global hotkeys**: press *Change*, then the new keys (`Esc` cancels, `Backspace` or `Delete` clears the hotkey, *Restore defaults* brings them all back). The new combination works at once, with the window hidden, and is kept between runs in `%APPDATA%\V-COMMS\hotkeys.json`; *Save JSON Profile* stores the hotkeys in the profile too, and *Reset all settings* restores the defaults. Hotkeys stay keyboard-only `RegisterHotKey` registrations (no hooks, no mouse buttons). Letters, digits, `Space` and the arrow, `Insert`, `Delete`, `Home`, `End` and `Page` keys need `Ctrl`, `Alt` or `Shift`; `F1`-`F11`, `F13`-`F24` and the numpad can stand alone; `F12` is reserved by Windows. A registered combination is taken away from every program, the game included, so avoid a lone `Shift` or `Ctrl` with a letter or digit (VALORANT walks with `Shift` and crouches with `Ctrl`): V-COMMS warns you about that.

---

## Presets, Reset, Performance and Support

* **Custom presets:** **Voice > Presets > Save as preset** keeps your current sliders under your own name (up to 24). They sit next to the four built-in voices on the **Live** page, each with a `...` menu to rename, export or delete it; the four hotkeys (`Ctrl + Alt + F1`-`F4` by default) stay on the built-ins. Presets are shared as small `.json` files (*Import preset*): the file is read as plain data, limited to 64 KB, and every value is clamped to its slider's range. They are stored in `custom_presets.json` next to the profile file.
* **Reset all settings** (Settings > Reset) restores the first-launch effects, switches, devices, buffer size, hotkeys and theme and rewrites the auto-loaded profile. Custom presets are kept unless you tick *Also delete my custom presets*.
* **Lite mode** and **Stability buffer** (Settings > Performance) are for older or busy PCs: Lite mode uses slower meters, no fades or Mica, and hands memory back to Windows while the window is hidden; the Stability buffer asks the audio drivers for a bigger buffer (fewer dropouts, a little more delay; Settings shows how many ms on your devices).
* **Report a problem** (Settings > Support) sends a bug, glitch or idea. You see the exact text, diagnostics included, before anything leaves your PC. To receive reports, put your Formspree form address in `report_config.py`; until then the report is copied to the clipboard.

---

## Compiling Standalone `.exe` Locally

To package V-COMMS manually using **PyInstaller** with the custom application icon:

```powershell
pip install pyinstaller
pyinstaller --noconsole --onefile --name "V-COMMS" --icon app_icon.ico --collect-all qfluentwidgets val_voice_changer.py
```
The compiled executable will be generated in `dist\V-COMMS.exe`.

---

## License
MIT License
