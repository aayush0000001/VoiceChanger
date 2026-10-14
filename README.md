# V-COMMS // Real-Time Low-Latency Tactical Voice Engine (Windows 11)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Platform Windows 11](https://img.shields.io/badge/Platform-Windows%2011%20(WASAPI)-0078D6?logo=windows)
![Latency ~19ms](https://img.shields.io/badge/Est.%20Roundtrip%20Latency-~19ms-00F5D4)
![Vanguard Safe](https://img.shields.io/badge/Riot%20Vanguard-100%25%20Compliant-FF4655)

**V-COMMS** is a lightweight, wait-free real-time DSP and ONNX voice conversion desktop application engineered for tactical gaming voice chat (**VALORANT**, **Discord**, **OBS Studio**) on Windows 11.

Built to operate strictly at the OS audio-routing layer via **Windows WASAPI**, it targets **~19 ms estimated roundtrip latency** at 256 samples / 48 kHz (two buffers + DSP time + ~6 ms driver allowance, as shown in the app's status bar) and pins itself to the last two logical cores, which are the Efficiency Cores (E-Cores) on Intel hybrid CPUs.

---

## Key Engineering Highlights

* **100% Riot Vanguard Compliant:** Never reads/writes game memory, never enumerates processes, and never injects `WH_KEYBOARD_LL` global input hooks. Global hotkeys use the native Win32 `RegisterHotKey` OS message pump (`0.00%` idle CPU).
* **Wait-Free Real-Time Audio Callback (`O(1)` Auxiliary Memory):**
  * **Read-Copy-Update (RCU) Parameter Swapping:** UI slider adjustments compile `float32` second-order section (`sos`) filter matrices off-thread and swap an immutable snapshot pointer atomically—the audio thread **never** acquires a mutex lock.
  * **Power-of-Two Bitwise Pitch Shifter:** Dual-head rotating delay line ($2^{12} = 4096$ samples) using bitwise masking (`& 4095`), single-multiply fused linear interpolation, and the complementary Hann identity $y = v_B + g_A(v_A - v_B)$ with zero heap allocations per block.
  * **Wait-Free SPSC Monitor Ring Buffer:** Lock-free monotonic ring buffer ($2^{14} = 16384$ samples) with automatic clock-drift clamping for zero-lag *"Hear Myself"* side-tone monitoring.
  * **Zero-Roll 2x Mirrored ONNX Buffer:** Eliminates $\mathcal{O}(W)$ `np.roll` array copies during streaming AI inference by maintaining a double-length ($2W$) mirrored ring buffer that exposes $\mathcal{O}(1)$ C-contiguous views.
* **Hybrid CPU Affinity Isolation:** Raises process priority to Above Normal and, on any PC with 8+ logical cores, pins the whole process to the final 2 logical cores. On Intel 12th–14th Gen hybrid CPUs these are E-Cores, leaving VALORANT the P-Cores; on AMD or non-hybrid CPUs they are ordinary cores. The 60Hz GUI telemetry timer also pauses when the window is closed to the system tray.

---

## Prerequisites

1. **Operating System:** Windows 11 (64-bit)
2. **Python:** Python 3.10 or newer (tested on 3.13)
3. **Virtual Audio Device:** [VB-CABLE Virtual Audio Device](https://vb-audio.com/Cable/) (Free)

---

## Quick Start Installation

### 1. Clone the Repository & Create a Virtual Environment
```powershell
git clone https://github.com/<your-username>/v-comms-voice-engine.git
cd v-comms-voice-engine

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Automated Self-Test & Latency Benchmark
Verify that all DSP stages, lock-free buffers, and real-time deadlines pass on your hardware before launching the GUI:
```powershell
python val_voice_changer.py --self-test
```

### 3. Launch the Application
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
1. **Physical Mic Input:** Select your hardware microphone (`[Windows WASAPI]`).
2. **Virtual Cable Output:** Select `CABLE Input (VB-Audio Virtual Cable) [Windows WASAPI]`.
3. **Headphone Monitor:** Select your gaming headphones (`[Windows WASAPI]`). It must be a different device from the Virtual Cable Output, otherwise no monitor stream is opened. Tick **Hear Myself** to enable it.
4. **Buffer Size (@48kHz):** Select **256 samples (5.3 ms)**.
5. Click **START AUDIO ENGINE**.
6. *(Optional)* Close the window (`X`) to minimize V-COMMS to the Windows System Tray—this pauses the 60 FPS UI meters for zero GPU/DWM overhead while keeping hotkeys active.

### Inside VALORANT (and Discord / OBS):
1. Open **VALORANT** -> **Settings** -> **Audio** -> **Voice Chat**.
2. Set **Input Device** to **CABLE Output (VB-Audio Virtual Cable)**.

---

> **Note on the ONNX AI engine (experimental):** `onnxruntime-directml` is optional; the app runs without it. The model is fed the last 4096 samples as a single `1 x 4096` float32 input and runs inside the audio callback, so most real voice-conversion models (e.g. RVC) will not work as-is.

> **Profiles:** On startup the app auto-loads `tactical_voice_profile.json` from the folder it is launched from, if present. Use **Save JSON Profile** with that name to make it your default.

---

## Global In-Game Hotkeys

All hotkeys work in fullscreen VALORANT with zero hook injection:

| Global Hotkey | Action / Preset |
| :--- | :--- |
| `Ctrl + Alt + F1` | **Deep Shadow (Omen)** — Sub-harmonic pitch/formant drop + dark spatializer |
| `Ctrl + Alt + F2` | **Tactical Comms (Radio)** — Tight band-pass (420Hz–2.9kHz) + Pade overdrive crunch |
| `Ctrl + Alt + F3` | **Cybernetic Agent (KAY/O)** — Phase-continuous 58Hz ring modulator + vocal tract shift |
| `Ctrl + Alt + F4` | **Pitch Clean (Up / Helium-Lite)** — Clean upward pitch & formant shift |
| `Ctrl + Alt + F9` | **Bypass Toggle** — Instant passthrough to your raw natural voice |
| `Ctrl + Alt + F10` | **Hardware Mute Toggle** — Zero-fill output buffer instantly |

---

## Compiling a Standalone `.exe` (Optional Deployment)

To package V-COMMS into a standalone Windows executable using **PyInstaller**:

```powershell
pip install pyinstaller
pyinstaller --noconsole --onefile --name "V-COMMS" val_voice_changer.py
```
The compiled executable will be generated in `dist\V-COMMS.exe`.

---

## License
MIT License