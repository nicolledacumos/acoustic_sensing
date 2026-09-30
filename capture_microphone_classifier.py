import os
import queue
import time
from collections import deque

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import get_window

SAMPLE_RATE = 96_000
BLOCK_SIZE = 1024
CHANNELS = 1

FFT_SIZE = 4096

DISPLAY_MIN_FREQUENCY = 0
DISPLAY_MAX_FREQUENCY = 22050

# CHANGE THIS TO MATCH YOUR SYSTEM
INPUT_DEVICE = 2



ENERGY_THRESHOLD = 0.02        # RMS above this = "something happened"
SILENCE_RATIO_END = 0.4        # fraction of ENERGY_THRESHOLD that counts as "quiet again"

HISTORY_LEN = 25               
VOICE_MIN_ACTIVE_FRAMES = 6    

CLAP_FLATNESS_MIN = 0.30       
CLAP_HIGH_BAND_MIN = 0.15      
KNOCK_HIGH_BAND_MAX = 0.12     

# Set True to print feature values live for threshold calibration
DEBUG_PRINT_FEATURES = False

LABEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "current_label.txt")


audio_queue = queue.Queue(maxsize=50)


def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)

    try:
        audio_queue.put_nowait(indata[:, 0].copy())
    except queue.Full:
        try:
            audio_queue.get_nowait()
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Empty:
            pass


frequency_bins = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)

min_bin = np.searchsorted(frequency_bins, DISPLAY_MIN_FREQUENCY)
max_bin = np.searchsorted(frequency_bins, DISPLAY_MAX_FREQUENCY)

frequency_bins_display = frequency_bins[min_bin:max_bin]

window = get_window("hann", FFT_SIZE)

sample_buffer = np.zeros(FFT_SIZE, dtype=np.float32)


def compute_fft_and_features(block):
    global sample_buffer

    sample_buffer = np.roll(sample_buffer, -len(block))
    sample_buffer[-len(block):] = block

    centered = sample_buffer - np.mean(sample_buffer)
    windowed = centered * window
    spectrum = np.fft.rfft(windowed)

    magnitude = np.abs(spectrum) / np.sum(window)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

    eps = 1e-12
    power = magnitude + eps
    geo_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = float(geo_mean / arith_mean)
    centroid = float(np.sum(frequency_bins * power) / np.sum(power))

    total_energy = np.sum(power ** 2)

    def band_ratio(lo, hi):
        mask = (frequency_bins >= lo) & (frequency_bins < hi)
        return float(np.sum(power[mask] ** 2) / total_energy) if total_energy > 0 else 0.0

    feat = {
        "rms": float(np.sqrt(np.mean(block ** 2))),
        "flatness": flatness,
        "centroid": centroid,
        "low": band_ratio(0, 500),
        "mid": band_ratio(500, 2000),
        "high": band_ratio(2000, 8000),
        "very_high": band_ratio(8000, SAMPLE_RATE / 2),
    }

    return magnitude_db[min_bin:max_bin], feat


feature_history = deque(maxlen=HISTORY_LEN)
event_open = False


def classify_event(features_in_event):
    n_frames = len(features_in_event)
    avg = {
        k: np.mean([f[k] for f in features_in_event])
        for k in ("flatness", "centroid", "high", "very_high", "low")
    }
    high_band = avg["high"] + avg["very_high"]

    if high_band >= CLAP_HIGH_BAND_MIN and avg["flatness"] >= CLAP_FLATNESS_MIN:
        return "CLAP"
    if high_band <= KNOCK_HIGH_BAND_MAX:
        return "KNOCK"
    return "CLAP" if high_band > KNOCK_HIGH_BAND_MAX else "KNOCK"


def write_label(label):
    try:
        with open(LABEL_FILE, "w") as f:
            f.write(f"{label},{time.time()}")
    except OSError as e:
        print("Could not write label file:", e)


def process_classification(feat):
    global event_open

    if DEBUG_PRINT_FEATURES:
        print(f"rms={feat['rms']:.3f} flat={feat['flatness']:.2f} "
              f"high={feat['high'] + feat['very_high']:.2f}")

    is_loud = feat["rms"] > ENERGY_THRESHOLD
    is_quiet_again = feat["rms"] < ENERGY_THRESHOLD * SILENCE_RATIO_END

    if is_loud:
        feature_history.append(feat)
        event_open = True
    elif event_open and is_quiet_again:
        if feature_history:
            label = classify_event(list(feature_history))
            print(f"Detected: {label}")
            write_label(label)
        feature_history.clear()
        event_open = False


def update_plot(_frame):
    latest_fft = None

    while True:
        try:
            block = audio_queue.get_nowait()
        except queue.Empty:
            break
        latest_fft, feat = compute_fft_and_features(block)
        process_classification(feat)

    if latest_fft is not None:
        line.set_ydata(latest_fft)

        peak_index = np.argmax(latest_fft)
        peak_frequency = frequency_bins_display[peak_index]
        peak_amplitude = latest_fft[peak_index]

        peak_text.set_text(
            f"Peak Frequency: {peak_frequency:7.1f} Hz\n"
            f"Amplitude:      {peak_amplitude:7.1f} dB"
        )

        peak_marker.set_data([peak_frequency], [peak_amplitude])

    return line, peak_text, peak_marker


# -----------------------------
# Start audio stream and plot
# -----------------------------
print("Available audio devices:")
print(sd.query_devices())

print("\nStarting microphone FFT + classifier.")
print("Close the plot window or press Ctrl+C to stop.")
print(f"Detections will be written to: {LABEL_FILE}")

fig, ax = plt.subplots(figsize=(11, 6))

initial_fft = np.full(len(frequency_bins_display), -100.0)

line, = ax.plot(
    frequency_bins_display,
    initial_fft,
    color="blue",
    linewidth=1.5,
)

peak_marker, = ax.plot(
    [0], [-100],
    marker="o",
    markersize=8,
    color="purple",
    linestyle="None",
)

peak_text = ax.text(
    0.02, 0.95,
    "Peak Frequency: ----- Hz\nAmplitude:      ---.-- dB",
    transform=ax.transAxes,
    horizontalalignment="left",
    verticalalignment="top",
    fontsize=13,
    color="purple",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax.set_title("Live Microphone FFT + Sound Classifier")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (dB)")
ax.set_xlim(DISPLAY_MIN_FREQUENCY, DISPLAY_MAX_FREQUENCY)
ax.set_ylim(-100, 0)
ax.grid(True, alpha=0.3)

stream = sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype="float32",
    channels=(CHANNELS, CHANNELS),
    device=(INPUT_DEVICE, None),
    callback=audio_callback,
    latency="low",
)

try:
    with stream:
        animation = FuncAnimation(
            fig,
            update_plot,
            interval=50,
            blit=True,
            cache_frame_data=False,
        )
        plt.show()

except KeyboardInterrupt:
    pass

finally:
    plt.close(fig)
    print("Stopped.")
