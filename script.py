"""main script"""

import tkinter as tk
from tkinter import ttk
from os import listdir
import os
from pathlib import Path
import subprocess
import glob
import librosa
import pandas as pd
import numpy as np
import simpleaudio as sa
from PIL import Image
import cv2

import matplotlib.pyplot as plt


def __main__():
    # for sentences
    # base_path = "sentences/"
    # clips = [f for f in listdir(base_path + "clips/")]
    # df = pd.read_csv(base_path + "transc.csv")
    # items = [': '.join(row[0:2]) for row in df.values]

    # for words
    base_path = "words/"
    clips = [f for f in listdir(base_path + "clips/")]
    df = pd.read_csv(base_path + "transc.csv")
    items = [': '.join(row[0:2]) for row in df.values]

    root = tk.Tk()
    root.title("Audio Clip Player")
    root.geometry("500x150")

    cmb = ttk.Combobox(root, textvariable=tk.StringVar(value="Select Clip"), values=items, width=75)
    cmb.pack(pady=15)
    cmb.current(0)

    tk.Button(root,
              text="Render",
              command=lambda: on_play_button_click(base_path + "clips/" + clips[cmb.current()],
                                                   items[cmb.current()])
              ).pack(pady=10)
    tk.Button(root, text="Done", command=root.destroy).pack(pady=5)

    root.mainloop()

def on_play_button_click(path, name):
    """generates spectrogram and video with spectral visualizer"""
    # play_audio(path)
    generate_spectrogram(path, name)
    generate_frames(path)
    print("Frames generated.")
    generate_video(path)
    print("Video generated.")
    clear_output()
    print("cleaned up output.")


def generate_spectrogram(path, name):
    """ draws and saves spectrogram"""
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=10000)

    plt.ioff()

    plt.figure(figsize=(10, 5))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=10000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    fig = plt.gcf()
    fig.patch.set_facecolor('black')
    ax = plt.gca()
    ax.set_facecolor('black')
    ax.title.set_color('white')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    # style any colorbar axes (usually the remaining axes)
    for cax in fig.axes[1:]:
        cax.set_facecolor('black')
        cax.tick_params(colors='white')
        for tl in cax.get_yticklabels():
            tl.set_color('white')
    plt.tight_layout()

    filename = Path("output/spectrograms") / f"{path.split('/')[-1]}.png"

    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved to: {filename}")

def generate_frames(path):
    """ draws and saves spectral visualization frames"""
    y, sr = librosa.load(path, sr=None)

    plt.ioff()

    for i in range(0, len(y), int(sr/60)):
        # 60 frames per second
        frame = y[i:i + int(sr/60)]

        # get fft (not really sure what these do individually)
        window = np.hanning(len(frame))
        spectrum = np.fft.rfft(frame * window)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)

        plt.clf()
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.set_box_aspect(1)
        ax.title.set_color('white')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        plt.gcf().patch.set_facecolor('black')
        plt.plot(freqs, mags, color='white')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.ylim(0, 160)
        plt.xlim(0, 10000)
        plt.grid(True, color='gray', alpha=0.3)
        plt.savefig(f"output/frames/spectrogram_frame_{i//int(sr/60):04d}.png", dpi=150, facecolor=plt.gcf().get_facecolor(), bbox_inches='tight')
        print("Saved frame " + str(i//int(sr/60)) + " of " + str(len(y)//int(sr/60)))
    
    plt.close()
    return

def generate_video(path):
    """ generates a video from the saved .png frames"""
    print("Generating video...")
    frames = sorted(glob.glob("output/frames/spectrogram_frame_*.png"))

    # Read the first frame to get dimensions
    first_frame = Image.open(frames[0])
    frame_width, frame_height = first_frame.size

    out_path = "output/spectrogram.avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 60, (frame_width, frame_height))

    for frame_path in frames:
        # Convert PIL Image to OpenCV format
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

    # Merge video with audio
    audio_path = path
    video_path = "output/spectrogram.avi"
    output_path = "output/visualizers/" + Path(path).name + ".mp4"

    # Use ffmpeg to combine video and audio
    cmd = f'ffmpeg -i {video_path} -i {audio_path} -c:v copy -c:a aac {output_path} -y'
    return subprocess.run(cmd, shell=True, check=True)

def play_audio(path):
    """ plays the passed audio file"""
    # TODO: for now has some noise in it. figure out why
    y, sr = librosa.load(path, sr=None)
    audio = (y * 32767).astype(np.int16)  # Convert to 16-bit PCM
    sa.play_buffer(audio, 1, 2, sr)

def clear_output():
    """ clears the output frames and temporary video files"""
    files = glob.glob("output/frames/spectrogram_frame_*.png")
    for f in files:
        os.remove(f)
    temp_video = "output/spectrogram.avi"
    if os.path.exists(temp_video):
        os.remove(temp_video)

if __name__ == "__main__":
    __main__()
