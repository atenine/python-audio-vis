"""main script"""

import tkinter as tk
from tkinter import ttk
from os import listdir
import librosa
import pandas as pd
import numpy as np
import simpleaudio as sa
from pathlib import Path
import re
import matplotlib.pyplot as plt


def __main__():
    clips = [f for f in listdir("sentences/clips/")]
    df = pd.read_csv("sentences/transc.csv")
    items = [': '.join(row[0:2]) for row in df.values]

    root = tk.Tk()
    root.title("Audio Clip Player")
    root.geometry("500x150")

    cmb = ttk.Combobox(root, textvariable=tk.StringVar(value="Select Clip"), values=items, width=75)
    cmb.pack(pady=20)
    cmb.current(0)

    tk.Button(root,
              text="Play Clip and Show Spectrogram",
              command=lambda: on_play_button_click("sentences/clips/" + clips[cmb.current()],
                                                   items[cmb.current()])
              ).pack(pady=10)

    root.mainloop()

def on_play_button_click(path, name):
    """runs when the play button is clicked. plays audio and draws spectrogram simultaneously"""
    play_audio(path)
    generate_spectrogram(path, name)


def generate_spectrogram(path, name):
    """ uses librosa to draw a spectrogram"""
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, x_axis='time', y_axis='mel')
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
    out_dir = Path("spectrograms")
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"{path.split('/')[-1]}.png"

    plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"Spectrogram saved to: {filename}")
    plt.show()



def play_audio(path):
    """ plays the passed audio file"""
    # TODO: for now has some noise in it. figure out why
    y, sr = librosa.load(path, sr=None)
    audio = (y * 32767).astype(np.int16)  # Convert to 16-bit PCM
    sa.play_buffer(audio, 1, 2, sr)

if __name__ == "__main__":
    __main__()
