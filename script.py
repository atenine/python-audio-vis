"""main script"""

import tkinter as tk
from threading import Thread
from tkinter import ttk
from os import listdir
import librosa
import pandas as pd
import numpy as np
import simpleaudio as sa
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
              text="Play Clip",
              command=lambda: on_play_button_click("sentences/clips/" + clips[cmb.current()],
                                                   items[cmb.current()])
              ).pack(pady=10)

    root.mainloop()

def on_play_button_click(path, name):
    """runs when the play button is clicked. plays audio and draws spectrogram simultaneously"""
    # kinda a gross solution but I want the audio playback to happen while the plot is visible
    Thread(target=play_audio, args=(path,)).start()
    Thread(target=generate_spectrogram, args=(path, name)).start()


def generate_spectrogram(path, name):
    """ uses librosa to draw a spectrogram"""
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    plt.tight_layout()
    plt.show()


def play_audio(path):
    """ plays the passed audio file"""
    # TODO: for now has some noise in it. figure out why
    y, sr = librosa.load(path, sr=None)
    audio = (y * 32767).astype(np.int16)  # Convert to 16-bit PCM
    play_obj = sa.play_buffer(audio, 1, 2, sr)
    play_obj.wait_done()

if __name__ == "__main__":
    __main__()
