# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:56:34 2025

@author: monse
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import wavfile
import sounddevice as sd

def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

# cierro ventanas anteriores
plt.close('all')

####################
# Lectura de audio #
####################

fs_audio, wav_data = wavfile.read('la cucaracha.wav')
#fs_audio, wav_data = wavfile.read('prueba psd.wav')
#fs_audio, wav_data = wavfile.read('silbido.wav')

# -----------------------
# 1.Estimación de la densidad espectral de potencia (PSD)
# PSD con Welch normalizado al pico
# -----------------------

# Cálculo de PSD con Welch
from scipy.signal import welch
N_audio = len(wav_data)
nperseg_audio = int(N_audio / 6)

# Welch (sin normalización de potencia)
f_welch, Pxx_welch = welch(
    wav_data,
    fs=fs_audio,
    window='hann',
    nperseg=nperseg_audio,
    noverlap=nperseg_audio // 2,
    detrend=False,
    scaling='density'
)

# Normalización al pico (dB)
Pxx_welch_db = 10 * np.log10(Pxx_welch)
Pxx_welch_db -= np.max(Pxx_welch_db)

# Gráfico
plt.figure()
plt.plot(f_welch, Pxx_welch_db, label='Audio (Welch, normalizado al pico)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('PSD - Audio (normalizada al pico)')
plt.grid()
plt.xlim(0, fs_audio / 2)
plt.legend()


# -----------------------
# 2. Cálculo del ancho de banda usando FFT (energía acumulada)
# -----------------------

# Normalización de la señal a potencia 1 W
wav_norm = wav_data / np.std(wav_data)

# FFT directa
N = len(wav_norm)
fft_wav = np.fft.fft(wav_norm)
Pxx_fft = np.abs(fft_wav) ** 2

# Frecuencias correspondientes con linspace y máscara
df = fs_audio / N
ff = np.linspace(0, (N - 1) * df, N)  # Grilla de frecuencia
bfrec = ff <= fs_audio / 2           # Solo la mitad positiva

ff = ff[bfrec]
Pxx_fft = Pxx_fft[bfrec]

# Normalización en potencia total
Pxx_norm = Pxx_fft / np.sum(Pxx_fft)

# Energía acumulada
Pxx_acumulada = np.cumsum(Pxx_norm)

# Cálculo de BW al 95% y 98%
i_95 = np.where(Pxx_acumulada >= 0.95)[0][0]
i_98 = np.where(Pxx_acumulada >= 0.98)[0][0]
BW_95 = ff[i_95]
BW_98 = ff[i_98]

# Gráfico de PSD por Welch con líneas de BW_95 y BW_98 calculadas con FFT
plt.figure(figsize=(12, 5))
plt.plot(f_welch, Pxx_welch_db, lw=1.5, label='PSD (Welch, normalizada al pico)')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Análisis de Ancho de Banda por Potencia Acumulada (en PSD Welch)')
plt.grid()
plt.legend()
plt.xlim(0, fs_audio / 2)
plt.tight_layout()
plt.show()


# Gráfico de la curva de energía acumulada
plt.figure()
plt.plot(ff, Pxx_acumulada, lw=2)
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - Audio (FFT)')
plt.grid()
plt.legend()
