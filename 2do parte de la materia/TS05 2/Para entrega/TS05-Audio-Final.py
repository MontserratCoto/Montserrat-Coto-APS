# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:58:15 2025

@author: monse
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

from scipy import signal
from scipy.fft import fft, fftshift

def vertical_flaten(a): #Llevan vectores planos  a columnas

    return a.reshape(a.shape[0],1)

from scipy.io import wavfile


# cierro ventanas anteriores
plt.close('all')

####################
# Lectura de audio #
####################

fs_audio, wav_data = wavfile.read('la cucaracha.wav')
#fs_audio, wav_data = wavfile.read('prueba psd.wav')
#fs_audio, wav_data = wavfile.read('silbido.wav')

# Tiempo para el eje x en segundos
N_audio = len(wav_data)
nperseg_audio = int(N_audio / 6)
t_audio = np.arange(N_audio) / fs_audio

plt.figure(figsize=(12, 4))
plt.plot(t_audio, wav_data, color='C0')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal de audio')
plt.grid()
plt.tight_layout()

# -----------------------
# 1.Estimación de la densidad espectral de potencia (PSD)
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


# ---------------------------------
# Cálculo del ancho de banda usando FFT (energía acumulada)
# ---------------------------------

# Normalización a potencia 1 W (quitando escala de amplitud)
wav_norm = wav_data / np.std(wav_data)

# Detrending (opcional si hay DC)
wav_detrended = wav_norm - np.mean(wav_norm)

# FFT y potencia
N = len(wav_detrended)
fft_wav = np.fft.rfft(wav_detrended)
Pxx_fft = np.abs(fft_wav) ** 2

# Frecuencias correspondientes
ff = np.fft.rfftfreq(N, 1/fs_audio)

# Normalización: potencia total unitaria
Pxx_norm = Pxx_fft / np.sum(Pxx_fft)

# Energía acumulada
Pxx_acumulada = np.cumsum(Pxx_norm)

# Índices donde se alcanza el 95% y 98% de la energía
i_95 = np.where(Pxx_acumulada >= 0.95)[0][0]
i_98 = np.where(Pxx_acumulada >= 0.98)[0][0]

# Frecuencias correspondientes
BW_95 = ff[i_95]
BW_98 = ff[i_98]

print(f"Ancho de banda 95% (FFT): {BW_95:.2f} Hz")
print(f"Ancho de banda 98% (FFT): {BW_98:.2f} Hz")

# ---------------------------------
# Gráfico 1: Potencia espectral (FFT) con líneas de BW
# ---------------------------------
plt.figure(figsize=(12, 5))
plt.plot(ff, 10 * np.log10(Pxx_norm), lw=1.5, label='PSD (FFT, normalizada a potencia total)')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Análisis de Ancho de Banda por Potencia Acumulada (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_audio / 2)
plt.tight_layout()


# ---------------------------------
# Gráfico 2: Energía acumulada (FFT)
# ---------------------------------
plt.figure(figsize=(12, 4))
plt.plot(ff, Pxx_acumulada, lw=2, label='Energía acumulada (normalizada)')
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - Audio (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_audio / 2)
plt.tight_layout()