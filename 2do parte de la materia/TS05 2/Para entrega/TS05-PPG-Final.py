# -*- coding: utf-8 -*-
"""
Created on Mon May 26 21:05:16 2025

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

plt.close('all')


####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

N_ppg = len(ppg)

t = np.arange(len(ppg)) / fs_ppg
plt.figure()
plt.plot(t, ppg)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (unidades arbitrarias)')
plt.title('Señal de Pletismografía (PPG)')

# ---------------------------------
# 1. PSD con Welch normalizado al pico
# ---------------------------------

f_welch, Pxx_welch = signal.welch(ppg, fs=fs_ppg, window='hann',
                                  nperseg=int(N_ppg/6), noverlap=int(N_ppg/12),
                                  detrend=False, scaling='density')

Pxx_welch_db = 10 * np.log10(Pxx_welch)
Pxx_welch_db -= np.max(Pxx_welch_db)  # Normalizo al pico para PSD

plt.figure()
plt.plot(f_welch, Pxx_welch_db, label='PPG original (Welch, normalizado al pico)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('PSD PPG (normalizada al pico)')
plt.grid()
plt.xlim(0, fs_ppg/2)
plt.legend()

# ---------------------------------------
# Filtrado pasa banda Butterworth 4to orden (0.3 a 10 Hz)
# ---------------------------------------
f_low = 0.3
f_high = 10
nyq = fs_ppg / 2

b, a = signal.butter(4, [f_low / nyq, f_high / nyq], btype='bandpass')

#filtrado sobre la señal original, no normalizada
ppg_filtrada = signal.filtfilt(b, a, ppg)

t_ppg = np.arange(N_ppg) / fs_ppg

plt.figure(figsize=(12, 6))
plt.plot(t_ppg, ppg, alpha=0.4, label='PPG original (sin filtrar)')
plt.plot(t_ppg, ppg_filtrada, label='PPG filtrada (0.3 - 10 Hz)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal PPG antes y después del filtrado')
plt.legend()
plt.grid()
plt.tight_layout()

# PSD para señal filtrada (normalización al pico)
f_welch_filt, Pxx_welch_filt = signal.welch(ppg_filtrada, fs=fs_ppg, window='hann',
                                            nperseg=int(N_ppg/6), noverlap=int(N_ppg/12),
                                            detrend=False, scaling='density')

Pxx_welch_filt_db = 10 * np.log10(Pxx_welch_filt)
Pxx_welch_filt_db -= np.max(Pxx_welch_filt_db)

plt.figure()
plt.plot(f_welch, Pxx_welch_db, label='PPG original (Welch)')
plt.plot(f_welch_filt, Pxx_welch_filt_db, label='PPG filtrada (Welch)', linewidth=2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Comparación PSD PPG antes y después del filtrado')
plt.grid()
plt.xlim(0, fs_ppg/2)
plt.legend()
plt.tight_layout()

plt.show()

# ---------------------------------
# 2. Cálculo del ancho de banda usando FFT con normalización a potencia 1
# ---------------------------------

ppg_norm = ppg / np.std(ppg)
ppg_detrended = ppg_norm - np.mean(ppg_norm)  # Quitar componente DC

N = len(ppg_detrended)
fft_ppg = np.fft.rfft(ppg_detrended)
Pxx_fft = np.abs(fft_ppg)**2
ff = np.fft.rfftfreq(N, 1/fs_ppg)

# Normalización de potencia total a 1
Pxx_norm = Pxx_fft / np.sum(Pxx_fft)
Pxx_acumulada = np.cumsum(Pxx_norm)

# BW por energía acumulada
i_95 = np.where(Pxx_acumulada >= 0.95)[0][0]
i_98 = np.where(Pxx_acumulada >= 0.98)[0][0]
BW_95 = ff[i_95]
BW_98 = ff[i_98]

print(f"Ancho de banda 95%: {BW_95:.2f} Hz")
print(f"Ancho de banda 98%: {BW_98:.2f} Hz")

# Espectro FFT en dB normalizado al pico
Pxx_fft_db = 10 * np.log10(Pxx_fft / np.max(Pxx_fft))

# Gráfico del espectro FFT con líneas de BW
plt.figure(figsize=(12, 5))
plt.plot(ff, Pxx_fft_db, lw=1.5, label='Espectro (FFT, normalizado al pico)')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Análisis de Ancho de Banda por Energía Acumulada (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ppg / 2)
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
plt.title('Curva de Energía Acumulada - PPG (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ppg / 2)
plt.tight_layout()

# -------------------------------
# 4. BW por energía acumulada - señal filtrada
# -------------------------------

# Normalizamos y quitamos DC
ppg_filt_norm = ppg_filtrada / np.std(ppg_filtrada)
ppg_filt_detrended = ppg_filt_norm - np.mean(ppg_filt_norm)

# FFT
N_filt = len(ppg_filt_detrended)
fft_ppg_filt = np.fft.rfft(ppg_filt_detrended)
Pxx_fft_filt = np.abs(fft_ppg_filt)**2
ff_filt = np.fft.rfftfreq(N_filt, 1/fs_ppg)

# Normalización de energía total a 1
Pxx_norm_filt = Pxx_fft_filt / np.sum(Pxx_fft_filt)
Pxx_acumulada_filt = np.cumsum(Pxx_norm_filt)

# Índices para BW 95% y 98%
i_95_filt = np.where(Pxx_acumulada_filt >= 0.95)[0][0]
i_98_filt = np.where(Pxx_acumulada_filt >= 0.98)[0][0]
BW_95_filt = ff_filt[i_95_filt]
BW_98_filt = ff_filt[i_98_filt]

print(f"Ancho de banda (filtrada) 95%: {BW_95_filt:.2f} Hz")
print(f"Ancho de banda (filtrada) 98%: {BW_98_filt:.2f} Hz")

# -------------------------------
# Gráfico: Espectro FFT con líneas de BW
# -------------------------------

Pxx_fft_db = 10 * np.log10(Pxx_fft_filt / np.max(Pxx_fft_filt))  # Normalizado al pico

plt.figure(figsize=(12, 5))
plt.plot(ff_filt, Pxx_fft_db, lw=1.5, label='Espectro (FFT, normalizado al pico)')
plt.axvline(BW_95_filt, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_filt:.1f} Hz')
plt.axvline(BW_98_filt, color='brown', linestyle='--', label=f'BW 98% = {BW_98_filt:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.title('Análisis de Ancho de Banda por Energía Acumulada (FFT - PPG filtrada)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ppg / 2)
plt.tight_layout()

# -------------------------------
# Gráfico: Curva de energía acumulada
# -------------------------------

plt.figure()
plt.plot(ff_filt, Pxx_acumulada_filt, lw=2)
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95_filt, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_filt:.1f} Hz')
plt.axvline(BW_98_filt, color='brown', linestyle='--', label=f'BW 98% = {BW_98_filt:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - PPG filtrada (FFT)')
plt.grid()
plt.legend()
plt.tight_layout()
