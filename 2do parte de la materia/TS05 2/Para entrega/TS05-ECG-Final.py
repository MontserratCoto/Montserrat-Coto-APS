# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:54:28 2025

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

##################
# Lectura de ECG #
##################
fs_ecg = 1000  # Hz
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])

# Usar solo las primeras 12000 muestras
ecg_one_lead = ecg_one_lead[:12000]
N_ecg = len(ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead[5000:12000])
plt.xlabel('Tiempo (ms)')   # 12000 muestras a 1000 Hz = 12 segundos, acá mostramos de 5000 a 12000 (7 segundos)
plt.ylabel('Amplitud (mV)')
plt.title('Señal ECG')

############
# 1. PSD con Welch (normalizada al pico en dB)
#############################

nperseg_ecg = int(N_ecg / 6)



ecg_f_welch, ecg_Pxx_welch = signal.welch(
    ecg_one_lead[:, 0],
    fs=fs_ecg,
    window='hann',
    nperseg=nperseg_ecg,
    noverlap=nperseg_ecg // 2,
    detrend=False,
    scaling='density'
)

# Normalización al pico (en dB)
ecg_Pxx_welch_db = 10 * np.log10(ecg_Pxx_welch)
ecg_Pxx_welch_db -= np.max(ecg_Pxx_welch_db)

# Gráfico PSD
plt.figure()
plt.plot(ecg_f_welch, ecg_Pxx_welch_db, label='ECG (Welch, normalizada al pico)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('PSD - ECG (normalizada al pico)')
plt.grid()
plt.xlim(0, fs_ecg / 2)
plt.legend()

#############################
# 2. Análisis de BW con FFT directa (ECG)
#############################

# Normalización a potencia total 1 W
ecg_signal = ecg_one_lead[:, 0]
ecg_norm = ecg_signal / np.std(ecg_signal)

# Detrending (opcional)
ecg_detrended = ecg_norm - np.mean(ecg_norm)

# FFT directa y potencia
N_ecg = len(ecg_detrended)
fft_ecg = np.fft.rfft(ecg_detrended)
Pxx_fft_ecg = np.abs(fft_ecg) ** 2

# Frecuencia asociada
ff_ecg = np.fft.rfftfreq(N_ecg, 1/fs_ecg)

# Normalización a potencia unitaria
Pxx_ecg_norm = Pxx_fft_ecg / np.sum(Pxx_fft_ecg)

# Energía acumulada
Pxx_ecg_acum = np.cumsum(Pxx_ecg_norm)

# Cálculo de BW al 95% y 98%
i_95_ecg = np.where(Pxx_ecg_acum >= 0.95)[0][0]
i_98_ecg = np.where(Pxx_ecg_acum >= 0.98)[0][0]
BW_95_ecg = ff_ecg[i_95_ecg]
BW_98_ecg = ff_ecg[i_98_ecg]

print(f"BW 95% ECG (FFT): {BW_95_ecg:.2f} Hz")
print(f"BW 98% ECG (FFT): {BW_98_ecg:.2f} Hz")

# ---------------------------------
# Gráfico 1: PSD (FFT) con líneas de BW
# ---------------------------------
plt.figure(figsize=(12, 5))
plt.plot(ff_ecg, 10 * np.log10(Pxx_ecg_norm), lw=1.5, label='PSD (FFT, normalizada a potencia total)')
plt.axvline(BW_95_ecg, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_ecg:.1f} Hz')
plt.axvline(BW_98_ecg, color='brown', linestyle='--', label=f'BW 98% = {BW_98_ecg:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('Análisis de Ancho de Banda por Potencia Acumulada - ECG (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ecg / 2)
plt.tight_layout()

# ---------------------------------
# Gráfico 2: Energía acumulada (FFT)
# ---------------------------------
plt.figure(figsize=(12, 4))
plt.plot(ff_ecg, Pxx_ecg_acum, lw=2, label='Energía acumulada (normalizada)')
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95_ecg, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_ecg:.1f} Hz')
plt.axvline(BW_98_ecg, color='brown', linestyle='--', label=f'BW 98% = {BW_98_ecg:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - ECG (FFT)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ecg / 2)
plt.tight_layout()
