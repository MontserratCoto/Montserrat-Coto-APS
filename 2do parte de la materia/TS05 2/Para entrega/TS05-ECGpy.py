# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:31:57 2025

@author: monse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio

def vertical_flaten(a):
    return a.reshape(a.shape[0], 1)

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


hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

plt.figure()
plt.plot(hb_1)

plt.figure()
plt.plot(hb_2)
#############################
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
# 2. Análisis de BW con FFT directa
#############################

# Normalización a potencia total 1 W
ecg_norm = ecg_one_lead[:, 0] / np.std(ecg_one_lead[:, 0])

# FFT directa
fft_ecg = np.fft.fft(ecg_norm)
Pxx_fft_ecg = np.abs(fft_ecg) ** 2

# Frecuencia asociada
df_ecg = fs_ecg / N_ecg
ff_ecg = np.linspace(0, (N_ecg - 1) * df_ecg, N_ecg)
bfrec_ecg = ff_ecg <= fs_ecg / 2

ff_ecg = ff_ecg[bfrec_ecg]
Pxx_fft_ecg = Pxx_fft_ecg[bfrec_ecg]

# Normalización en potencia total
Pxx_ecg_norm = Pxx_fft_ecg / np.sum(Pxx_fft_ecg)

# Energía acumulada
Pxx_ecg_acum = np.cumsum(Pxx_ecg_norm)

# Cálculo de BW al 95% y 98%
i_95_ecg = np.where(Pxx_ecg_acum >= 0.95)[0][0]
i_98_ecg = np.where(Pxx_ecg_acum >= 0.98)[0][0]
BW_95_ecg = ff_ecg[i_95_ecg]
BW_98_ecg = ff_ecg[i_98_ecg]

# Gráfico PSD con líneas de BW
plt.figure(figsize=(12, 5))
plt.plot(ecg_f_welch, ecg_Pxx_welch_db, lw=1.5, label='PSD (Welch, normalizada al pico)')
plt.axvline(BW_95_ecg, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_ecg:.1f} Hz')
plt.axvline(BW_98_ecg, color='brown', linestyle='--', label=f'BW 98% = {BW_98_ecg:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Análisis de Ancho de Banda - ECG')
plt.grid()
plt.legend()
plt.xlim(0, fs_ecg / 2)
plt.tight_layout()
plt.show()

# Gráfico energía acumulada
plt.figure()
plt.plot(ff_ecg, Pxx_ecg_acum, lw=2)
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95_ecg, color='orangered', linestyle='--', label=f'BW 95% = {BW_95_ecg:.1f} Hz')
plt.axvline(BW_98_ecg, color='brown', linestyle='--', label=f'BW 98% = {BW_98_ecg:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - ECG (FFT)')
plt.grid()
plt.legend()
