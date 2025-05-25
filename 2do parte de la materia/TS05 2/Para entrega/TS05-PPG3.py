# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:15:54 2025

@author: monse
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cierro ventanas anteriores
plt.close('all')

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

plt.figure()
plt.plot(ppg)

N_ppg = len(ppg)

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

# ---------------------------------
# 2. Cálculo del ancho de banda usando FFT con normalización a potencia 1
# ---------------------------------

ppg_norm = ppg / np.std(ppg)
ppg_detrended = ppg_norm - np.mean(ppg_norm)  # Quitar componente DC

N = len(ppg_detrended)
fft_ppg = np.fft.rfft(ppg_detrended)
Pxx_fft = np.abs(fft_ppg)**2

ff = np.fft.rfftfreq(N, 1/fs_ppg)

Pxx_norm = Pxx_fft / np.sum(Pxx_fft)  # Normalización total potencia = 1

Pxx_acumulada = np.cumsum(Pxx_norm)

i_95 = np.where(Pxx_acumulada >= 0.95)[0][0]
i_98 = np.where(Pxx_acumulada >= 0.98)[0][0]

BW_95 = ff[i_95]
BW_98 = ff[i_98]

print(f"Ancho de banda 95%: {BW_95:.2f} Hz")
print(f"Ancho de banda 98%: {BW_98:.2f} Hz")

plt.figure(figsize=(12, 5))
plt.plot(f_welch, Pxx_welch_db, lw=1.5, label='PSD (Welch, normalizado al pico)')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.title('Análisis de Ancho de Banda por Potencia Acumulada (en PSD Welch)')
plt.grid()
plt.legend()
plt.xlim(0, fs_ppg / 2)
plt.tight_layout()

plt.figure()
plt.plot(ff, Pxx_acumulada, lw=2)
plt.axhline(0.95, color='orangered', linestyle='--', label='95% de energía')
plt.axhline(0.98, color='brown', linestyle='--', label='98% de energía')
plt.axvline(BW_95, color='orangered', linestyle='--', label=f'BW 95% = {BW_95:.1f} Hz')
plt.axvline(BW_98, color='brown', linestyle='--', label=f'BW 98% = {BW_98:.1f} Hz')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Energía acumulada (normalizada)')
plt.title('Curva de Energía Acumulada - PPG (FFT)')
plt.grid()
plt.legend()

# ---------------------------------------
# 3. Filtrado pasa banda Butterworth 4to orden (0.3 a 10 Hz)
# ---------------------------------------

f_low = 0.3
f_high = 10
nyq = fs_ppg / 2

b, a = signal.butter(4, [f_low / nyq, f_high / nyq], btype='bandpass')
ppg_filtrada = signal.filtfilt(b, a, ppg_norm)

t_ppg = np.arange(N_ppg) / fs_ppg

plt.figure(figsize=(12, 6))
plt.plot(t_ppg, ppg_norm, alpha=0.4, label='PPG normalizada (sin filtrar)')
plt.plot(t_ppg, ppg_filtrada, label='PPG filtrada (0.3 - 10 Hz)', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal PPG antes y después del filtrado')
plt.legend()
plt.grid()
plt.tight_layout()

# PSD para señal filtrada
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
