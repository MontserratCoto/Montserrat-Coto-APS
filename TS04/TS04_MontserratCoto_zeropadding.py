# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 21:51:14 2025

@author: monse
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from scipy.fft import fft, fftshift

# Datos de la simulación
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000 # cantidad de muestras
Nfft = 9 * N  # nuevo tamaño de FFT para el 0 padding

ts = 1/fs  # tiempo de muestreo
df = fs / Nfft  # nueva resolución espectral para el 0 padding

R = 200 # Numero de pruebas
SNR = 10 #dB

# Generación de Señal
fr = np.random.uniform(-1/2, 1/2, size=(1, R))  # Vector flat de [1, R]
tt = np.linspace(0, (N-1)*ts, N).reshape((N, 1))  # Vector columna de [N, 1]
vtt = np.tile(tt, (1, R))  # Vector columna de [N, R]; repite R veces algo que es de [N;1]

f0 = fs / 4  # Mitad de franja digital
f1 = f0 + fr * df  # Frecuencia de las señales

a1 = np.sqrt(2)
xk = a1 * np.sin(2 * np.pi * f1 * vtt)  # Señal

# Generación del ruido
Pnk = 10**(-SNR / 10)  # Potencia del Ruido
sigma = np.sqrt(Pnk)
nk = np.random.normal(0, sigma, (N, R))

S = xk + nk

#%% FFT
# Aplicar FFT con zero padding
ft_S = np.fft.fft(S, n=Nfft, axis=0) / N  # FFT por columnas
ff = np.linspace(0, (Nfft-1) * df, Nfft)  # Grilla de sampleo frecuencial con zero padding

# Máscara para frecuencias positivas (hasta fs/2)
bfrec = ff <= fs / 2

# Gráfico del espectro
# plt.close('all')
# plt.figure(1)
# for i in range(R):
#     plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_S[bfrec, i]) ** 2))

# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad de Potencia (dB)')
# plt.title('Espectro de la señal')
# plt.grid(True)
# plt.show()

#%% FFT con ventanas
# Ventana Flattop
ventana = signal.windows.flattop(N).reshape((N, 1))
SFlattop = S * ventana

ft_SFlattop = np.fft.fft(SFlattop, n=Nfft, axis=0) / N  # FFT por columnas
# plt.figure(2)
# for i in range(R):
#     plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SFlattop[bfrec, i]) ** 2))

# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad de Potencia (dB)')
# plt.title('Espectro de la señal Flattop')
# plt.grid(True)
# plt.show()

# Ventana Blackmanharris
ventana = signal.windows.blackmanharris(N).reshape((N, 1))
SBlackmanharris = S * ventana

ft_SBlackmanharris = np.fft.fft(SBlackmanharris, n=Nfft, axis=0) / N  # FFT por columnas
# plt.figure(3)
# for i in range(R):
#     plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SBlackmanharris[bfrec, i]) ** 2))

# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad de Potencia (dB)')
# plt.title('Espectro de la señal Blackman Harris')
# plt.grid(True)
# plt.show()

# Ventana Chebwin
ventana3 = signal.windows.chebwin(N, 60).reshape((N, 1))  # 60 dB atenuación
SChebwin = S * ventana3

ft_SChebwin = np.fft.fft(SChebwin, n=Nfft, axis=0) / N  # FFT por columnas
# plt.figure(4)
# for i in range(R):
#     plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SChebwin[bfrec, i]) ** 2))

# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Densidad de Potencia (dB)')
# plt.title('Espectro de la señal Chebwin')
# plt.grid(True)
# plt.show()

#%% Histograma de omega1
## Estimador omega1 para ventana rectangular
P_est1 = (1 / N) * np.abs(ft_S[bfrec, :]) ** 2  # Estimación del espectro
omega1_est_r = np.argmax(P_est1, axis=0) * df  # Índice del máximo por columna

sesgo_omega1_r = np.mean(omega1_est_r - f1)
varianza_omega1_r = np.var(omega1_est_r - f1)

## Estimador omega1 para ventana flattop
P_est2 = (1 / N) * np.abs(ft_SFlattop[bfrec, :]) ** 2
omega1_est_F = np.argmax(P_est2, axis=0) * df

sesgo_omega1_f = np.mean(omega1_est_F - f1)
varianza_omega1_f = np.var(omega1_est_F - f1)

## Estimador omega1 para ventana Blackmanharris
P_est3 = (1 / N) * np.abs(ft_SBlackmanharris[bfrec, :]) ** 2
omega1_est_B = np.argmax(P_est3, axis=0) * df

sesgo_omega1_b = np.mean(omega1_est_B - f1)
varianza_omega1_b = np.var(omega1_est_B - f1)

## Estimador omega1 para ventana Chebwin
P_est4 = (1 / N) * np.abs(ft_SChebwin[bfrec, :]) ** 2
omega1_est_c = np.argmax(P_est4, axis=0) * df

sesgo_omega1_c = np.mean(omega1_est_c - f1)
varianza_omega1_c = np.var(omega1_est_c - f1)

# Crear la figura para el histograma
plt.figure(12)

# Graficar el histograma de omega1 por tipo de ventana
plt.hist(omega1_est_r, bins=10, label='FFT RECTANGULAR', alpha=0.3)
plt.hist(omega1_est_F, bins=10, label='FFT FLATTOP', alpha=0.5)
plt.hist(omega1_est_B, bins=10, label='FFT BLACKMANHARRIS', alpha=0.3)
plt.hist(omega1_est_c, bins=10, label='FFT CHEBWIN', alpha=0.3)
plt.title('Histograma de estimadores omega1 (por tipo de ventana con zero-padding)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
