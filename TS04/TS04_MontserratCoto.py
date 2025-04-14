# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:32:45 2025

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
ts = 1/fs  # tiempo de muestreo
df =fs/N  # resolución espectral


R=200 # Numero de pruebas
SNR=10 #dB


#Generación de Señal

fr=np.random.uniform(-1/2,1/2,size=(1,R)) #Vector flat de [1, R]
tt = np.linspace(0, (N-1)*ts, N).reshape((N,1)) # Vector columna de [N, 1], el reshape me da formato vector de 1000 muestras por 1
vtt=np.tile(tt, (1,R)) #vector columna de [N, R]; repite R veces algo que es de [N;1]

f0=fs/4 #mitad de franja digital
f1=f0+fr*df

a1= np.sqrt(2)
xk = a1*np.sin(2*np.pi*f1*vtt) 

#Generación del ruido
Pnk=10**(-SNR/10) #Potencia del Ruido
sigma=np.sqrt(Pnk)
nk=np.random.normal(0,sigma,(N,R))

S=xk+nk

#%%FFT
for i in range(R):
    plt.plot(tt,xk[:,i])

ft_S=np.fft.fft(S,axis=0)/N #FFT por columnas
ff=np.linspace(0,(N-1)*df,N) # # grilla de sampleo frecuencial

bfrec=ff<=fs/2

# cierro ventanas anteriores
plt.close('all')
plt.figure(1)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_S[bfrec, i]) ** 2))


plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad de Potencia (dB)')
plt.title('Espectro de la señal')
plt.grid(True)
plt.show()

# #%%FFT ventanas
#%%Flattop
ventana=signal.windows.flattop(N).reshape((N,1))
SFlattop=S*ventana

ft_SFlattop=np.fft.fft(SFlattop,axis=0)/N #FFT por columnas
ff=np.linspace(0,(N-1)*df,N)

plt.figure(2)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SFlattop[bfrec, i]) ** 2))



plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad de Potencia (dB)')
plt.title('Espectro de la señal Flattop')
plt.grid(True)
plt.show()

#%%Blackmanharris
ventana=signal.windows.blackmanharris(N).reshape((N,1))
SBlackmanharris=S*ventana

ft_SBlackmanharris=np.fft.fft(SBlackmanharris,axis=0)/N #FFT por columnas, el axis 0 me hace ir por columnas
ff=np.linspace(0,(N-1)*df,N)

plt.figure(3)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SBlackmanharris[bfrec, i]) ** 2))


plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad de Potencia (dB)')
plt.title('Espectro de la señal Blackman Harris')
plt.grid(True)
plt.show()

#%%Chebwin
ventana=signal.windows.chebwin(N,30).reshape((N,1)) #30 db atenuación en decibeles (dB) de las lóbulos laterales de la ventana Chebyshev.
SChebwin=S*ventana

ft_SChebwin=np.fft.fft(SChebwin,axis=0)/N #FFT por columnas, el axis 0 me hace ir por columnas
ff=np.linspace(0,(N-1)*df,N)

plt.figure(4)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SChebwin[bfrec, i]) ** 2))


plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad de Potencia (dB)')
plt.title('Espectro de la señal Chebwin')
plt.grid(True)
plt.show()



#%%Comparación de espectros con distintas ventanas en una misma figura

plt.figure(99, figsize=(10, 12))  # Tamaño ajustado para 4 plots verticales

# FFT rectangular
plt.subplot(4,1,1)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_S[bfrec, i]) ** 2))
plt.title('FFT Rectangular')
plt.ylabel('Densidad de Potencia (dB)')
plt.grid(True)

# FFT Flattop
plt.subplot(4,1,2)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SFlattop[bfrec, i]) ** 2))
plt.title('FFT Flattop')
plt.ylabel('Densidad de Potencia (dB)')
plt.grid(True)

# FFT Blackman-Harris
plt.subplot(4,1,3)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SBlackmanharris[bfrec, i]) ** 2))
plt.title('FFT Blackman-Harris')
plt.ylabel('Densidad de Potencia (dB)')
plt.grid(True)

# FFT Chebwin
plt.subplot(4,1,4)
for i in range(R):
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_SChebwin[bfrec, i]) ** 2))
plt.title('FFT Chebwin')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad de Potencia (dB)')
plt.grid(True)

plt.tight_layout()
plt.show()

#%%Histograma de a1
#Estimador a1
a1S = np.abs(ft_S[250, :])#fila de mi señal con fft rect
a1SFlattop = np.abs(ft_SFlattop[250, :]) #fila de mi señal con fft flat
a1SBlackmanharris = np.abs(ft_SBlackmanharris[250, :]) 
a1SChebwin = np.abs(ft_SChebwin[250, :]) 


# Crear la figura
plt.figure(5)

# # Graficar cada vector por separado
plt.hist(a1S,bins=10,label='FFT RECTANGULAR',alpha=0.3)
plt.hist(a1SFlattop,bins=10,label='FFT FLATTOP',alpha=0.3)
plt.hist(a1SBlackmanharris,bins=10,label='FFT BLACKMANHARRIS',alpha=0.3)
plt.hist(a1SChebwin,bins=10,label='FFT CHEBWIN',alpha=0.3)
plt.title('Histograma del estimador a1 (por tipo de ventana)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

#%%Histograma de w1
##Estimador omega1 para ventana rectangular
# Usar solo la parte positiva del espectro
P_est1 = (1/N)*np.abs(ft_S[bfrec,:])**2  #bfrec frecuencias en Hz, hasta fs/2
omega1_est_box= np.argmax(P_est1, axis=0)*df  # índice del máximo por columna, y si lo múltiplico por df me da e Hertz

#esta=a1S-a1 ara el sesgo y la varianza
sesgo_a1_box=np.mean(a1S-a1) #Sesgo ventana rectangular
varianza_a1_box=np.var(a1S-a1) #Varianza ventana rectangular
a1SInsesgado=a1S-sesgo_a1_box #cuano hago a1S-sesgo, insesgo el error del sistema
ValorInsesgado=np.mean(a1SInsesgado-a1) #me tiene que dar 0, porque saque el sesgo
sesgo_omega1_box=np.mean(omega1_est_box-f1)
varianza_omega1_box=np.var(omega1_est_box-f1)

##Estimador omega1 para ventana Flattop
P_est2 = (1/N)*np.abs(ft_SFlattop[bfrec,:])**2  #bfrec frecuencias en Hz, hasta fs/2
omega1_est_f= np.argmax(P_est2, axis=0)*df  # índice del máximo por columna, y si lo múltiplico por df me da e Hertz
#resta=a1SFlattop-a1 para el sesgo y la varianza
sesgo_omega1_f=np.mean(omega1_est_f-f1)
varianza_omega1_f=np.var(omega1_est_f-f1)


##Estimador omega1 para ventana Blakmanharris
P_est3 = (1/N)*np.abs(ft_SBlackmanharris[bfrec,:])**2  #bfrec frecuencias en Hz, hasta fs/2
omega1_est_B= np.argmax(P_est3, axis=0)*df  # índice del máximo por columna, y si lo múltiplico por df me da e Hertz
#resta=a1SBlackmanharris-a1 para el sesgo y la varianza
sesgo_omega1_b=np.mean(omega1_est_B-f1)
varianza_omega1_b=np.var(omega1_est_B-f1)


##Estimador omega1 para ventana Chebwin
P_est4 = (1/N)*np.abs(ft_SChebwin[bfrec,:])**2  #bfrec frecuencias en Hz, hasta fs/2
omega1_est_N= np.argmax(P_est4, axis=0)*df  # índice del máximo por columna, y si lo múltiplico por df me da e Hertz
#resta=a1SBlackmanharris-a1 para el sesgo y la varianza
sesgo_omega1_n=np.mean(omega1_est_N-f1)
varianza_omega1_n=np.var(omega1_est_N-f1)


# Crear la figura
plt.figure(6)

## Graficar cada vector por separado
plt.hist(omega1_est_box, bins=10, label='FFT RECTANGULAR', alpha=0.5, color='royalblue')
plt.hist(omega1_est_f, bins=10, label='FFT FLATTOP', alpha=0.5, color='orange')
plt.hist(omega1_est_B, bins=10, label='FFT BLACKMANHARRIS', alpha=0.3, color='forestgreen')
plt.hist(omega1_est_N, bins=10, label='FFT CHEBWIN', alpha=0.3, color='crimson')

plt.title('Histograma de estimadores omega1 (por tipo de ventana)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
