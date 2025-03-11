# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:44:11 2025

@author: monse
"""

#TS01: Programar una función que genere señales senoidales que permita parametrizar:
# la amplitud máxima de la senoidal (volts)
# su valor medio (volts)
# la frecuencia (Hz)
# la fase (radianes)
# la cantidad de muestras digitalizada por el ADC (# muestras)
# la frecuencia de muestreo del ADC.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Defino mis parámetros
vmax = 1 #la amplitud máxima de la senoidal (volts)
dc = 0 #su valor medio (volts)
ff = 500 #la frecuencia fundamental de la onda (Hz)
ph = 0 #la fase (radianes)
N = 1000 #la cantidad de muestras digitalizada por el ADC (# muestras)
fs = 1000 #la frecuencia de muestreo del ADC
ts = 1/fs #tiempo entre cada muestra
#fs nos indica cuantas muestras se toman por segundo, 
#su inversa el tiempo de esas muestras.
#fs la frecuencia con la que el conversor analógico-digital (ADC) toma muestras, 
#dependiendo el conversor es el tipo de sampleo que puedo hacer
#es mi variable limitante

# Defino la función pedida
def mi_funcion_sen(vmax, dc, ff, ph, N, fs):
   #tt es el vector de instantes de tiempo en los que se toman las muestras de la señal
    tt = np.linspace(0, (N-1)*ts, num=N) #Vector de N posiciones
    # Argumento del seno: 2π*frecuencia*tiempo + fase
    arg=2*np.pi*ff*tt + ph
    #Señal senoidal
    xx = vmax * np.sin(arg) + dc
    return tt, xx  
#tt guarda los tiempos en los que se toman las muestras.
#xx guarda los valores de la señal en esos tiempos.

# Llamo a la función usando las variables definidas antes
tt, xx = mi_funcion_sen(vmax, dc, ff, ph, N, fs)

# Grafico
plt.plot(tt, xx)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.title("Señal Senoidal")
plt.grid()
plt.show()
