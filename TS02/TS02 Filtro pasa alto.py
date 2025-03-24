# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 18:36:10 2025

@author: monse
"""

# Importamos los módulos necesarios
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# Parámetros del filtro pasa banda 
#Elegí w0=1 para normalizar y Q=1 para que los polos estén sobre el circulo unitario
Q = 1 
w0 = 1

# Coeficientes del numerador y denominador de la función de transferencia 
#Filtro pasa alto
num = np.array([ 1,0,0])
den = np.array([ 1., w0 / Q, w0**2 ])

# Función de transferencia
H1 = sig.TransferFunction(num, den)

# Respuesta en magnitud y fase (Bode plot)
frec = np.logspace(-2, 2, 200)  # Frecuencias en rad/s. 
#Especifico las frecuencias a evaluar para alinear el gráfico
w, mag, phase = sig.bode(H1,frec)

# Gráfico de la respuesta en magnitud (en dB)
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag) #crea un gráfico con una escala logarítmica en el eje x y una escala lineal en el eje y.
plt.title("Respuesta en Magnitud")
plt.xlabel("Frecuencia [rad/s]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)

# Gráfico de la respuesta en fase (en radianes)
plt.subplot(2, 1, 2)
plt.semilogx(w, np.deg2rad(phase)) # Conversión de la fase de grados a radianes
plt.title("Respuesta en Fase")
plt.xlabel("Frecuencia [rad/s]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout() #Ajuste de los margenes
plt.show()

# Mapa de ceros y polos
zeros, poles, _ = sig.tf2zpk(num, den)
plt.figure(figsize=(8, 8))  # Tamaño de la figura
plt.plot(np.real(zeros), np.imag(zeros), 'go', label="Ceros")  # Ceros en verde
plt.plot(np.real(poles), np.imag(poles), 'rx', label="Polos")  # Polos en rojo
plt.xlim(-1.5, 1.5) # Los límites de los ejes para asegurarnos de ver la circunferencia
plt.ylim(-1.5, 1.5)
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', label="Circunferencia Unitaria") # Ddistribución circular
plt.gca().add_artist(circle)
# Título y etiquetas
plt.title("Mapa de Ceros y Polos")
plt.xlabel(r'$\theta$')
plt.ylabel("jw")
plt.grid(True)
# Líneas horizontales y verticales en 0
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
# Leyenda
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # Asegura que la relación de aspecto sea 1:1
plt.show()