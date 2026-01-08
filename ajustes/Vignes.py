
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión lineal
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from plots.MAG import R_m

"""Mars Global Surveyor se lanzó el 7 de noviembre de 1996, pero comenzó a colectar datos desde 1997 hasta 2006. Entonces cuando Vignes dice que analizó los datos del primer año de la misión supongo 1997.
El mínimo solar de esa década del Sol se encontró en Mayo y Agosto del 1996, luego, a lo largo de 1997 comenzó a subir levemente.
Podemos aproximar el ajuste de Vignes como el de un mínimo solar.
En ésta década, el mínimo solar se produjo en el año 2019 con su mínimo más pronunciado en la segunda mitad del año.
Entonces, puedo ajustar los datos de 2019 de fruchtman y ver si son parecidos a los de Vignes, con esos tengo una curva del mínimo para todo el bow shock.

Por otra parte, el máximo de ésta década fue el último mes de 2013, y a lo largo de los primeros 6 meses de 2014 aproximadamente. De fruchtman se tienen mediciones de los últimos 45 días del año 2014, por lo que podemos ajustar esos para tener el bow shock del máximo.
Aún así, cabe aclarar que el máximo de esta década fue pequeño, prácticamente la mitad del del año 2000, y menor que el ocurrido a fines del 2024.
"""




# Parámetros del ajuste de Vignes para el Bow Shock:
X0     = 0.64 # ± 0.02 [R_m]
epsilon = 1.03 # ± 0.01       # Excentricidad
L       = 2.04 # ± 0.02 [R_m] # Semi-latus rectum

#X0 = 0.78*R_m # DE DÓNDE SALE ESTO QUE USA CAMILA? NO LO ENCUENTRO EN EL PAPER DE VIGNES.

# Otros:
R_SD    = 1.64 # ± 0.08 [R_m] # Éstos dos se pueden obtener (distancia sub-solar: desde el planeta al punto de la cónica en el eje X'_ss).
R_TD    = 2.62 # ± 0.09 [R_m] # distancia sub-solar: desde el planeta al punto de la cónica en el eje Y'_ss.
alfa    = 4    #° (grados)    # Ángulo de aberración con respecto al sistema MSO (SS) donde X' es opuesto al flujo medio de viento solar.
N_b     = 450  # 

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# ajustar_Vignes_MAG: función para ajustar 1 único archivo basado en una medición
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


def ajustar_Vignes_MAG(
    directorio: str,                                                                   # Carpeta donde se encuentra el archivo a recortar.
    archivo: str,                                                                      # Nombre del archivo en formato string a recortar.
    coord: str = 'pc'                                                                  # Tipo de coordenadas del archivo a recortar.
) -> None:

  return










def funcion_lineal(x,a,b):
  return a*x + b

"""def funcion_ajuste(
    x, y, funcion, eje_x='x', eje_y='y',
    data='Datos', fit='Ajuste', datos_iniciales=None, plot=True
):
  popt, pcov = curve_fit(funcion, x, y, p0=datos_iniciales) # Ajusto los datos a la función
  perr = np.sqrt(np.diag(pcov))

  print(f"Parámetros de ajuste: {popt}")
  print(f"Desviación estándar: {perr}")
  
  if plot:
    grilla_x = np.linspace(min(x),max(x),100)
    plt.errorbar(x, y, yerr=perr[1], fmt='o', color='black', label=data)
    plt.plot(grilla_x, funcion(grilla_x, *popt), color='blue', label=fit)
    plt.xlabel(eje_x)
    plt.ylabel(eje_y)
    plt.legend()
    plt.grid(True)
    plt.show()
  return popt, pcov

# Example data
X = np.array([1, 2, 3, 4, 5])
t = np.array([2.1, 4.1, 6.2, 8.1, 5])

# Perform the linear fit
popt, pcov = funcion_ajuste(X, t, funcion_cuadratica, eje_x='$\Delta x$ [m]', eje_y='$T$ [K]')"""

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————