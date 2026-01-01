
# Editar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión lineal y no-lineal
#============================================================================================================================================

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def funcion_lineal(x,a,b):
  return a*x + b

def funcion_cuadratica(x,a,b,c):
  return a*(x**2) + b*x + c

def funcion_exponencial(x,a,b,c):
  return a*(np.exp(b*x)) + c

def funcion_seno(x,a,b,c,d):
  return a*(np.sin(b*x+c)) + d

def funcion_ajuste(
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
popt, pcov = funcion_ajuste(X, t, funcion_cuadratica, eje_x='$\Delta x$ [m]', eje_y='$T$ [K]')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————