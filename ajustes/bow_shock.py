
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión lineal
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as p
from scipy.optimize import curve_fit
from numpy          import cos, sin

# Módulos Propios:
from plots.estilo_plots import disco_2D

# Parámetros del ajuste de Vignes para el Bow Shock:
X0      = 0.64 # ± 0.02 [R_m]
epsilon = 1.03 # ± 0.01       # Excentricidad
L       = 2.04 # ± 0.02 [R_m] # Semi-latus rectum
N_b     = 450  #
alfa    = np.deg2rad(4) #° (grados)    # Ángulo de aberración con respecto al sistema MSO (SS) donde X' es opuesto al flujo medio de viento solar.
# Otros:
R_SD    = 1.64 # ± 0.08 [R_m] # Éstos dos se pueden obtener (distancia sub-solar: desde el planeta al punto de la cónica en el eje X'_ss).
R_TD    = 2.62 # ± 0.09 [R_m] # distancia sub-solar: desde el planeta al punto de la cónica en el eje Y'_ss.
#X0 = 0.78*R_m # DE DÓNDE SALE ESTO QUE USA CAMILA? NO LO ENCUENTRO EN EL PAPER DE VIGNES.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# ajustar_Vignes_MAG: función para ajustar 1 único archivo basado en una medición
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# Vignes (1997) => mínimo solar
def ajuste_Vignes_MGS() -> None:
  disco_2D(resolución_r=200, resolución_theta=200)

  theta = np.linspace(-alfa, 2.3, N_b)
  R = L / (1 + epsilon*cos(theta))
  x = (R*cos(theta) + X0) * cos(alfa) - (R*sin(theta)) * sin(alfa)
  y = (R*cos(theta) + X0) * sin(alfa) + (R*sin(theta)) * cos(alfa)
  p.plot(x,y, label='Ajuste del BS (1997)')
  p.title('Ajuste de Vignes del año 1997 (coordenadas ss aberradas) de la MGS')
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                        # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")               # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.

# Fruchtman (2014-2019) =>








#def funcion_lineal(x,a,b):
#  return a*x + b

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