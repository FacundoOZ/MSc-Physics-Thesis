
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from scipy.optimize import curve_fit
from numpy          import sqrt, cos, sin

# Módulos Propios:
from plots.estilo_plots import disco_2D
from base_de_datos.conversiones import R_m

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

# PARÁMETROS VIGNES
theta = np.linspace(-alfa, 2.3, N_b)
R = L / (1 + epsilon*cos(theta))
x = R*cos(theta) + X0                # Coordenadas SS (x,y,z)
y = R*sin(theta)
x_a = x*cos(alfa) - y*sin(alfa)      # Coordenadas SS aberradas (x',y',z')
y_a = x*sin(alfa) + y*cos(alfa)

# Vignes (1997) => mínimo solar
def ajuste_Vignes_MGS() -> None:
  disco_2D(resolución_r=200, resolución_theta=200)
  p.plot(x,y, label='Ajuste Vignes (MGS, 1997) en SS', color='grey')
  p.plot(x,y, label='Ajuste Vignes (MGS, 1997) en SS aberrado', color='black')
  p.title('Ajuste del Bow Shock de Vignes (MGS)')
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                        # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")               # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.

# Fruchtman (2014-2019) =>

def graficador_datos_fruchtman(
    directorio: str,
    #año: str
) -> None:
  disco_2D(resolución_r=200, resolución_theta=200)
  p.plot(x,y, label='Ajuste Vignes (MGS, 1997) en SS', color='grey')
  p.plot(x_a,y_a, label='Ajuste Vignes (MGS, 1997) en SS aberrado', color='black')

  for año in ['2014', '2015', '2016', '2017', '2018', '2019']:
    data: np.ndarray = leer_archivos_fruchtman(directorio, año)                  # Leo los archivos mag que correspondan al intervalo (t0,tf)
    Xss,Yss,Zss = [data[:,j] for j in [7,8,9]]
    p.scatter(Xss/R_m, sqrt(Yss**2+Zss**2)/R_m, s=2, label=f'Fruchtman (MAVEN, {año}) en SS')

  p.title('Ajustes para los datos de Fruchtman (MAVEN)')
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                        # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")               # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.







def hyperbola_vignes(x_data, X0, *, epsilon, L, alfa, theta_grid):
  """
  Modelo de Vignes con un solo parámetro libre: X0.
  epsilon, L y alfa están fijos.
  """
  R = L / (1 + epsilon * np.cos(theta_grid))
  x = R * np.cos(theta_grid) + X0
  y = R * np.sin(theta_grid)
  x_a = x * np.cos(alfa) - y * np.sin(alfa)  # Aberración
  y_a = x * np.sin(alfa) + y * np.cos(alfa)
  y_model = np.empty_like(x_data)            # Para cada x observado, busco el punto más cercano en x_a
  for i, xi in enumerate(x_data):
    j = np.argmin(np.abs(x_a - xi))
    y_model[i] = np.abs(y_a[j])
  return y_model

# Datos Fruchtman
x_data = Xss / R_m
y_data = np.sqrt(Yss**2 + Zss**2) / R_m

theta_grid = np.linspace(-alfa, 2.3, 5000)

popt, pcov = curve_fit(
    lambda x, X0: hyperbola_vignes(x, X0, epsilon=epsilon, L=L, alfa=alfa, theta_grid=theta_grid),
    x_data,
    y_data,
    p0=[X0_inicial]
)

X0_fit   = popt[0]
sigma_X0 = np.sqrt(pcov[0,0])




















#———————————————————————————————————————————————————————————————————————————————————————
# Funciones auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def leer_archivos_fruchtman(
    directorio: str,        # Directorio base donde se encuentra la carpeta "merge".
    año: str                # Año del archivo a leer (string o convertible a string).
) -> np.ndarray:
  """
  La función leer_archivos_fruchtman recibe un directorio base y un año, y lee el archivo 'fruchtman_{año}_merge.sts' ubicado dentro de la carpeta 'merge' correspondiente.
  Devuelve un np.ndarray con los datos cargados.
  """
  nombre_archivo: str = f'fruchtman_{año}_merge.sts'            # Nombre del archivo.
  ruta_archivo: str = os.path.join(directorio, 'merge', nombre_archivo)  # Ruta completa.
  datos: np.ndarray = np.loadtxt(ruta_archivo)                  # Cargo el archivo.
  return datos                                                   # Devuelvo los datos.








def funcion_lineal(x,a,b):
  return a*x + b

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
    p.errorbar(x, y, yerr=perr[1], fmt='o', color='black', label=data)
    p.plot(grilla_x, funcion(grilla_x, *popt), color='blue', label=fit)
    p.xlabel(eje_x)
    p.ylabel(eje_y)
    p.legend()
    p.grid(True)
    p.show()
  return popt, pcov

# Example data
#X = np.array([1, 2, 3, 4, 5])
#t = np.array([2.1, 4.1, 6.2, 8.1, 5])

# Perform the linear fit
#popt, pcov = funcion_ajuste(X, t, funcion_lineal, eje_x='$\Delta x$ [m]', eje_y='$T$ [K]')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————