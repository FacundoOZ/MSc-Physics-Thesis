
# EDITAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as p
from scipy.optimize import curve_fit
from numpy          import sqrt

# Módulos Propios:
from base_de_datos.conversiones import R_m
from base_de_datos.lectura      import leer_archivo_Fruchtman
from plots.estilo_plots         import disco_2D
from ajustes.Vignes             import cónica_Vignes, función_hipérbola_Vignes

# Vignes    (1997) => aprox. mínimo solar
# Fruchtman (2014) => aprox. máximo solar
# Fruchtman (2019) => aprox. mínimo solar
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_ajustes: función que permite graficar los ajustes de Vignes y Fruchtman.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_ajustes(
    directorio: str,
    #año: str
) -> None:
  disco_2D(resolución_r=200, resolución_theta=200)
  x,y,x_a,y_a = cónica_Vignes()
  p.plot(x,y, label='Vignes (1997) ss', color='grey')
  p.plot(x_a,y_a, label='Vignes (1997) ss aberrado', color='black')

  for año in ['2014', '2015', '2016', '2017', '2018', '2019']:
    data: np.ndarray = leer_archivo_Fruchtman(directorio, año)                  # Leo los archivos mag que correspondan al intervalo (t0,tf)
    Xss,Yss,Zss = [data[:,j] for j in [7,8,9]]
    p.scatter(Xss/R_m, sqrt(Yss**2+Zss**2)/R_m, s=2, label=f'Fruchtman ({año}) ss')

  p.title('Ajustes para los datos de Fruchtman (MAVEN)')
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                        # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")               # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.


# Datos Fruchtman
"""x_data = Xss / R_m
y_data = np.sqrt(Yss**2 + Zss**2) / R_m
θ_malla = np.linspace(-α, 2.3, 5000)
popt, pcov = curve_fit(
    lambda x, X0: hyperbola_vignes(x, X0, ε=ε, L=L, α=α, θ_malla=θ_malla),
    x_data,
    y_data,
    p0=[X0_inicial]
)
X0_fit   = popt[0]
sigma_X0 = np.sqrt(pcov[0,0])
"""


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def ajuste_Fruchtman(
    directorio: str,
    año: str
) -> None:
  # Gráfico MARTE:
  disco_2D(resolución_r=200, resolución_theta=200)

  # Cónica Vignes (1997):
  x,y,x_a,y_a = cónica_Vignes()
  p.plot(x,y, label='Vignes (1997) ss', color='grey')
  p.plot(x_a,y_a, label='Vignes (1997) ss aberrado', color='black')

  # Datos Fruchtman año=año:
  data: np.ndarray = leer_archivo_Fruchtman(directorio, año)                  # Leo los archivos mag que correspondan al intervalo (t0,tf)
  Xss,Yss,Zss = [data[:,j] for j in [7,8,9]]
  p.scatter(Xss/R_m, sqrt(Yss**2+Zss**2)/R_m, s=2, label=f'Fruchtman ({año}) ss')

  # Ajuste de datos Fruchtman por Vignes (solo X0 libre):
  popt, pcov = curve_fit(
    lambda x, X0: función_hipérbola_Vignes(x, X0=X0, cant_puntos=450),
    Xss/R_m,
    sqrt(Yss**2+Zss**2)/R_m,
    p0=[0.64] # X0=0.64 es el X0 hallado por Vignes => lo paso como parámetro inicial.
  )
  # Gráfico del ajuste realizado:
  x_ajuste = np.linspace(np.min(Xss/R_m), np.max(Xss/R_m), 500)
  y_ajuste = función_hipérbola_Vignes(x_ajuste, X0=popt[0], cant_puntos=450)
  p.plot(x_ajuste, y_ajuste, color='red', linewidth=2, label=f'Ajuste por Vignes (X0={popt[0]:.3f}) $R_M$')

  perr = np.sqrt(np.diag(pcov))
  print(f'Parámetros de ajuste: {popt}')
  print(f'Desviación estándar: {perr}')

  p.title('Ajustes para los datos de Fruchtman (MAVEN)')
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                        # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")               # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.


















def función_ajuste(
    x, y, funcion,
    datos_iniciales=None
):
  popt, pcov = curve_fit(funcion, x, y, p0=datos_iniciales) # Ajusto los datos a la función
  perr = np.sqrt(np.diag(pcov))

  print(f"Parámetros de ajuste: {popt}")
  print(f"Desviación estándar: {perr}")
  
  #p.errorbar(x, y, yerr=perr[1], fmt='o', color='black', label=data)
  #p.plot(grilla_x, funcion(grilla_x, *popt), color='blue', label=fit)
  return popt, pcov

# Example data
#X = np.array([1, 2, 3, 4, 5])
#t = np.array([2.1, 4.1, 6.2, 8.1, 5])

# Perform the linear fit
#popt, pcov = funcion_ajuste(X, t, funcion_lineal, eje_x='$\Delta x$ [m]', eje_y='$T$ [K]')

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————