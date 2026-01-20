
# MODULARIZAR, DOCUMENTAR Y COMENTAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión
#============================================================================================================================================

import numpy             as np
import matplotlib.pyplot as p
from numpy             import sqrt
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit

# Módulos Propios:
from base_de_datos.conversiones import R_m
from base_de_datos.lectura      import leer_archivo_Fruchtman, leer_archivos_MAG
from plots.estilo_plots         import disco_2D
from base_de_datos.recorte      import preparar_región_Vignes
from ajustes.Vignes             import (hipérbola_Vignes, función_hipérbola_Vignes, hipérbola_mínima, hipérbola_máxima,
                                        máximo_2015, mínimo_2019, mínimo_2015)

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_ajustes(
    directorio: str,
    #tiempo_inicial: str, tiempo_final: str
    #año: str,
    #promedio: int = 1
) -> None:

  #——————————————————————————————————————————————————————————————————————————————
  # MARTE
  disco_2D(resolución_r=200, resolución_theta=200)
  #——————————————————————————————————————————————————————————————————————————————
  # HIPÉRBOLA VIGNES
  x,y,x_a,y_a = hipérbola_Vignes()
  #p.plot(x,y, label=r'Vignes (1997) ss', color='black')
  p.plot(x_a,y_a, label=r'Vignes (1997) ss ($\alpha = 4$°)', color='black')
  #——————————————————————————————————————————————————————————————————————————————
  # HIPÉRBOLA IZQUIERDA
  x_a_min,y_a_min = hipérbola_mínima()
  p.plot(x_a_min, y_a_min, label=r'Curva mín: ($X_0 = 0.14, \alpha=20$°)', color='purple')
  #——————————————————————————————————————————————————————————————————————————————
  # HIPÉRBOLA DERECHA
  x_a_max,y_a_max = hipérbola_máxima()
  p.plot(x_a_max, y_a_max, label=r'Curva máx: ($X_0 = 1.4, \alpha=-10$°)', color='red')
  #——————————————————————————————————————————————————————————————————————————————
  # SEGMENTO IZQUIERDO
  región = preparar_región_Vignes()
  recta = región['recta']
  y_A = región['y_A']
  y_B = región['y_B']
  y_grilla = np.linspace(y_A, y_B, 300)
  p.plot(recta(y_grilla), y_grilla, color='green')
  #——————————————————————————————————————————————————————————————————————————————

  # REGIÓN SOMBREADA
  #——————————————————————————————————————————————————————————————————————————————
  j_max = np.argsort(y_a_max)                         # Sort curves by y (required for interpolation)
  j_min = np.argsort(y_a_min)
  Y = np.linspace(                                    # Common y-grid where both curves exist
    max(y_a_min[j_min].min(), y_a_max[j_max].min()),
    min(y_a_min[j_min].max(), y_a_max[j_max].max()),
    1000  # ↑ increase for more smoothness
  )
  x_max_int = interp1d(y_a_max[j_max], x_a_max[j_max], kind='linear')  # Interpolators
  x_min_int = interp1d(y_a_min[j_min], x_a_min[j_min], kind='linear')
  p.fill_betweenx(Y, x_min_int(Y), x_max_int(Y), color='green', alpha=0.3, linewidth=0, label='Región de interés')
  x_polígono = np.concatenate(([x_a_min[-2]], x_a_max[y_a_max > y_a_min[-2]], [x_a_min[-2]]))
  y_polígono = np.concatenate(([y_a_min[-2]], y_a_max[y_a_max > y_a_min[-2]], [y_a_min[-2]]))
  p.fill(x_polígono, y_polígono, color='green', alpha=0.3, linewidth=0)
  #——————————————————————————————————————————————————————————————————————————————

  """data = leer_archivos_MAG(directorio, tiempo_inicial, tiempo_final, promedio)
  Xss,Yss,Zss = [data[j].to_numpy() for j in [7,8,9]]
  A = Xss/R_m
  B = sqrt(Yss**2+Zss**2)/R_m
  p.scatter(A,B, s=1)"""

  # DATOS FRUCHTMAN
  #——————————————————————————————————————————————————————————————————————————————
  for año in ['2014','2015','2016','2017','2018','2019']:
    data: np.ndarray = leer_archivo_Fruchtman(directorio, año)                  # Leo los archivos mag que correspondan al intervalo (t0,tf)
    Xss,Yss,Zss = [data[:,j] for j in [7,8,9]]
    A = Xss/R_m
    B = sqrt(Yss**2+Zss**2)/R_m
    p.scatter(A,B, s=2, label=f'Fruchtman ({año}) ss')
  #——————————————————————————————————————————————————————————————————————————————

    #——————————————————————————————————————————————————————————————————————————————
    # AJUSTE NO LINEAL POR HIPÉRBOLA VIGNES (sobre datos Fruchtman)
    """popt, pcov = curve_fit(
      lambda x, X0: función_hipérbola_Vignes(x, X0=X0, cant_puntos=450),
      Xss/R_m,
      sqrt(Yss**2+Zss**2)/R_m,
      p0=[0.64] # X0=0.64 es el X0 hallado por Vignes => lo paso como parámetro inicial.
    )
    x_ajuste = np.linspace(np.min(Xss/R_m), np.max(Xss/R_m), 500)
    y_ajuste = función_hipérbola_Vignes(x_ajuste, X0=popt[0], cant_puntos=450)
    p.plot(x_ajuste, y_ajuste, color='red', linewidth=2, label=f'Ajuste por Vignes (X0={popt[0]:.3f}) $R_M$')
    perr = np.sqrt(np.diag(pcov))
    print(f'Parámetros de ajuste: {popt}')
    print(f'Desviación estándar: {perr}')"""
    #——————————————————————————————————————————————————————————————————————————————

  #p.scatter(mínimo_2015()[0], mínimo_2015()[1], s=10, color='black', label='mínimo 2015')
  p.scatter(mínimo_2019()[0], mínimo_2019()[1], marker='x', s=200, linewidth=1, color='purple', label='mínimo 2019')
  p.scatter(máximo_2015()[0], máximo_2015()[1], marker='x', s=200, linewidth=1, color='red', label='máximo 2015')
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

  print(f'Parámetros de ajuste: {popt}')
  print(f'Desviación estándar: {perr}')
  
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