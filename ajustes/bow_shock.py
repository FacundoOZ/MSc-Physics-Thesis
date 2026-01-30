
# MODULARIZAR Y COMENTAR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from typing            import Union
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit

# Módulos Propios:
from base_de_datos.conversiones import R_m, módulo
from base_de_datos.lectura      import leer_archivo_Fruchtman, leer_archivos_MAG, leer_bow_shocks_KNN
from plots.estilo_plots         import disco_2D
from base_de_datos.recorte      import preparar_región_Vignes
from ajustes.Vignes             import (hipérbola_Vignes, función_hipérbola_Vignes, hipérbola_mínima, hipérbola_máxima,
                                        máximo_2015, mínimo_2019)

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_ajustes(
    directorio: str,
    objetos: str,
    ajuste_Fruchtman: bool = False,
    trayectoria: bool=False,
    recorte: str = 'recorte_Vignes',
    tiempo_inicial: str='01/01/2015-00:00:00', tiempo_final: str='30/3/2015-23:59:00',
    promedio: int = 1
) -> None:
  if 'marte' in objetos:                                                                   #
    disco_2D(resolución_r=200, resolución_theta=200)                                       #
  if 'Vignes' in objetos:                                                                  #
    x, y, x_a, y_a = hipérbola_Vignes()                                                    #
    #p.plot(x,y, label=r'Vignes (1997) ss', color='black')                                 # Sin aberración
    p.plot(x_a,y_a, label=r'Vignes (1997) ss ($\alpha = 4$°)', color='black')              #
  if 'Fruchtman' in objetos:                                                               # DATOS FRUCHTMAN
    for año in ['2014','2015','2016','2017','2018','2019']:                                #
      data_Fru: pd.DataFrame = leer_archivo_Fruchtman(directorio, año, hemisferio_N=False) # Leo archivos MAG que correspondan al intervalo (t0,tf)
      Xss, Yss, Zss = [data_Fru[j] for j in [7,8,9]]                                       #
      p.scatter(Xss/R_m, módulo(Yss, Zss, norm=R_m), s=2, label=f'Fruchtman ({año}) ss')   #
      if ajuste_Fruchtman:                                                                 # AJUSTE NO LINEAL POR HIPÉRBOLA VIGNES (sobre datos Fruchtman)
        ajustar_Fruchtman_por_Vignes(Xss,Yss,Zss, año)
    if 'mín' in objetos:                                                                   #
      p.scatter(mínimo_2019()[0], mínimo_2019()[1],                                        #
                marker='x', s=200, color='purple', label='mínimo 2019')                    #
    if 'máx' in objetos:                                                                   #
      p.scatter(máximo_2015()[0], máximo_2015()[1],                                        #
                marker='x', s=200, color='red', label='máximo 2015')                       #
  if 'región' in objetos:                                                                  #
    x_a_min, y_a_min = hipérbola_mínima(); x_a_max, y_a_max = hipérbola_máxima()           # Hipérbola izquierda y derecha, respectivamente
    p.plot(x_a_min,y_a_min, label=r'Curva mín: ($X_0=0.14, \alpha=20$°)', color='purple')  #
    p.plot(x_a_max,y_a_max, label=r'Curva máx: ($X_0=1.4, \alpha=-10$°)', color='red')     #
    dibujar_segmento(región=preparar_región_Vignes())                                      #
    sombrear_área(x_a_min, y_a_min, x_a_max, y_a_max)                                      # SOMBREADO
  if trayectoria:                                                                          # TRAYECTORIA MAVEN
    data_MAG: pd.DataFrame = leer_archivos_MAG(os.path.join(directorio, recorte),          #
                                               tiempo_inicial, tiempo_final, promedio)     #
    Xss,Yss,Zss = [data_MAG[j] for j in [7,8,9]]                                           #
    p.plot(Xss/R_m, módulo(Yss, Zss, norm=R_m))                                            #
  if 'KNN' in objetos:                                                                     # BOW SHOCKS KNN
    data_BS: pd.DataFrame = leer_bow_shocks_KNN(directorio, '2014')                        #
    Xss,Yss,Zss = [data_BS[j] for j in [7,8,9]]                                            #
    p.scatter(Xss/R_m, módulo(Yss, Zss, norm=R_m), s=1)                                    #
  p.title('Ajustes para los datos de Fruchtman (MAVEN)')                                   #
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]")                                                    # coloco labels tipo SS en x
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]")                          # y en y.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                                # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                               # Escribo los labels
  #guardar_figura()                                                                        # guardo la figura
  p.show()                                                                                 # Enseño el plot.


def ajustar_Fruchtman_por_Vignes(Xss, Yss, Zss, año) -> None:
  popt, pcov = curve_fit(
    lambda x, X0: función_hipérbola_Vignes(x, X0=X0, cant_puntos=450),
    Xss/R_m,
    módulo(Yss,Zss,norm=R_m),
    p0=[0.64] # X0=0.64 es el X0 hallado por Vignes => lo paso como parámetro inicial.
  )
  x_ajuste = np.linspace(np.min(Xss/R_m), np.max(Xss/R_m), 500)
  y_ajuste = función_hipérbola_Vignes(x_ajuste, X0=popt[0], cant_puntos=450)
  p.plot(x_ajuste, y_ajuste, linewidth=2, label=f'Ajuste Vignes {año}: X0={popt[0]:.3f} $R_M$')
  perr = np.sqrt(np.diag(pcov))
  print(f'Parámetros de ajuste: {popt}')
  print(f'Desviación estándar: {perr}')



def dibujar_segmento(región) -> None:
  recta, y_A, y_B = [región[v] for v in ['recta','y_A','y_B']]
  y_grilla = np.linspace(y_A, y_B, 300)
  p.plot(recta(y_grilla), y_grilla, color='green')

def sombrear_área(Xmin, Ymin, Xmax, Ymax) -> None:
  # REGIÓN SOMBREADA
  j_min, j_max = np.argsort(Ymin), np.argsort(Ymax)  # Sort curves by y (required for interpolation)
  Y = np.linspace(                                    # Common y-grid where both curves exist
    max(Ymin[j_min].min(), Ymax[j_max].min()),
    min(Ymin[j_min].max(), Ymax[j_max].max()),
    1000  # ↑ increase for more smoothness
  )
  x_max_int = interp1d(Ymax[j_max], Xmax[j_max], kind='linear')  # Interpolators
  x_min_int = interp1d(Ymin[j_min], Xmin[j_min], kind='linear')
  p.fill_betweenx(Y, x_min_int(Y), x_max_int(Y), color='green', alpha=0.3, linewidth=0, label='Región de interés')
  x_polígono = np.concatenate(([Xmin[-2]], Xmax[Ymax > Ymin[-2]], [Xmin[-2]]))
  y_polígono = np.concatenate(([Ymin[-2]], Ymax[Ymax > Ymin[-2]], [Ymin[-2]]))
  p.fill(x_polígono, y_polígono, color='green', alpha=0.3, linewidth=0)




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