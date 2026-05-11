
#============================================================================================================================================
# Tesis de Licenciatura | Archivo para estudiar modelos de regresión
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from typing            import Any, Union
from scipy.interpolate import interp1d
from scipy.optimize    import curve_fit

# Módulos Propios:
from base_de_datos.conversiones import R_m, módulo
from base_de_datos.lectura      import leer_archivo_Fruchtman, leer_archivos_MAG, leer_bow_shocks_KNN
from plots.estilo_plots         import disco_2D, guardar_figura
from base_de_datos.recorte      import preparar_región_Vignes
from ajustes.Vignes             import (hipérbola_Vignes, función_hipérbola_Vignes, hipérbola_mínima, hipérbola_máxima,
                                        máximo_2015, mínimo_2019)

# Offsets seteados manualmente para mayor visibilización de los ajustes Vignes para los BS's de Fruchtman (ambos hemisferios y hemisferio_N)
offsets_Fruchtman: dict[str,list[int]] = {
  '2014': [910,1000],
  '2015': [1000,980],
  '2016': [962,950],
  '2017': [980,1000],
  '2018': [1000,990],
  '2019': [950,940]
}

# Offsets seteados manualmente para mayor visibilización de los ajustes Vignes para los BS's detectados por el KNN
offsets_KNN: dict[str,int] = {
  '2014': 1000,
  '2015': 977,
  '2016': 900,
  '2017': 890,
  '2018': 850,
  '2019': 800,
  '2020': 862,
  '2021': 823,
  '2022': 841,
  '2023': 825,
  '2024': 1000,
  '2025': 735,
}

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador_ajustes: función para realizar el plot sqrt(Yss**2 + Zss**2) contra Xss (normalizado por R_m) con sus objetos y ajustes Vignes.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador_ajustes(
    directorio: str,                                                                       # Carpeta donde están los archivos a graficar.
    objetos: list[str]=['Marte','Vignes','Fruchtman','mín','máx','región','KNN','propios'],# Objetos que se desean graficar.
    años_Fruchtman: list[str] = ['2014','2015','2016','2017','2018','2019'],               # Años de Fruchtman cuyos datos deseo graficar.
    ajuste_Fruchtman: bool    = False,                                                     # Booleano para realizar ajuste Vignes a Fruchtman.
    hemisferio_N:     bool    = False,                                                     # Booleano para elegir mediciones recortadas o no.
    trayectoria: bool   = False,                                                           # Booleano para graficar trayectoria común de MAVEN.
    recorte: str        = 'recorte_Vignes',                                                # Tipo de recorte a usar para la trayectoria MAVEN.
    tiempo_inicial: str = '01/01/2015-00:00:00', tiempo_final: str='30/3/2015-23:59:00',   # Tiempo inicial y final de datos de trayectoria.
    promedio: int       = 1,                                                               # Promedio a utilizar por leer_archivos_MAG.
    modelo: str              = 'salvation_K1',                                             # Modelo del KNN cuyos bow shocks quiero graficar.
    post_procesamiento: bool = False,                                                      # Booleano para modelos post-procesados.
    años_KNN: list[str]      = ['2014'],                                                   # Bow shocks predichos por KNN del año asignado.
    ajuste_KNN: bool         = False,                                                      # Booleano para realizar ajuste Vignes a KNN.
    ajuste_prop: bool = False,                                                             # Booleano para ajustar BS catalogados del 2014.
    guardar: bool = False                                                                  # Booleano para guardar figura en formato .pdf.
) -> None:
  """
  La función graficador_ajustes es una función para realizar un plot 2D, en formato sqrt(Yss**2 + Zss**2)/R_m contra Xss/R_m. Recibe en
  formato string un 'directorio', donde se encuentran los archivos MAG, Fruchtman y bow shocks KNN en sus subcarpetas correspondientes. La
  lista de strings 'objetos' permite graficar el semi-disco 'Marte', el ajuste por hipérbola realizado por 'Vignes' (en 1997), las mediciones
  de 'Fruchtman' de bow shocks detectados, con sus respectivos 'mín' (mínimo BS) y 'máx' (máximo BS), la 'región' total encerrada tras el
  recorte_Vignes realizado, y los bow shocks detectados por el 'KNN' y catalogados manualmente para el año 2014.
  El parámetro 'años_Fruchtman' determina los bow shocks de qué años se graficarán, y si 'ajuste_Fruchtman'=True, sus ajuste no lineales
  por cónicas del tipo Vignes. Si hemiferio_N=True graficará solo a aquellos bow shocks que se produjeron en el hemisferio norte, y si es
  False, todos ellos.
  Si el booleano 'trayectoria'=True, se graficarán las trayectorias realizadas por MAVEN en este plot cilíndrico con el 'recorte' que se
  haya seleccionado, en el intervalo ('tiempo_inicial', 'tiempo_final'), y con el 'promedio' elegido.
  El string 'modelo' representa el tipo de modelo KNN cuyos bow shocks desean graficarse, y el booleano 'post_procesamiento' si desean
  graficarse las mediciones procesadas (bow shocks promediados) o no. La lista de strings 'años_KNN' determina los bow shocks detectados por
  el KNN de cuyo año se desean graficar, y el booleano 'ajuste_KNN'=True, realiza y grafica un ajuste por función de Vignes correspondiente.
  Si el booleano 'ajuste_prop'=True, realiza un ajuste tipo Vignes a los bow shocks detectados manualmente del 2014, y si el booleano
  'guardar'=True, guarda la figura en formato .pdf.
  """
  if 'Marte' in objetos:                                                                   # Si 'Marte' figura en la lista de objetos,
    disco_2D(resolución_r=200, resolución_theta=200)                                       # grafico el semi-disco correspondiente.
  if 'Vignes' in objetos:                                                                  # Si 'Vignes' figura en la lista de objetos,
    x, y, x_a, y_a = hipérbola_Vignes()                                                    # obtengo componentes cartesianas de la hipérbola 
    #p.plot(x,y, label=r'Vignes (1997) ss', color='black')                                 # de Vignes, y grafico el ajuste realizado por él,
    p.plot(x_a,y_a,linestyle='--',label=r'Ajuste Vignes aberrado (MGS, 1997)',color='black')# y el ajuste con coordenadas aberradas (x',y').
  if 'Fruchtman' in objetos:                                                               # Si 'Fruchtman' figura en la lista de objetos,
    for año in años_Fruchtman:                                                             # Para cada año de los años seleccionados,
      data_Fru: pd.DataFrame = leer_archivo_Fruchtman(directorio, año, hemisferio_N)       # leo el archivo Fruchtman del año correspondiente,
      Xss, Yss, Zss = [data_Fru[j] for j in [7,8,9]]                                       # obtengo las componentes (X,Y,Z) en sistema SS,
      if hemisferio_N:                                                                     # Si quiero solo los BS's del norte,
        p.scatter(Xss/R_m, módulo(Yss,Zss, norm=R_m), s=1, alpha=.6,                       # scattereo los datos cilíndricos N normalizados.
                  label=f'Bow shock del año {año} (hemisferio norte)')                     # con una etiqueta que aclare el hemisferio.
      else:                                                                                # Si no,
        p.scatter(Xss/R_m, módulo(Yss,Zss, norm=R_m), s=1, alpha=.6,                       # scattereo los datos cilíndricos normalizados,
                  label=f'Bow shock del año {año}')                                        # con etiqueta normal.
      if ajuste_Fruchtman:                                                                 # Si el booleano ajuste_Fruchtman=True,
        if hemisferio_N:                                                                   # y el hemisferio es el norte, usando el offset
          ajustar_por_función_Vignes(Xss,Yss,Zss, año, offset=offsets_Fruchtman[año][1])   # correspondiente ajusto por Vignes.
        else:                                                                              # Si no, si uso todos los hemisferios,
          ajustar_por_función_Vignes(Xss,Yss,Zss, año, offset=offsets_Fruchtman[año][0])   # ajuste por Vignes normalmente con sus offsets.
    if 'mín' in objetos:                                                                   # Si 'min' está en la lista de objetos,
      p.scatter(mínimo_2019()[0], mínimo_2019()[1],                                        # marco el mínimo en el plot con una cruz grande
                marker='x', s=100, color='purple', label='mínimo 2019')                    # (uso el mismo color que los datos del mínimo).
    if 'máx' in objetos:                                                                   # Si 'máx' está en la lista de objetos,
      p.scatter(máximo_2015()[0], máximo_2015()[1],                                        # lo marco también con una cruz grande
                marker='x', s=100, color='red', label='máximo 2015')                       # (uso el mismo color que los del máximo).
  if 'región' in objetos:                                                                  # Si 'región' está en la lista de objetos, cargo
    x_a_min, y_a_min = hipérbola_mínima(); x_a_max, y_a_max = hipérbola_máxima()           # componentes cartesianas de Hipérbola izq/der.
    p.plot(x_a_min,y_a_min, label=r'Curva mín: ($X_0=0.14, \alpha=20$°)', color='purple')  # Grafico la hiperbola izquierda (mínima)
    p.plot(x_a_max,y_a_max, label=r'Curva máx: ($X_0=1.4, \alpha=-10$°)', color='red')     # y la hipérbola derecha (máxima).
    dibujar_segmento(región=preparar_región_Vignes())                                      # Dibujo el segmento que une los techos de
    sombrear_área(Xmin=x_a_min, Ymin=y_a_min, Xmax=x_a_max, Ymax=y_a_max)                  # ambas, y sombreo todo el área encerrada.
  if trayectoria:                                                                          # Si el booleano trayectoria=True,
    data_MAG: pd.DataFrame = leer_archivos_MAG(os.path.join(directorio, recorte),          # obtengo los datos MAG con el recorte deseado,
                                               tiempo_inicial, tiempo_final, promedio)     # del intervalo (t0,tf) con el promedio indicado.
    Xss, Yss, Zss = [data_MAG[j] for j in [7,8,9]]                                         # Extraigo solo las componentes (X,Y,Z) en SS,
    p.scatter(Xss/R_m, módulo(Yss,Zss,norm=R_m),                                           # y grafico en formato cilíndrico normalizado
              s=1, alpha=.6, label='Trayectoria de MAVEN', rasterized=True)                # usando puntos chicos (s=1) y alta definición.
  if 'KNN' in objetos:                                                                     # Si 'KNN' está en la lista de objetos,
    for año in años_KNN:                                                                   # para cada año cuyos bow shocks deseo graficar,
      data_BS: pd.DataFrame = leer_bow_shocks_KNN(directorio,modelo,post_procesamiento,año)# leo los bow shocks detectados por KNN,
      Xss, Yss, Zss = [data_BS[j] for j in [7,8,9]]                                        # extraigo solo las componentes (X,Y,Z) en SS,
      p.scatter(Xss/R_m, módulo(Yss,Zss,norm=R_m),                                         # y grafico en formato cilíndrico normalizado,
                s=5, alpha=.6, label=f'BS $k$-NN Eclipse Optimizado ({año})')              # elijo tamaño de scatter, transparencia y label.
      if ajuste_KNN:                                                                       # Si el booleano ajuste_Fruchtman=True,
        ajustar_por_función_Vignes(Xss,Yss,Zss, año, offset=offsets_KNN[año])              # realizo ajuste no lineal por hipérbola Vignes.
  if 'propios' in objetos:                                                                 # Si quiero graficar los BS's catalogados del 2014,
      ruta_prop: str = os.path.join(directorio,'propios','catálogo_Fruchtman-propios_2014.sts')# obtengo la ruta en donde se encuentran,
      data_prop: pd.DataFrame = pd.DataFrame(np.loadtxt(ruta_prop))                        # leo los bow shocks detectados por KNN,
      Xss, Yss, Zss = [data_prop[j] for j in [7,8,9]]                                      # extraigo solo las componentes (X,Y,Z) en SS,
      p.scatter(Xss/R_m, módulo(Yss,Zss,norm=R_m), s=15,alpha=.6,label='BS totales (2014)')# y grafico en formato cilíndrico normalizado.
      if ajuste_prop:                                                                      # Si el booleano ajuste_prop=True,
        ajustar_por_función_Vignes(Xss,Yss,Zss, año='2014')                                # realizo ajuste tipo Vignes.
  p.title('Ajuste tipo Vignes para los BS de Fruchtman (MAVEN)', fontsize=8)               # Título del gráfico.
  p.xlabel(r"$x'_{\text{ss}}$ [$R_M$]", fontsize=9)                                        # Coloco labels tipo SS en el eje x.
  p.xticks(fontsize=7); p.yticks(fontsize=7)                                               # Modifico el tamaño de los puntos en x e y.
  p.ylabel(r"$\sqrt{y'_{\text{ss}}^2+z'_{\text{ss}}^2}$ [$R_M$]", fontsize=9)              # y coloco labels tipo SS en el eje y.
  p.grid(which='major', alpha=.2,  linestyle='-', linewidth=.5)                            # Configuro los ejes principales y los secundarios
  p.grid(which='minor', alpha=.15, linestyle=':', linewidth=.5)                            # con transparencia y ancho personalizados.
  p.legend(fontsize=5)                                                                     # Escribo los labels.
  if guardar:                                                                              # Si el 'guardar' es True, guardar_figura()
    guardar_figura()                                                                       # pide un mensaje y guarda tras apretar enter.
  p.show()                                                                                 # Enseño el plot.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def ajustar_por_función_Vignes(
    Xss: np.ndarray, Yss: np.ndarray, Zss: np.ndarray,
    año: str,
    offset: Union[int, None] = None
) -> None:
  """
  La función ajustar_por_función_Vignes recibe 3 arrays 'Xss', 'Yss' y 'Zss' que corresponden a los vectores que contienen las coordenadas
  exactas de bow shocks, ya sean los detectados por Fruchtman o los detectados por mi KNN en el espacio circundante a Marte, y el string
  'año' representa el año a cuyos bow shocks corresponden. Dados todos esos BS's, realiza un ajuste no lineal por una hipérbola del tipo
  Vignes con coordenadas aberradas llamando a función_hipérbola_Vignes (que usa aberración de 4° como Vignes) y lo grafica en el plot
  cilíndrico sqrt(Yss**2 + Zss**2) contra Xss, normalizado por R_m. El parámetro 'offset' puede ser None o un entero, y cuando es no nulo,
  determina hasta qué punto se graficará la hipérbola ajustada, evitando graficar una recta constante en y=0 al final. No devuelve nada.
  """
  popt, pcov = curve_fit(                                                        # popt: parámetros ajustados. pcov: matriz de covarianza.
    lambda x,x_0: función_hipérbola_Vignes(x, x_0=x_0, cant_puntos=1000),        # Función ajuste: tipo hipérbola-Vignes (solo ajusto x_0).
    Xss/R_m,                                                                     # Componente Xss normalizada por R_m.
    módulo(Yss,Zss,norm=R_m),                                                    # Componente Y=sqrt(Yss**2 + Zss**2) normalizada por R_m.
    p0=[0.64]                                                                    # Param inicial: x_0=0.64 (x_0 hallado por Vignes en 1997).
  )                                                                              # Guardo el ajuste en las variables popt, pcov.
  perr = np.sqrt(np.diag(pcov))                                                  # Obtengo errores como las raíces de la diagonal de pcov.
  x: np.ndarray = np.linspace(np.min(Xss/R_m), np.max(Xss/R_m), 1000)            # Creo grilla en el eje x para dibujar la curva ajustada.
  y: np.ndarray = función_hipérbola_Vignes(x, x_0=popt[0], cant_puntos=1000)     # Obtengo los valores en Y para los puntos de la grilla.
  if offset is not None:                                                         # Si offset no es None,
    p.plot(x[:offset], y[:offset], linewidth=2, alpha=.9, #color='darkorange',   # grafico curva ajustada interpolada (reporto x_0),
           label=f'Ajuste tipo Vignes: $x_0$={popt[0]:.3f} $R_M$')               # hasta el valor de offset tanto en x como en y.
  else:                                                                          # Si no,
    p.plot(x,y, linewidth=2, alpha=.9, #color='darkorange',                      # grafico normalmente la curva ajustada interpolada con
         label=f'Ajuste tipo Vignes: $x_0$={popt[0]:.3f} $R_M$')                 # ancho y transparencia personalizadas, y su label.
  print(f'Parámetros de ajuste: {popt}')                                         # Mediante prints, devuelvo parámetros óptimos ajustados 
  print(f'Desviación estándar: {perr}')                                          # y los errores obtenidos (desviación estándar).

#———————————————————————————————————————————————————————————————————————————————————————
def dibujar_segmento(región: dict[str, Any]) -> None:
  """
  La función dibujar_segmento recibe un diccionario de claves string y valores de distinto tipo en 'región', que representa la región
  encerrada que se construye mediante dos hipérbolas del tipo Vignes y un segmento superior que conecta sus extremos y grafica dicho
  segmento techo adecuadamente en el plot cilíndrico sqrt(Yss**2 + Zss**2) contra Xss.
  """
  recta, y_A, y_B = [región[v] for v in ['recta','y_A','y_B']] # Extraigo la función recta y los límites verticales del dicc región.
  y_grilla = np.linspace(y_A, y_B, 300)                        # Creo una grilla uniforme de valores entre y_A e y_B.
  p.plot(recta(y_grilla), y_grilla, color='green')             # Grafico la recta evaluada sobre el eje y.

#———————————————————————————————————————————————————————————————————————————————————————
def sombrear_área(Xmin: np.ndarray, Ymin: np.ndarray, Xmax: np.ndarray, Ymax: np.ndarray) -> None:
  """
  La función sombrear_área recibe 4 arrays (Xmin,Ymin) y (Xmax,Ymax) que representan los puntos inicial y terminal de dos curvas de la forma
  Xmin(Y) y Xmax(Y), respectivamente, y dibuja (grafica) un sombreado entre ellas correctamente con cierta transparencia (alpha).
  """
  j_min, j_max = np.argsort(Ymin), np.argsort(Ymax)                       # Obtengo índices que ordenan Ymin/Ymax para interpolar.
  Y = np.linspace(max(Ymin[j_min].min(),Ymax[j_max].min()),               # Defino una grilla común de valores Y donde existen ambas curvas.
                  min(Ymin[j_min].max(),Ymax[j_max].max()), 1000)         # con suavidad=1000 fija.
  x_max_int = interp1d(Ymax[j_max], Xmax[j_max], kind='linear')           # Interpolador lineal de Xmax en función de Y.
  x_min_int = interp1d(Ymin[j_min], Xmin[j_min], kind='linear')           # Interpolador lineal de Xmin en función de Y.
  p.fill_betweenx(Y, x_min_int(Y), x_max_int(Y), color='green',           # Relleno el área entre ambas curvas x_min_int y x_max_int,
                  alpha=0.2, linewidth=0, label='Región de interés')      # con cierta transparencia (alpha) y sin bordes (linewidth=0).
  x_polígono = np.concatenate(([Xmin[-2]],Xmax[Ymax>Ymin[-2]],[Xmin[-2]]))# Construyo el polígono que cierra el área sombreada en el eje x,
  y_polígono = np.concatenate(([Ymin[-2]],Ymax[Ymax>Ymin[-2]],[Ymin[-2]]))# y en el eje y usando los límites de (Xmin,Ymin); (Xmax,Ymax).
  p.fill(x_polígono, y_polígono, color='green', alpha=0.2, linewidth=0)   # Dibujo el polígono sombreado con transparencia y sin bordes.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————