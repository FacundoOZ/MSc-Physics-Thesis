
# Comentar y modularizar plot gemelo en GRAFICADOR

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar magnitudes físicas medidas por MAG: https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/MAG
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
from datetime import datetime
from tqdm     import tqdm
from typing   import Union
from cycler   import cycler

# Módulos Propios:
from base_de_datos.conversiones import R_m, módulo
from base_de_datos.lectura      import leer_archivos_MAG
from plots.estilo_plots         import guardar_figura, plot_xy, disco_2D, esfera_3D

ruta: str = 'C:/Users/facuo/Documents/Tesis/MAG/'
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador: función para graficar campo magnético y posiciones y trayectoria 2D y 3D de MAVEN medidos por el instrumento MAG (Magnetometer)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador(
    directorio: str,                                                                # Carpeta de los archivos que se desean plotear
    tiempo_inicial: str, tiempo_final: str,                                         # t_inicial y t_final en formato str 'DD/MM/YYYY-HH:MM:SS'
    promedio: int = 1,                                                              # Promedio de lectura de archivos MAG.
    B: bool    = False, B_x: bool = False, B_y: bool = False, B_z: bool = False,    # Campo magnético B=sqrt(Bx**2+By**2+Bz**2) y componentes
    normalización: bool = False,                                                    # 
    x_pc: bool = False, y_pc: bool = False, z_pc: bool = False,                     # Posición en coordenadas PC
    x_ss: bool = False, y_ss: bool = False, z_ss: bool = False,                     # Posición en coordenadas SS
    R: bool    = False,                                                             # R = sqrt(Xpc**2+Ypc**2+Zpc**2) (contra t)
    cil: bool  = False,                                                             # cil = sqrt(y**2 + z**2) (contra x) (trayectoria)
    trayectoria: bool  = False,                                                     # Gráfico 2D (x,y) ó 3D (x,y,z, junto a Marte)
    tamaño_ejes: float = 2.5,                                                       # Ajusta el tamaño máx. de ejes x,y,z a la vez
    scatter: bool      = False,                                                     # Si scatter=True -> grafico sin interpolar (puntos), con
    tamaño_puntos: int = 2,                                                         # 'tamaño_puntos' el diámetro de los puntos.
    coord: str = 'pc',                                                              # Sistema de coordenadas a graficar ('pc' ó 'ss')
    bow_shocks: Union[list[str], None]    = None,                                   # Tipo de predicción cuyos bow shocks deseo graficar.
    modelo_KNN: Union[str, None]          = None,                                   # Tipo de modelo KNN a utilizar.
    post_procesamiento: Union[bool, None] = None,                                   # Booleano para utilizar post-procesamiento de bow shocks.
    guardar: bool = False                                                           # Booleano para guardar en formato PDF la figura creada.
) -> None:
  """
  La función graficador recibe en formato string tres elementos:
    - Un 'directorio' que representa la ruta donde se encuentran los archivos a graficar y con el parámetro 'promedio' que determina el
    promedio en segundos que se quiere tomar sobre los datos.
    - Un 'tiempo_inicial' en formato 'DD/MM/YYYY-HH:MM:SS'.
    - Un 'tiempo_final'   en formato 'DD/MM/YYYY-HH:MM:SS'.
  y dependiendo de los valores booleanos 'B' (módulo de campo magnético), 'B_x', 'B_y', 'B_z' (las componentes de campo magnético), 'x', 'y',
  'z' (las posiciones de la sonda en sistemas PC y SS), 'R' y 'cil', y el parámetro 'trayectoria', permite realizar los siguientes gráficos:
    - El módulo de campo magnético (|B|) o sus componentes B_x, B_y, y B_z (en [nT]) detectado por el instrumento MAG de MAVEN con respecto
    al tiempo de interés seleccionado.
    - La posición de la sonda x, y, y/ó z con respecto al intervalo de tiempo de interés seleccionado.
    - (Si trayectoria y algunas componentes de posición ó campo son = True): Las curvas de trayectoria de la sonda en 2D (y con respecto a x;
    z con respecto a x; ó z con respecto a y) ó en 3D (x,y,z) en el intervalo de tiempo de interés seleccionado.
    - Si normalización=True, realiza un plot gemelo con tiempo UTC en eje x, y con un eje y izquierdo para B o sus componentes, y un eje y
    derecho para la distancia al planeta o las componentes de la posición.
  Los parámetros 'tamaño_ejes', 'scatter' (booleano) y 'tamaño_puntos' permiten ajustar el tamaño de los ejes (cúbicamente) del plot 3D,
  graficar sin interpolación (los puntos) tanto las posiciones como los campos, y ajustar el tamaño de dichos puntos de scatter (cuando 
  scatter=True), respectivamente.
  El parámetro coord = 'pc' ó 'ss' determina si se graficará el campo magnético o sus componentes, o las posiciones en los sistemas de
  coordenadas PlanetoCéntricas (PC) (incluye la rotación de Marte sobre su eje, z apunta al polo norte) ó Sun-State (SS) centradas en el Sol
  (no incluye la rotación de Marte sobre su eje).
  Los parámetros 'bow_shocks', 'modelo_KNN' y 'post_procesamiento' son OPCIONALES, y se utilizan cuando se desean graficar líneas verticales
  para los bow shocks detectados por Fruchtman o el modelo_KNN elegido (con o sin post-procesado), generalmente, junto a las mediciones del
  módulod de B.
  Si 'guardar'=True, pide al usuario para presionar ENTER y así guardar la figura en formato PDF de alta calidad.
  """
  data: pd.DataFrame = leer_archivos_MAG(directorio, tiempo_inicial, tiempo_final,  # Leo archivos MAG que correspondan al intervalo (t0,tf)
                                         promedio)                                  # con el promedio deseado.
  t,Bx,By,Bz,Xpc,Ypc,Zpc,Xss,Yss,Zss = [data[j].to_numpy() for j in range(0,10)]    # Extraigo la información del .sts en ese intervalo
  if trayectoria:                                                                   # Si trayectoria = True, entonces:
    if coord=='pc':                                                                 # 1) si quiero coordenadas PC,
      graficar_trayectoria(Xpc,Ypc,Zpc, x_pc,y_pc,z_pc, cil,                        # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)              # colocando los parámetros que correspondan
    elif coord=='ss':                                                               # 2) si quiero coordenadas SS,
      graficar_trayectoria(Xss,Yss,Zss, x_ss,y_ss,z_ss, cil,                        # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)              # colocando los parámetros correspondientes.
  else:                                                                             # Si no,
    hay_B_j: bool = B or B_x or B_y or B_z                                          # 
    hay_X_j: bool = R or x_pc or y_pc or z_pc or x_ss or y_ss or z_ss               # 
    if normalización and hay_B_j and hay_X_j:                                       # NORMALIZACIÓN ES EL PLOT DOBLE NUEVOOOOOOOOOOO
      ax1 = p.gca(); ax2 = ax1.twinx()                                              #
      ax2.set_prop_cycle(cycler(color=['aqua','gold','magenta','lime','darkviolet','coral','black']))
      if B:                                                                         # --- B (EJE IZQUIERDO)
        plot_xy(t,módulo(Bx,By,Bz), r'$|\mathbf{B}|$', scatter,tamaño_puntos,ax=ax1)#
      for Bj,datos,label in zip([B_x,B_y,B_z],[Bx,By,Bz],['$B_x$','$B_y$','$B_z$']):#
        if Bj:                                                                      #
          plot_xy(t, datos, label, scatter, tamaño_puntos, ax=ax1)                  #
      ax1.set_ylabel('Campo Magnético [nT]')                                        #
      r_módulo = módulo(Xpc,Ypc,Zpc)                                                # --- Posición (EJE DERECHO)
      for Xj, datos, label in zip(                                                  #
        [R, x_pc,y_pc,z_pc, x_ss,y_ss,z_ss], [r_módulo, Xpc,Ypc,Zpc, Xss,Yss,Zss],  #
        [r'$|\mathbf{r}|$',r'$x_{\text{pc}}$',r'$y_{\text{pc}}$',r'$z_{\text{pc}}$',#
                           r'$x_{\text{ss}}$',r'$y_{\text{ss}}$',r'$z_{\text{ss}}$']):
        if Xj:                                                                      #
          plot_xy(t, datos/R_m, label, scatter, tamaño_puntos, ax=ax2)              #
      ax2.set_ylabel('Posición de MAVEN [$R_M$]', color='aqua')                     #
      lines1, labels1 = ax1.get_legend_handles_labels();                            # --- Combined legend
      lines2, labels2 = ax2.get_legend_handles_labels()                             #
      ax1.legend(lines1 + lines2, labels1 + labels2)                                #
      ax1.grid(True, which='minor', linestyle=':', linewidth=0.5)                   #
      ax2.tick_params(axis='y', colors='aqua')                                      #
      ax2.spines['right'].set_color('aqua')                                         #
      ax2.yaxis.grid(True, color='aqua', linestyle='--', alpha=0.5)                 #
      formatear_ejes_y_titulo(tiempo_inicial, tiempo_final, ax=ax1)                 #
    else:                                                                           #
      if B:                                                                         # Si B = True,
        plot_xy(t,módulo(Bx,By,Bz),r'$\left|\mathbf{B}\right|$',scatter,tamaño_puntos)# Grafico el módulo de B usando el graficador 2D: plot_xy.
        p.ylabel('Campo Magnético [nT]')                                            # y nombro al eje y para el campo B (en nanoTesla => [nT])
      graficar_componentes(                                                         # Si alguna componente de B_i = True, la grafico,
        t, [Bx,By,Bz], [B_x,B_y,B_z], ['$B_x$','$B_y$','$B_z$'],                    # con su correspondiente etiqueta,
        'Campo Magnético [nT]', scatter, tamaño_puntos)                             # y nombre del eje y, scatter y tamaño de puntos.
      r_módulo = módulo(Xpc,Ypc,Zpc)                                                # Defino la distancia de MAVEN a Marte (da igual SS=PC) 
      graficar_componentes(                                                         # Ídem, grafico las posiciones con respecto al tiempo
        t, [r_módulo, Xpc,Ypc,Zpc, Xss,Yss,Zss],[R, x_pc,y_pc,z_pc, x_ss,y_ss,z_ss],# tanto para coordenadas PC como SS,
        [r'$|\mathbf{r}|$',r'$x_{\text{pc}}$',r'$y_{\text{pc}}$',r'$z_{\text{pc}}$',# colocando las etiquetas correspondientes: PC,
                          r'$x_{\text{ss}}$',r'$y_{\text{ss}}$',r'$z_{\text{ss}}$'],# y SS.
        'Posición de MAVEN [$R_M$]', scatter, tamaño_puntos, escala=R_m)            # Normalizo por el radio marciano (ESCALA X10).
      if bow_shocks is not None:                                                    # Si deseo graficar bow shocks,
        if 'Fruchtman' in bow_shocks:                                               # Cuando 'bow_shocks' contiene 'Fruchtman',
          graficar_bow_shocks(tiempo_inicial, tiempo_final, origen='Fruchtman',     # los busco y grafico en el intervalo (t_inicial,t_final).
                              color='red',  etiqueta='BS Fruchtman')                # con color rojo y etiquetados.
        if 'KNN' in bow_shocks:                                                     # Si no, si 'KNN' pertenece a 'bow_shocks', uso los del KNN,
          graficar_bow_shocks(tiempo_inicial, tiempo_final, origen='KNN',           # los busco y grafico en el intervalo (t_inicial,t_final),
                              color='green',etiqueta='BS $k$-NN',modelo_KNN=modelo_KNN,# con color verde, etiquetados, y con el modelo elegido,
                              post_procesamiento=post_procesamiento)                # y utilizando post-procesamiento, si lo hubiera.
      formatear_ejes_y_titulo(tiempo_inicial, tiempo_final)                         # Adapto el eje temporal x con el formato que corresponda,
  if not normalización:                                                             # Si no hago un plot gemelo (que ya tiene grilla),
    p.grid(True, which='minor', linestyle=':', linewidth=0.5)                       # Pongo doble grilla, fina y con formato ':',
    p.legend()                                                                      # y escribo los labels.
  if guardar:                                                                       # Si el booleano 'guardar' es True, guardar_figura()
    guardar_figura()                                                                # pide un mensaje y la guarda tras apretar enter.
  p.show()                                                                          # Enseño el plot.
  p.close()                                                                         # Cierro al terminar el proceso (sino se cuelga el input).

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def graficar_trayectoria(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,                                  # Posiciones de la sonda en formato np.ndarray en x, y, z
    x: bool, y: bool, z: bool,                                                    # Valor booleano que determina las componentes a graficar
    cil: bool,                                                                    # Valor booleano de la componente cilíndrica sqrt(y**2+z**2)
    tamaño_ejes: float,                                                           # Permite ajustar el tamaño de los ejes del plot 3D.
    scatter: bool = False,                                                        # Si scatter=True se grafica sin interpolar (puntos), donde
    tamaño_puntos: int = 2,                                                       # tamaño_puntos representa el diámetro de los puntos.
    coord: str = 'pc'                                                             # Sistema de coordenadas 'pc' ó 'ss'.
) -> None:
  """
  Grafica la trayectoria de la sonda MAVEN, ya sea en coordenadas polares 2D (x,y); (x,z); (y,z), ó en coordenadas cartesianas 3D (x,y,z)
  según corresponda. Cuando una ó dos variables x, y, ó z son True, grafica el contenido de los arrays correspondientes X, Y ó Z en
  coordenadas polares 2D. Si todas son True, realiza un plot 3D con el contenido de los arrays X, Y y Z.
  """
  if cil:                                                                         # Si cil=True, entonces
    disco_2D(resolución_r=200, resolución_theta=200)                              # Grafico la circunferencia rellena (Marte) con shade.
    proy_yz = módulo(Y,Z)                                                         # Grafico la proyección del plano y,z.
    plot_xy(X/R_m, proy_yz/R_m, 'Posición de MAVEN', scatter, tamaño_puntos)      # Hago plot 2D sqrt(y^2+z^2) vs x normalizado por R_m.
    if coord=='pc':                                                               # Si las coordenadas son PC,
      p.xlabel(r'$x_{\text{pc}}$ [$R_M$]')                                        # entonces coloco labels tipo PC en x
      p.ylabel(r'$\sqrt{y_{\text{pc}}^2+z_{\text{pc}}^2}$ [$R_M$]')               # y en y.
    elif coord=='ss':                                                             # Sino, si son SS,
      p.xlabel(r'$x_{\text{ss}}$ [$R_M$]')                                        # coloco labels tipo SS en x
      p.ylabel(r'$\sqrt{y_{\text{ss}}^2+z_{\text{ss}}^2}$ [$R_M$]')               # y en y.
  elif x and y and z:                                                             # Si x=y=z=True, entonces se realiza un plot 3D.
    ax    = p.figure().add_subplot(111, projection='3d')                          # Genero un plot 3D.
    u,v,w = esfera_3D(resolución=100)                                             # Grafico Marte como una esfera perfecta de referencia,
    ax.plot_surface(u,v,w, color='red', alpha=0.5)                                # de color rojo, y con cierta transparencia (alpha).
    ax.plot3D(X/R_m, Y/R_m, Z/R_m, label='Posición de MAVEN')                     # Grafico la posición de MAVEN, y normalizo por R_m.
    ax.set_xlim([-tamaño_ejes, tamaño_ejes])                                      # Ajusto el tamaño de los ejes del plot 3D en x,
    ax.set_ylim([-tamaño_ejes, tamaño_ejes])                                      # en y,
    ax.set_zlim([-tamaño_ejes, tamaño_ejes])                                      # y en z en igual proporción.
    ax.set_box_aspect([1,1,1])                                                    # Aspecto cúbico para el plot.
    if coord=='pc':                                                               # Etiquetas de los ejes, tipo PC
      ax.set(xlabel=r'$x_{\text{pc}}$ [$R_M$]',                                   # normalizadas por R_M (Radio Marciano)
             ylabel=r'$y_{\text{pc}}$ [$R_M$]',zlabel=r'$z_{\text{pc}}$ [$R_M$]') # en x,y,z.
    elif coord=='ss':                                                             # Etiquetas de los ejes, tipo SS
      ax.set(xlabel=r'$x_{\text{ss}}$ [$R_M$]',                                   # normalizadas por R_M,
             ylabel=r'$y_{\text{ss}}$ [$R_M$]',zlabel=r'$z_{\text{ss}}$ [$R_M$]') # en x,y,z.
  else:                                                                           # Si los 3 booleanos no son True,
    pares = [(x,y,X,Y,'$x$','$y$'), (x,z,X,Z,'$x$','$z$'), (y,z,Y,Z,'$y$','$z$')] # Creo los pares ordenados (x,y); (x,z); (y,z) con labels
    for cond1, cond2, e1, e2, lx, ly in pares:                                    # Para cada combinación posible,
      if cond1 and cond2:                                                         # Si dos booleanos x_i, x_j son True,
        plot_xy(e1/R_m, e2/R_m, 'Posición de MAVEN', scatter, tamaño_puntos)      # realizo su plot correspondiente.
        p.xlabel(lx+' [$R_M$]'); p.ylabel(ly+' [$R_M$]')                          # Coloco las etiquetas correspondientes del eje x e y
        break                                                                     # Si se cumplió x_i=x_j=True => ya ploteé => rompo el for.

#———————————————————————————————————————————————————————————————————————————————————————
def graficar_componentes(
    t: np.ndarray,                                                     # Array de tiempos para MAVEN en cada punto.
    componentes: list[np.ndarray],                                     # Lista de componentes a graficar en formatos np.ndarray.
    activos: list[bool],                                               # Lista de booleanos que representa qué componentes se graficarán.
    etiquetas: list[str],                                              # Lista de etiquetas de las componentes correspondientes.
    ylabel: str,                                                       # Etiqueta del eje y.
    scatter: bool = False,                                             # Si scatter=True se grafica sin interpolar (puntos), donde
    tamaño_puntos: int = 2,                                            # tamaño_puntos representa el diámetro de los puntos.
    escala: float = 1.0                                                # Escala permite normalizar las componentes correspondientes.
) -> None:
  """
  Grafica las componentes de MAVEN con respecto al tiempo, ya sean las componentes del campo magnético B_x, B_y, B_z ó las componentes de la
  posición de MAVEN x, y, z, (ó |r| = sqrt(x**2 + y**2 + z**2)) del instrumento MAG según corresponda.
  """
  for activo, datos, etiqueta in zip(activos, componentes, etiquetas): # Para cada componente de la lista de activos,
    if activo:                                                         # Si el activo correspondiente = True, entonces se graficará contra t.
      plot_xy(t, datos/escala, etiqueta, scatter, tamaño_puntos)       # Grafico la componente correspondiente normalizada por 'escala'
      p.ylabel(ylabel)                                                 # Coloco la etiqueta que corresponda en el eje y.

#———————————————————————————————————————————————————————————————————————————————————————
def graficar_bow_shocks(
    tiempo_inicial: str, tiempo_final: str,                                      # Tiempos inicial y final en formato str 'DD/MM/YYYY-HH:MM:SS'.
    origen: str,                                                                 # Origen de los bow shocks a graficar ('Fruchtman' ó 'KNN').
    color: str,                                                                  # Parámetro para colorear los BS 'Fruchtman' ó 'KNN'.
    etiqueta: str,                                                               # Etiqueta a asignar a los bow shocks graficados.
    modelo_KNN: Union[str, None] = None,                                         # Tipo de modelo KNN a utilizar.
    post_procesamiento: Union[bool, None] = None,                                # Booleano para utilizar post-procesamiento de bow shocks.
) -> None:
  """
  Grafica líneas verticales que representan los tiempos cuando ocurrieron los bow shocks en el intervalo ('tiempo_inicial','tiempo_final')
  detectados. El string 'origen' debe ser igual a 'Fruchtman' ó 'KNN', y la función va a buscar los dataframes que contienen los tiempos de
  los bow shocks en día decimal, en las subcarpetas correspondientes. Los parámetros 'color' y 'etiqueta' permiten asignar 1 único color y
  etiqueta a todos los bow shocks del KNN ó a todos los de Fruchtman, y los parámetros 'modelo_KNN' y 'post_procesamiento' son OPCIONALES, y
  se utilizarán cuando se grafiquen solamente los del modelo KNN. La función no devuelve nada.
  """
  t0: pd.Timestamp = pd.to_datetime(tiempo_inicial, format='%d/%m/%Y-%H:%M:%S')  # Convierto a objeto datetime los tiempos strings inicial
  tf: pd.Timestamp = pd.to_datetime(tiempo_final,   format='%d/%m/%Y-%H:%M:%S')  # y final, cuyo formato es 'DD/MM/YYYY-HH:MM:SS'.
  año: int = t0.year                                                             # Obtengo el año de tiempo inicial (no graficaré 2 años).
  t_ref: pd.Timestamp = pd.Timestamp(f'{año}-01-01 00:00:00')                    # Obtengo tiempo cero como referencia (1 de enero del año).
  if origen=='Fruchtman':                                                        # Si el origen es Fruchtman,
    archivo_Fru: str = f'fruchtman_{año}_merge_hemisferio_N.sts'                 # reconstruyo el nombre del archivo con el año indicado,
    ruta_Fru: str    = os.path.join(ruta,'fruchtman','hemisferio_N', archivo_Fru)# obtengo la ruta_completa + nombre_archivo,
    día_decimal: np.ndarray = np.loadtxt(ruta_Fru, usecols=0)                    # y obtengo los días decimales (solo la columna 0).
  elif origen=='KNN':                                                            # Si no, si el origen es mi KNN,
    ruta_base: str     = os.path.join(ruta,'KNN','predicción', modelo_KNN)       # construyo la ruta base donde se encuentran los archivos.
    if post_procesamiento:                                                       # Si quiero tomar post-procesamiento,
      archivo_KNN: str = f'tiempos_BS_{año}_promedio.txt'                        # reconstruyo el nombre del año correspondiente,
      ruta_KNN: str    = os.path.join(ruta_base,'post_procesamiento',archivo_KNN)# obtengo la ruta_completa correspondiente + nombre_archivo,
    else:                                                                        # Si no,
      ruta_KNN: str    = os.path.join(ruta_base, f'tiempos_BS_{año}.txt')        # obtengo la ruta_completa + nombre_archivo original.
    día_decimal: np.ndarray = np.loadtxt(ruta_KNN, skiprows=1)                   # y obtengo los días decimales (omito título='día_decimal').
  t: pd.DatetimeIndex = t_ref + pd.to_timedelta(día_decimal-1, unit='D')         # Obtengo los tiempos en formato datetime,
  t_máscara           = t[(t >= t0) & (t <= tf)]                                 # y me quedo solo con aquellos pertenecientes al intervalo.
  ax = p.gca()                                                                   # Get Current Axes (obtener ejes actuales) en la var ax.
  ax.axvline(t_máscara[0], alpha=1, color=color, label=etiqueta)               # Grafico por separado el primer BS para agregarle etiqueta.
  for t_BS in t_máscara[1:]:                                                     # Para el resto de tiempos (día_decimal) de BS de la lista,
    ax.axvline(t_BS, alpha=1, color=color)                                     # grafico líneas verticales transparentes (alpha) sin label.

#———————————————————————————————————————————————————————————————————————————————————————
def formatear_ejes_y_titulo(
    tiempo_inicial: str, tiempo_final: str,                                        # Tiempos en formato string 'DD/MM/YYYY-HH:MM:SS'.
    ax = None                                                                      # Parámetro opcional para los ejes.
) -> None:
  """
  Ajusta el formato del eje temporal dependiendo del intervalo de tiempo graficado y crea el título correspondiente. Si se grafican
  mediciones correspondientes a un mismo día y mes, se utilizará el formato HH:MM:SS; y si no, se utilizará el formato DD/MM. El parámetro
  'ax' es OPCIONAL, cuando se utiliza permite graficar plots gemelos.
  """
  t0: pd.Timestamp = pd.to_datetime(tiempo_inicial, format='%d/%m/%Y-%H:%M:%S')    # Convierto a objeto datetime los tiempos strings inicial
  tf: pd.Timestamp = pd.to_datetime(tiempo_final,   format='%d/%m/%Y-%H:%M:%S')    # y final, cuyo formato es 'DD/MM/YYYY-HH:MM:SS'.
  if ax is None:
    ax = p.gca()                                                                   # Get Current Axes (obtener ejes actuales) en la var ax.
  if t0.date() == tf.date():                                                       # Si el día y el mes es el mismo,
    ax.set_title(f"Mediciones del día {t0.strftime('%d/%m/%Y')} (resolución 1 Hz)")# uso el título 'Mediciones del día ...'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))                 # y el eje en formato 'HH:MM:SS'.
    ax.set_xlabel('Tiempo UTC (HH:MM:SS)')                                         # Coloco el label.
  else:                                                                            # Si los días son distintos, uso formato 'DD/MM/YYYY', y 
    ax.set_title(f"Mediciones del {t0.strftime('%d/%m/%Y')} al {tf.strftime('%d/%m/%Y')} (1 Hz)")# modifico el título,
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))                 # y el eje en formato 'DD/MM'.
    ax.set_xlabel('Fecha UTC (DD/MM/YYYY)')                                        # Título del eje x.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————