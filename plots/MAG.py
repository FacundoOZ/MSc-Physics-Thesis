
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar magnitudes físicas medidas por MAG: https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/MAG
#============================================================================================================================================

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
from numpy    import sqrt
from datetime import datetime
from tqdm     import tqdm

# Módulos Propios:
from base_de_datos.conversiones import R_m
from base_de_datos.lectura      import leer_archivos_MAG
from plots.estilo_plots         import guardar_figura, plot_xy, disco_2D, esfera_3D

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# graficador: función para graficar campo magnético y posiciones y trayectoria 2D y 3D de MAVEN medidos por el instrumento MAG (Magnetometer)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador(
    directorio: str,                                                               # Carpeta de los archivos que se desean plotear
    tiempo_inicial: str, tiempo_final: str,                                        # t_inicial y t_final en formato str 'DD/MM/YYYY-HH:MM:SS'
    B: bool = False, B_x: bool = False, B_y: bool = False, B_z: bool = False,      # Campo magnético B=sqrt(Bx**2+By**2+Bz**2) y componentes
    x_pc: bool = False, y_pc: bool = False, z_pc: bool = False,                    # Posición en coordenadas PC
    x_ss: bool = False, y_ss: bool = False, z_ss: bool = False,                    # Posición en coordenadas SS
    R: bool = False,                                                               # R = sqrt(Xpc**2+Ypc**2+Zpc**2) (contra t)
    cil: bool = False,                                                             # cil = sqrt(y**2 + z**2) (contra x) (trayectoria)
    trayectoria: bool = False,                                                     # Gráfico 2D (x,y) ó 3D (x,y,z, junto a Marte)
    tamaño_ejes: float = 2.5,                                                      # Ajusta el tamaño máx. de ejes x,y,z a la vez
    scatter: bool = False,                                                         # Si scatter=True -> grafico sin interpolar (puntos), con
    tamaño_puntos: int = 2,                                                        # 'tamaño_puntos' el diámetro de los puntos.
    coord: str = 'pc'                                                              # Sistema de coordenadas a graficar ('pc' ó 'ss')
) -> None:
  """
  La función graficador recibe en formato string tres elementos:
    - Un directorio que representa la ruta donde se encuentran los archivos que se desean graficar.
    - Un tiempo_inicial en formato 'DD/MM/YYYY-HH:MM:SS'.
    - Un tiempo_final   en formato 'DD/MM/YYYY-HH:MM:SS'.
  y dependiendo de los valores booleanos B (módulo de campo magnético), B_x, B_y, B_z (las componentes de campo magnético), x, y, z (las 
  posiciones de la sonda en varios sistemas de coordenadas), cil y R, y el parámetro 'trayectoria', permite realizar los siguientes gráficos:
    - El módulo de campo magnético (|B|) o sus componentes B_x, B_y, y B_z (en [nT]) detectado por el instrumento MAG de MAVEN con respecto
    al tiempo de interés seleccionado.
    - La posición de la sonda x, y, y/ó z con respecto al intervalo de tiempo de interés seleccionado.
    - (Si trayectoria y algunas componentes de posición ó campo son = True): Las curvas de trayectoria de la sonda en 2D (y con respecto a x;
    z con respecto a x; ó z con respecto a y) ó en 3D (x,y,z) en el intervalo de tiempo de interés seleccionado.
  Los parámetros 'tamaño_ejes', 'scatter' (booleano) y 'tamaño_puntos' permiten ajustar el tamaño de los ejes (cúbicamente) del plot 3D,
  graficar sin interpolación (los puntos) tanto las posiciones como los campos, y ajustar el tamaño de dichos puntos de scatter (cuando 
  scatter=True), respectivamente.
  El parámetro coord = 'pc' ó 'ss' determina si se graficará el campo magnético o sus componentes, o las posiciones en los sistemas de
  coordenadas PlanetoCéntricas (PC) (incluye la rotación de Marte sobre su eje, z apunta al polo norte) ó Sun-State (SS) centradas en el Sol
  (no incluye la rotación de Marte sobre su eje).
  """
  data: pd.DataFrame = leer_archivos_MAG(directorio, tiempo_inicial, tiempo_final) # Leo los archivos mag que correspondan al intervalo (t0,tf)
  t,Bx,By,Bz,Xpc,Ypc,Zpc,Xss,Yss,Zss = [data[j].to_numpy() for j in range(0,10)]   # Extraigo la información del .sts en ese intervalo
  if trayectoria:                                                                  # Si trayectoria = True, entonces:
    if coord=='pc':                                                                # 1) si quiero coordenadas PC,
      graficar_trayectoria(Xpc,Ypc,Zpc, x_pc,y_pc,z_pc, cil,                       # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)             # colocando los parámetros que correspondan
    elif coord=='ss':                                                              # 2) si quiero coordenadas SS,
      graficar_trayectoria(Xss,Yss,Zss, x_ss,y_ss,z_ss, cil,                       # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)             # colocando los parámetros correspondientes.
  else:                                                                            # Si no,
    if B:                                                                          # Si B = True,
      B_modulo = sqrt(Bx**2 + By**2 + Bz**2)                                       # grafico el módulo de B.
      plot_xy(t, B_modulo, r'$\left|\mathbf{B}\right|$', scatter, tamaño_puntos)   # Uso el graficador 2D: plot_xy.
      p.ylabel('Campo Magnético [nT]')                                             # y nombro al eje y para el campo B (en nanoTesla => [nT])
    graficar_componentes(                                                          # Si alguna componente de B_i = True, la grafico,
      t, [Bx,By,Bz], [B_x,B_y,B_z], ['$B_x$','$B_y$','$B_z$'],                     # con su correspondiente etiqueta,
      'Campo Magnético [nT]', scatter, tamaño_puntos)                              # y nombre del eje y, scatter y tamaño de puntos.
    r_modulo = sqrt(Xpc**2 + Ypc**2 + Zpc**2)                                      # Defino la distancia de MAVEN a Marte (da igual SS=PC) 
    graficar_componentes(                                                          # Ídem, grafico las posiciones con respecto al tiempo
      t, [Xpc,Ypc,Zpc, Xss,Yss,Zss, r_modulo],[x_pc,y_pc,z_pc, x_ss,y_ss,z_ss, R], # tanto para coordenadas PC como SS,
      [r'$x_{\text{pc}}$',r'$y_{\text{pc}}$',r'$z_{\text{pc}}$',                   # colocando las etiquetas correspondientes: PC,
       r'$x_{\text{ss}}$',r'$y_{\text{ss}}$',r'$z_{\text{ss}}$',r'$|\mathbf{r}|$'],# y SS.
      'Posición de MAVEN [$R_M$]', scatter, tamaño_puntos, escala=R_m/10)          # Normalizo por el radio marciano (ESCALA X10).
    formatear_ejes_y_titulo(                                                       # Adapto el eje temporal x con el formato que corresponda,
      pd.to_datetime(tiempo_inicial, format='%d/%m/%Y-%H:%M:%S'),                  # convirtiendo t_inicial y t_final a objeto datetime
      pd.to_datetime(tiempo_final,   format='%d/%m/%Y-%H:%M:%S'))                  # en formato 'DD/MM/YYYY-HH:MM:SS'.
  p.grid(True, which='minor', linestyle=':', linewidth=0.5)                        # Pongo doble grilla, fina y con formato ':'
  p.legend()                                                                       # Escribo los labels
  #guardar_figura()                                                                # guardo la figura
  p.show()                                                                         # Enseño el plot.
  p.close()                                                                        # Cierro al terminar el proceso (sino se cuelga el input).

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
    proy_yz = sqrt(Y**2 + Z**2)                                                   # Grafico la proyección del plano y,z.
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
        p.xlabel(lx + ' [$R_M$]')                                                 # Coloco las etiquetas correspondientes del eje x
        p.ylabel(ly + ' [$R_M$]')                                                 # y el eje y.
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
def formatear_ejes_y_titulo(
    t0: datetime, tf: datetime                                                      # Tiempos en formato de objeto datetime inicial y final.
) -> None:
  """
  Ajusta el formato del eje temporal dependiendo del intervalo de tiempo graficado y crea el título correspondiente. Si se grafican
  mediciones correspondientes a un mismo día y mes, se utilizará el formato HH:MM:SS; y si no, se utilizará el formato DD/MM.
  """
  ax = p.gca()                                                                      # Get Current Axes (obtener ejes actuales) en la var ax.
  if t0.date() == tf.date():                                                        # Si el día y el mes es el mismo,
    ax.set_title(f"Mediciones del día {t0.strftime('%d/%m/%Y')} (resolución 1 Hz)") # uso el título 'Mediciones del día ...'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))                  # y el eje en formato 'HH:MM:SS'.
    p.xlabel('Tiempo UTC (HH:MM:SS)')                                               # Coloco el label.
  else:                                                                             # Si los días son distintos, uso formato 'DD/MM/YYYY', y 
    ax.set_title(f"Mediciones del {t0.strftime('%d/%m/%Y')} al {tf.strftime('%d/%m/%Y')} (resolución 1 Hz)") # modifico el título,
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))                  # y el eje en formato 'DD/MM'.
    p.xlabel('Fecha UTC (DD/MM/YYYY)')                                              # Título del eje x.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————