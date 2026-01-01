
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar las magnitudes físicas medidas por MAVEN MAG en 2D, 3D y más.
#============================================================================================================================================

import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
import matplotlib.colors as colors
import matplotlib.dates  as mdates # Permite realizar gráficos en formatos de fecha 'DD/MM/YYYY', 'HH:MM:SS', etc.
import cdflib                      # para poder leer archivos .cdf, Common Data Frame (NASA)

from numpy                  import sqrt,pi,cos,sin,shape
from mpl_toolkits.mplot3d   import Axes3D
from datetime               import datetime, timedelta
from tqdm                   import tqdm
from cdflib                 import cdfepoch
#from bs4                   import BeautifulSoup
#from scipy.interpolate     import interp1d

from base_de_datos.descarga import dia_del_año

R_m: float = 3396.3 # Radio marciano máximo (km)

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Instrumento MAG (Magnetometer) # (https://pds-ppi.igpp.ucla.edu/mission/MAVEN/maven/MAG)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def graficador(
    directorio: str,                                                               # Carpeta de los archivos que se desean plotear
    tiempo_inicial: str, tiempo_final: str,                                        # t_inicial y t_final en formato str 'DD/MM/YYYY-HH:MM:SS'
    B: bool = False, B_x: bool = False, B_y: bool = False, B_z: bool = False,      # Campo magnético
    x_pc: bool = False, y_pc: bool = False, z_pc: bool = False,                    # Posición en coordenadas PC
    x_ss: bool = False, y_ss: bool = False, z_ss: bool = False,                    # Posición en coordenadas SS
    R: bool = False,                                                               # R = sqrt(y**2 + z**2)
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
  posiciones de la sonda en varios sistemas de coordenadas) y el parámetro 'trayectoria', permite realizar los siguientes gráficos:
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
      graficar_trayectoria(Xpc,Ypc,Zpc, x_pc,y_pc,z_pc, R,                         # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)             # colocando los parámetros que correspondan
    elif coord=='ss':                                                              # 2) si quiero coordenadas SS,
      graficar_trayectoria(Xss,Yss,Zss, x_ss,y_ss,z_ss, R,                         # grafico la trayectoria x,y,z que corresponda 2D ó 3D,
                           tamaño_ejes, scatter, tamaño_puntos, coord)             # colocando los parámetros correspondientes.
  else:                                                                            # Si no,
    if B:                                                                          # Si B = True,
      B_modulo = sqrt(Bx**2 + By**2 + Bz**2)                                       # grafico el módulo de B.
      plot_xy(t, B_modulo, r'$\left|\mathbf{B}\right|$', scatter, tamaño_puntos)   # Uso el graficador 2D: plot_xy.
      p.ylabel('Campo Magnético [nT]')                                             # y nombro al eje y para el campo B (en nanoTesla => [nT])
    graficar_componentes(                                                          # Si alguna componente de B_i = True, la grafico,
      t, [Bx,By,Bz], [B_x,B_y,B_z], ['$B_x$','$B_y$','$B_z$'],                     # con su correspondiente etiqueta,
      'Campo Magnético [nT]', scatter, tamaño_puntos)                              # y nombre del eje y, scatter y tamaño de puntos.
    graficar_componentes(                                                          # Ídem, grafico las posiciones con respecto al tiempo
      t, [Xpc,Ypc,Zpc, Xss,Yss,Zss], [x_pc,y_pc,z_pc, x_ss,y_ss,z_ss],             # tanto para coordenadas PC como SS,
      [r'$x_{\text{pc}}$',r'$y_{\text{pc}}$',r'$z_{\text{pc}}$',                   # colocando las etiquetas correspondientes: PC,
       r'$x_{\text{ss}}$',r'$y_{\text{ss}}$',r'$z_{\text{ss}}$'],                  # y SS.
      'Posición de MAVEN [$R_M$]', scatter, tamaño_puntos, escala=R_m/10)          # Normalizo por el radio marciano.
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
def leer_archivos_MAG(
    directorio: str,                                                             # Carpeta de la base de datos de los archivos a leer
    tiempo_inicial: str, tiempo_final: str                                       # tiempo inicial y final en formato str 'DD/MM/YYYY-HH:MM:SS'
) -> pd.DataFrame:
  """
  Lee y concatena ordenadamente todos los archivos .sts del año y en el directorio pasados por parámetro (tanto los de terminación 'r01' como
  'r02') que se encuentren entre el intervalo [t0, tf] (inclusive) determinado por las variables tiempo_inicial y final, en formato string
  'DD/MM/YYYY-HH:MM:SS'.
  """
  t0 = pd.to_datetime(tiempo_inicial, format='%d/%m/%Y-%H:%M:%S')                # Convierto strings tiempo_inicial/final a objetos datetime
  tf = pd.to_datetime(tiempo_final,   format='%d/%m/%Y-%H:%M:%S')                # (le indico a pandas como extraer DD/MM/AA, HH:MM:SS).
  if tf < t0:                                                                    # Si el tiempo_incial es posterior al inicial,
    raise ValueError('El str tiempo_final debe ser posterior a tiempo_inicial')  # arrojo un error de valores.
  cant_dias: int = (tf - t0).days + 1                                            # Cantidad de días a recorrer (inclusive).
  lista_sts: list[pd.DataFrame] = []                                             # Listas donde se acumularán los dataframes leídos.
  with tqdm(total=cant_dias, desc='Leyendo archivos MAG', unit='día') as pbar:   # Con tqdm, leo la cantidad de días con 1 día por unidad.
    for año in range(t0.year, tf.year+1):                                        # Para el iterador año entre año_0 y año_f (+1 por el range)
      inicio = max(t0, datetime(año, 1, 1))                                      # Calculo la fecha exacta de inicio usando max,
      fin    = min(tf, datetime(año, 12, 31, 23, 59, 59))                        # y min, mediante tiempos datetime
      j = inicio                                                                 # Creo otro iterador j que representará la fecha.
      while j <= fin:                                                            # El j irá desde fecha_inicial a fecha_final.
        dia, mes = j.strftime('%d'), j.strftime('%m')                            # Extraigo día y mes del iterador (strings) con 2 dígitos.
        DOY: str = dia_del_año(dia, mes, str(año))                               # Calculo el Day Of Year (DOY) de la fecha actual.
        ruta_base: str = os.path.join(directorio, str(año), str(int(mes)))       # Ruta base donde deberían estar los archivos de ese día.
        nombres: list[str] = [                                                   # Creo una lista de dos strings que contiene los posibles
          f'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r01_recortado.sts', # nombres que puede tener el archivo correspondiente a ese
          f'mvn_mag_l2_{año}{DOY}merge1s_{año}{mes}{dia}_v01_r02_recortado.sts'] # dia. Con terminación 'r01' ó 'r02' (r=revisión).
        if 'hemisferio_N' in directorio:                                         # Si se desea graficar el hemisferio norte,
          nombres = [x.replace('.sts', '_hemisferio_N.sts') for x in nombres]    # reemplazo el nombre por la terminación correpondiente.
        elif 'hemisferio_ND' in directorio:                                      # Y si se desea graficar solo el hemisferio norte diurno,
          nombres = [x.replace('.sts', '_hemisferio_ND.sts') for x in nombres]   # también.
        encontrado: bool = False                                                 # Creo una variable que representa el estado del archivo
        for archivo in nombres:                                                  # Para cada archivo de la lista 'nombres',
          ruta_archivo: str = os.path.join(ruta_base, archivo)                   # guardo su ubicación en la variable ruta_archivo.
          if os.path.exists(ruta_archivo):                                       # Si el archivo existe,
            df = pd.read_csv(ruta_archivo, sep=' ', header=None)                 # lo leo completamente,
            df[0] = dia_decimal_a_datetime(df[0].to_numpy(), año)                # convierto la col 0 a datetime CON EL AÑO CORRESPONDIENTE,
            lista_sts.append(df)                                                 # EN lista_sts CREO UN EJE t ABSOLUTO con todos los años
            encontrado = True                                                    # y actualizo la variable encontrado (se encontró)
            break                                                                # Ya no itero más el for.
        if not encontrado:                                                       # Si no se encontró ni el 'r01' ni el 'r02', entonces
          print(f'No se encontraron archivos del {dia}/{mes}/{año}')             # aviso la fecha del archivo que no se encontró.
        j += timedelta(days=1)                                                   # Avanzo al iterador al día siguiente,
        pbar.update(1)                                                           # y actualizo la barra de progreso.
  if not lista_sts:                                                              # Si no se encontró absolutamente ningún archivo, 
    raise FileNotFoundError('No se encontraron archivos en el rango dado.')      # devuelvo un mensaje.
  datos: pd.DataFrame = pd.concat(lista_sts, ignore_index=True)                  # Concateno todos los dataframes de lista_sts en uno solo.
  datos = datos[(datos[0] >= t0) & (datos[0] <= tf)]                             # Recorto los datos exactos del intervalo (t_i,t_f) ingresado,
  return datos.reset_index(drop=True)                                            # y devuelvo el dataframe final con índices limpios.

#———————————————————————————————————————————————————————————————————————————————————————
def dia_decimal_a_datetime(
    dia_decimal: np.ndarray,                                               # Lista de días en formato np.ndarray
    año: int                                                               # Año de los días de la lista
) -> pd.DatetimeIndex:
  """
  Recibe una lista de días decimales en formato float (por ejemplo [123.75, 361.98]) y un año (por ejemplo 2019), y devuelve un DatetimeIndex
  que contiene una lista de objetos datetime con los días decimales del año correspondiente convertidos a formato 'AÑO-MES-DÍA HH:MM:SS'.
  """
  base = datetime(año, 1, 1)                                               # Agrego la variable tiempo (de tipo datetime) a res.
  return pd.to_datetime([base + timedelta(days=d-1) for d in dia_decimal]) # Devuelvo res en formato datetime.

#———————————————————————————————————————————————————————————————————————————————————————
def guardar_figura() -> None:
  """
  La función guardar_figura no recibe ningún parámetro. Permite guardar la imagen graficada con la mejor resolución posible (formato PDF).
  """
  respuesta = input('Presione ENTER para guardar .PDF (escriba para omitir): ')  # Con input, pregunto al usuario si desea guardar el plot.
  if respuesta == '':                                                            # Si presiona ENTER,
    p.savefig('plot_MAG.pdf', format='pdf', bbox_inches='tight')                 # se guarda en formato .PDF (mejor calidad)
    print(f'Se ha guardado correctamente.')                                      # y devuelve un mensaje.
  else:                                                                          # Si escribe algo y presiona ENTER,
    print('Figura no guardada.')                                                 # no se guarda.

#———————————————————————————————————————————————————————————————————————————————————————
def plot_xy(
    x: np.ndarray,                                   # Array de puntos para la coordenada x (eje de abscisas).
    y: np.ndarray,                                   # Array de puntos para la coordenada y (eje de ordenadas).
    etiqueta: str = None,                            # Nombre de las mediciones.
    scatter: bool = False,                           # Graficar por puntos y no por interpolación.
    tamaño_puntos: int = 2                           # Tamaño de los puntos.
) -> None:
  """
  La función plot_xy realiza el gráfico 2D de x contra y de dos np.ndarrays pasados por parámetro. Coloca las etiquetas (str)
  correspondientes, y si el parámetro booleano scatter es True, realiza un plot de scatter (por puntos) y permite ajustar el tamaño de
  esos puntos. Si scatter=False, realiza un plot común por interpolación. No devuelve nada. 
  """
  if scatter:                                        # Si scatter=True,
    p.scatter(x, y, s=tamaño_puntos, label=etiqueta) # => realizo gráfico por puntos sin interpolación, con tamaño de puntos y label.
  else:                                              # Si no,
    p.plot(x, y, label=etiqueta)                     # realizo un plot común por interpolación y coloco el label.

#———————————————————————————————————————————————————————————————————————————————————————
def graficar_trayectoria(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,                                  # Posiciones de la sonda en formato np.ndarray en x, y, z
    x: bool, y: bool, z: bool,                                                    # Valor booleano que determina las componentes a graficar
    R: bool,                                                                      # Valor booleano de la componente cilíndrica sqrt(y**2+z**2)
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
  if R:                                                                           # Si R=True, entonces
    plot_xy(X/R_m, sqrt((Y/R_m)**2+(Z/R_m)**2), 'Posición de MAVEN', scatter, tamaño_puntos) # Hago plot 2D sqrt(y^2+z^2) vs x normalizado.
    if coord=='pc':                                                               # Si las coordenadas son PC,
      p.xlabel(r'$x_{\text{pc}}$ [$R_M$]')                                        # entonces coloco labels tipo PC en x
      p.ylabel(r'$\sqrt{y_{\text{pc}}^2+z_{\text{pc}}^2}$ [$R_M$]')               # y en y.
    elif coord=='ss':                                                             # Sino, si son SS,
      p.xlabel(r'$x_{\text{ss}}$ [$R_M$]')                                        # coloco labels tipo SS en x
      p.ylabel(r'$\sqrt{y_{\text{ss}}^2+z_{\text{ss}}^2}$ [$R_M$]')               # y en y.
  elif x and y and z:                                                             # Si x=y=z=True, entonces se realiza un plot 3D.
    fig   = p.figure()                                                            # Creo la figura.
    ax    = fig.add_subplot(111, projection='3d')                                 # Genero un plot 3D.
    u,v,w = esfera_3D(resolución=100)                                             # Grafico Marte como una esfera perfecta de referencia,
    ax.plot_surface(u,v,w, color='red', alpha=0.5)                                # de color rojo, y con cierta transparencia (alpha).
    ax.plot3D(X/R_m, Y/R_m, Z/R_m, label='MAVEN')                                 # Grafico la posición de MAVEN, y normalizo por R_m.
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
    ax.set_title('Posición de MAVEN')                                             # y título del plot.
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
  posición de MAVEN x, y, z, del instrumento MAG según corresponda.
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
def esfera_3D(
    resolución: float = 50                               # Resolución permite ajustar la definición de la esfera (predeterminado en 50)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Genera las coordenadas (x,y,z) de una esfera unitaria mediante coordenadas esféricas.
  """
  phi          = np.linspace(0, 2*pi, resolución)        # Creo un vector phi en el intervalo [0,2pi] con la resolución pasada por parámetro
  theta        = np.linspace(0,   pi, resolución)        # y un vector theta en el intervalo [0,pi].
  superficie_x = np.outer(cos(phi), sin(theta))          # Conversión a coordenadas cartesianas para la posición en x,
  superficie_y = np.outer(sin(phi), sin(theta))          # para la posición en y,
  superficie_z = np.outer(np.ones_like(phi), cos(theta)) # y para la posición en z mediante coordenadas esféricas.
  return (superficie_x, superficie_y, superficie_z)      # Devuelvo una tripla que representa a la esfera 3D.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————