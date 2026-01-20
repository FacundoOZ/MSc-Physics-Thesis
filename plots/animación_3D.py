
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar una animación de la trayectoria de la sonda MAVEN en el espacio 3D en coordenadas PC ó SS
#============================================================================================================================================

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p
from matplotlib.figure           import Figure
from matplotlib.lines            import Line2D
from matplotlib.widgets          import Button
from matplotlib.animation        import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Módulos Propios:
from base_de_datos.conversiones import R_m
from base_de_datos.lectura      import leer_archivos_MAG
from plots.estilo_plots         import esfera_3D

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# trayectoria_3D_MAVEN_MAG: función para graficar la trayectoria animada 3D de la sonda MAVEN alrededor de Marte en coordenadas PC ó SS.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def trayectoria_3D_MAVEN_MAG(
    directorio: str,                                                                      # Carpeta donde se encuentran los archivos a animar.
    tiempo_inicial: str, tiempo_final: str,                                               # t_0,t_f a animar (en formato DD/MM/YYYY-HH:MM:SS).
    promedio: int = 1,                                                                    # Promedio de lectura de archivos MAG.
    tamaño_ejes: float = 2.5,                                                             # Ajusta el tamaño máx. de ejes x,y,z a la vez.
    paso: int = 100,                                                                      # Ajusta el paso entre datos -> 'velocidad'.
    delay: int = 1,                                                                       # Delay entre frames (recomendado: >=1 ms).
    coord: str = 'pc'                                                                     # Coordenadas a graficar.
) -> None:
  """
  La función trayectoria_3D_MAVEN_MAG recibe en formato string un directorio, un tiempo inicial y un tiempo final que representan la ruta
  donde se encuentran los archivos MAG que se desean graficar en el intervalo de tiempo indicado, siendo éstos últimos dos en formato string
  'DD/MM/YYYY-HH:MM:SS', y los lee mediante la función 'leer_archivos_MAG' del módulo plots.MAG con el promedio en segundos ingresado.
  Mediante tamaño_ejes (float) puede ajustarse el tamaño máximo del plot 3D cúbico, y los enteros 'paso' y 'delay' permiten ajustar la
  cantidad de puntos que se recorreran por frame y con qué velocidad, respectivamente. Esta función permite graficar tanto coordenadas PC ó
  SS.
  La función realiza un gráfico 3D que muestra en forma animada la curva de trayectoria que traza la posición de MAVEN en el espacio (x,y,z)
  medida por el instrumento MAG, ya sea en coordenadas Planetocéntricas (PC) ó Sun-State (SS).
  """
  data: pd.DataFrame = leer_archivos_MAG(directorio,tiempo_inicial,tiempo_final,promedio) # Leo los archivos en el intervalo (t0,tf) en 'data'.
  t, _,_,_, Xpc,Ypc,Zpc, Xss,Yss,Zss = [data[j].to_numpy() for j in range(0, 10)]         # Extraigo solo el tiempo y las coordenadas PC y SS.
  fig, ax, line, point, x,y,z = crear_plot3D(Xpc,Ypc,Zpc,Xss,Yss,Zss, tamaño_ejes, coord) # Creo el plot 3D mediante la función 'crear_plot3D'.
  def update(i):                                                                          # Función de actualización llamada por FuncAnimation
    return actualizar_frame(i, line, point, ax, t, x, y, z)                               # que devuelve el frame i-ésimo.
  ani = FuncAnimation(                                                                    # Creo la animación:
    fig,                                                                                  # figura a animar,
    update,                                                                               # función de actualización,
    frames   = range(1, len(x), paso),                                                    # los frames van desde 1 hasta len(x) con paso dado,
    interval = delay,                                                                     # tiempo entre frames (parámetro de la función),
    blit     = False                                                                      # redibujado completo.
  )
  btn_start, btn_stop, btn_reset = crear_botones(fig, ani, update, x, paso, delay)        # Creo botones interactivos => controlo la animación.
  p.show()                                                                                # Muestra la figura.

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def crear_plot3D(
    Xpc: np.ndarray, Ypc: np.ndarray, Zpc: np.ndarray,                           # Coordenadas PC en formato array de numpy.
    Xss: np.ndarray, Yss: np.ndarray, Zss: np.ndarray,                           # Coordenadas SS en formato array de numpy.
    tamaño_ejes: float,                                                          # Tamaño de los ejes del plot (±).
    coord: str = 'pc'                                                            # Sistema de coordenadas a graficar.
) -> tuple[Figure, Axes3D, Line2D, Line2D, np.ndarray, np.ndarray, np.ndarray]:
  """
  Inicializa el gráfico 3D, define los ejes, selecciona el sistema de coordenadas y crea los objetos gráficos que se animarán.
  """
  fig = p.figure()                                                               # Creo la figura,
  ax  = fig.add_subplot(111, projection='3d')                                    # y el eje en 3D.
  ax.set_box_aspect([1,1,1])                                                     # Realizo un gráfico de igual proporción en los 3 ejes (cubo).
  ax.set_xlim([-tamaño_ejes,tamaño_ejes])                                        # Defino los límites del eje x,
  ax.set_ylim([-tamaño_ejes,tamaño_ejes])                                        # del eje y,
  ax.set_zlim([-tamaño_ejes,tamaño_ejes])                                        # y del eje z.
  if coord=='pc':                                                                # Si las coordenadas son PC,
    x,y,z = Xpc/R_m, Ypc/R_m, Zpc/R_m                                            # En x,y,z guardo los array X,Y,Z de PC normalizados por R_m,
    ax.set(xlabel=r'$x_{\text{pc}}$ [$R_M$]', ylabel=r'$y_{\text{pc}}$ [$R_M$]', # y en los ejes x,y,z del gráfico aclaro que se trata de
           zlabel=r'$z_{\text{pc}}$ [$R_M$]')                                    # coordenadas PC.
  elif coord=='ss':                                                              # Si no, si las coordenadas son SS,
    x,y,z = Xss/R_m, Yss/R_m, Zss/R_m                                            # Análogamente en x,y,z guardo los datos SS normalizados.
    ax.set(xlabel=r'$x_{\text{ss}}$ [$R_M$]', ylabel=r'$y_{\text{ss}}$ [$R_M$]', # En los ejes x,y,z del gráfico, aclaro que se trata de
           zlabel=r'$z_{\text{ss}}$ [$R_M$]')                                    # coordenadas SS.
  u,v,w  = esfera_3D(resolución=100)                                             # En u,v,w guardo una superficie paramétrica esférica, que
  ax.plot_surface(u,v,w, color='red', alpha=0.5)                                 # representa a Marte (radio=R_m) en el origen y la grafico.
  line,  = ax.plot([],[],[], lw=2, color='blue')                                 # Inicializa la curva de trayectoria (lw=linewidth) en azul.
  point, = ax.plot([],[],[],  'o', color='red')                                  # Inicializa el punto actual de trayectoria en rojo.
  return fig, ax, line, point, x,y,z                                             # Devuelvo la figura, ejes, trayectoria, punto y datos x,y,z.

#———————————————————————————————————————————————————————————————————————————————————————
def actualizar_frame(
    i: int,                                                          # Índice del frame actual.
    line: Line2D, point: Line2D, ax: Axes3D,                         # Objetos gráficos.
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray       # Datos temporales y espaciales a graficar.
) -> tuple[Line2D, Line2D]:
  """
  La función actualizar frame, actualiza la curva de la trayectoria de MAVEN, el punto actual, y el título del gráfico, graficando todas las
  mediciones hasta el índice i-ésimo, representando el frame i de la animación.
  """
  line.set_data_3d(x[:i], y[:i], z[:i])                              # Actualizo la curva de trayectoria x,y,z.
  point.set_data(x[i-1:i], y[i-1:i])                                 # Actualizo la posición del punto actual en x, en y,
  point.set_3d_properties(z[i-1:i])                                  # y en z.
  t_datetime = pd.to_datetime(np.datetime_as_string(t[i], unit='s')) # Convierto el tiempo del frame actual a objeto datetime.
  tiempo     = t_datetime.strftime('%d/%m/%Y - %H:%M:%S')            # Convierto el tiempo tipo datetime a formato str 'DD/MM/YYYY-HH:MM:SS'.
  ax.set_title(f'Trayectoria de MAVEN — {tiempo}')                   # Coloco el título con el tiempo del frame actual (i-ésimo).
  return line, point                                                 # Devuelvo la curva de trayectoria 3D y el punto actual donde está MAVEN.

#———————————————————————————————————————————————————————————————————————————————————————
def crear_botones(
    fig: Figure, ani: FuncAnimation,          # Figura y animación.
    update,                                   # Función de actualización.
    x: np.ndarray,                            # Datos espaciales (para poder crear una cantidad fija de frames)
    paso: int,                                # Paso entre frames.
    delay: int                                # Retardo entre frames.
) -> tuple[Button, Button, Button]:
  """
  La función crear_botones crea botones interactivos para iniciar, detener y reiniciar la animación en tiempo real.
  """
  ax_start  = p.axes([0.8, 0.5, 0.15, 0.075]) # Defino la posición del botón iniciar usando: [posición x, posición y, ancho, altura]
  ax_stop   = p.axes([0.8, 0.4, 0.15, 0.075]) # y la posición del botón detener,
  ax_reset  = p.axes([0.8, 0.3, 0.15, 0.075]) # y reiniciar.
  btn_start = Button(ax_start,'Iniciar')      # Creo los botones iniciar, 
  btn_stop  = Button(ax_stop, 'Detener')      # detener,
  btn_reset = Button(ax_reset,'Reiniciar')    # y reiniciar.
  def start(event):                           # Funciones llamadoras para iniciar la animación,
    ani.event_source.start()
  def stop(event):                            # para detenerla,
    ani.event_source.stop()
  def reset(event):                           # y para reiniciarla desde el inicio.
    nonlocal ani
    ani.event_source.stop()
    update(1)                                 # Llamo a la función de actualización y la inicializo en el primer frame (i=1).
    ani = FuncAnimation(                      # Recreo la animación nuevamente:
      fig,                                    # Creo la figura,
      update,                                 # uso el nuevo update (i=1)
      frames   = range(1, len(x), paso),      # utilizo la misma cantidad de frames de paso dado,
      interval = delay,                       # con delay entre frames,
      blit     = False)                       # y sin redibujado.
    p.draw()                                  # Dibujo la curva de trayectoria.
  btn_start.on_clicked(start)                 # Asocia las funciones llamadoras a los botones iniciar,
  btn_stop.on_clicked(stop)                   # detener,
  btn_reset.on_clicked(reset)                 # y reiniciar.
  return btn_start, btn_stop, btn_reset       # Devuelvo una tripla de los botones.
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————