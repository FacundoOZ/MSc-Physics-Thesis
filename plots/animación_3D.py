
# Comentar

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para graficar una animación de la trayectoria de la sonda MAVEN en el espacio 3D en coordenadas PC ó SS
#============================================================================================================================================

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as p

from plots.instrumentos   import leer_archivos_MAG, esfera_3D, R_m
from matplotlib.animation import FuncAnimation
from matplotlib.widgets   import Button

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# MAG_trajectory_animation_3D: función para graficar la trayectoria de MAVEN en animación PC ó SS
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def MAG_trajectory_animation_3D(
    directorio: str,                                                                      # Carpeta donde se encuentran los archivos a animar.
    tiempo_inicial: str, tiempo_final: str,                                               # t_0,t_f a animar (en formato DD/MM/YYYY-HH:MM:SS).
    tamaño_ejes: float = 2.5,                                                             # Ajusta el tamaño máx. de ejes x,y,z a la vez.
    paso: int = 100,                                                                      # Ajusta el paso entre datos -> 'velocidad'.
    delay: int = 1,                                                                       # Delay entre frames (recomendado: >=1 ms).
    coord: str = 'pc'                                                                     # Coordenadas a graficar.
) -> None:
  """
  La función MAG_trajectory_animation recibe en formato string un directorio, y un tiempo inicial y final que representan la ruta donde se
  encuentran los archivos que se desean graficar en el intervalo de tiempo indicado, siendo éstos últimos dos en formato DD/MM/YYYY-HH:MM:SS.
  Mediante tamaño_ejes (float) puede ajustarse el tamaño máximo del plot 3D cúbico, y los enteros 'paso' y 'delay' permiten ajustar la
  cantidad de puntos que se recorreran por frame y con qué velocidad, respectivamente. Esta función permite graficar tanto coordenadas PC ó
  SS.
  La función realiza una animación 3D de la trayectoria de MAVEN medida por el instrumento MAG en el sistema de coordenadas indicado.
  """
  data: pd.DataFrame = leer_archivos_MAG(directorio, tiempo_inicial, tiempo_final)        # 
  t, _,_,_, Xpc,Ypc,Zpc, Xss,Yss,Zss = [data[j].to_numpy() for j in range(0, 10)]         # 
  fig, ax, line, point, x,y,z = crear_plot3D(Xpc,Ypc,Zpc,Xss,Yss,Zss, tamaño_ejes, coord) # 
  def update(i):                                                                          # 
    return actualizar_frame(i, line, point, ax, t,x,y,z, paso, delay)                     # 
  ani = FuncAnimation(                                                                    # 
    fig,                                                                                  # 
    update,                                                                               # 
    frames   = range(1, len(x), paso),                                                    # 
    interval = delay,                                                                     # 
    blit     = False                                                                      # 
  )
  btn_start, btn_stop, btn_reset = crear_botones(fig, ani, update, x, paso, delay)        # 
  p.show()                                                                                # 

#———————————————————————————————————————————————————————————————————————————————————————
# Funciones Auxiliares
#———————————————————————————————————————————————————————————————————————————————————————
def crear_plot3D(
    Xpc: np.ndarray, Ypc: np.ndarray, Zpc: np.ndarray,                           #
    Xss: np.ndarray, Yss: np.ndarray, Zss: np.ndarray,                           #
    tamaño_ejes: float,                                                          #
    coord: str = 'pc'                                                            #
) -> None:
  """
  Initializes the 3D plot and sets the axis properties.
  """
  fig = p.figure()                                                               #
  ax  = fig.add_subplot(111, projection='3d')                                    #
  ax.set_box_aspect([1,1,1])                                                     #
  ax.set_xlim([-tamaño_ejes,tamaño_ejes])                                        #
  ax.set_ylim([-tamaño_ejes,tamaño_ejes])                                        #
  ax.set_zlim([-tamaño_ejes,tamaño_ejes])                                        #
  if coord=='pc':                                                                #
    x,y,z = Xpc/R_m, Ypc/R_m, Zpc/R_m                                            #
    ax.set(xlabel=r'$x_{\text{pc}}$ [$R_M$]', ylabel=r'$y_{\text{pc}}$ [$R_M$]', #
           zlabel=r'$z_{\text{pc}}$ [$R_M$]')                                    #
  elif coord=='ss':                                                              #
    x,y,z = Xss/R_m, Yss/R_m, Zss/R_m                                            #
    ax.set(xlabel=r'$x_{\text{ss}}$ [$R_M$]', ylabel=r'$y_{\text{ss}}$ [$R_M$]', #
           zlabel=r'$z_{\text{ss}}$ [$R_M$]')                                    #
  u,v,w  = esfera_3D(resolución=100)                                             #
  ax.plot_surface(u,v,w, color='red', alpha=0.5)                                 #
  line,  = ax.plot([],[],[], lw=2, color='blue')                                 #
  point, = ax.plot([],[],[],  'o', color='red')                                  #
  return fig, ax, line, point, x,y,z                                             #

#———————————————————————————————————————————————————————————————————————————————————————
def actualizar_frame(
    i,                                                               #
    line, point, ax,                                                 #
    t, x, y, z,                                                      #
    paso,                                                            #
    delay                                                            #
) -> None:
  """
  """
  line.set_data_3d(x[:i], y[:i], z[:i])                              #
  point.set_data(x[i-1:i], y[i-1:i])                                 #
  point.set_3d_properties(z[i-1:i])                                  #
  t_datetime = pd.to_datetime(np.datetime_as_string(t[i], unit='s')) #
  tiempo     = t_datetime.strftime('%d/%m/%Y - %H:%M:%S')            #
  ax.set_title(f'Trayectoria de MAVEN — {tiempo}')                   #
  return line, point                                                 #

#———————————————————————————————————————————————————————————————————————————————————————
def crear_botones(
    fig, ani, update,                         #
    x,                                        #
    paso,                                     #
    delay                                     #
) -> None:
  """
  """
  ax_reset  = p.axes([0.8, 0.5, 0.15, 0.075]) # [pos_x, pos_y, tamaño_x, tamaño_y]
  ax_start  = p.axes([0.8, 0.4, 0.15, 0.075]) #
  ax_stop   = p.axes([0.8, 0.3, 0.15, 0.075]) #
  btn_start = Button(ax_start,'Start')        #
  btn_stop  = Button(ax_stop, 'Stop')         #
  btn_reset = Button(ax_reset,'Reset')        #
  def start(event):                           #
    ani.event_source.start()                  #
  def stop(event):                            #
    ani.event_source.stop()                   #
  def reset(event):                           #
    nonlocal ani                              #
    ani.event_source.stop()                   #
    update(1)                                 #
    ani = FuncAnimation(                      #
      fig,                                    #
      update,                                 #
      frames   = range(1, len(x), paso),      #
      interval = delay,                       #
      blit     = False                        #
    )                                         #
    p.draw()                                  #
  btn_start.on_clicked(start)                 #
  btn_stop.on_clicked(stop)                   #
  btn_reset.on_clicked(reset)                 #
  return btn_start, btn_stop, btn_reset       #
#———————————————————————————————————————————————————————————————————————————————————————

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————