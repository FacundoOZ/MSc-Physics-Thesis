
# Terminado

#============================================================================================================================================
# Tesis de Licenciatura | Archivo para correr un algoritmo de k-vecinos cercanos (KNN)
#============================================================================================================================================

import os
import numpy  as np
import pandas as pd
import pickle

from typing                import Union
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Módulos Propios:
from base_de_datos.conversiones   import módulo, R_m, segundos_a_día
from base_de_datos.lectura        import leer_archivos_MAG, leer_archivo_Fruchtman
from machine_learning.estadística import estadística_B, estadística_R, estadística_componentes_B, estadística_componentes_R

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Clasificador_KNN_Binario : para la detección de Bow Shocks mediante mediciones de campo magnético del instrumento MAG de la sonda MAVEN.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Clasificador_KNN_Binario:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Inicializador (Constructor): Inicia las variables características del KNN.
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def __init__(self, K: int, variables: Union[list[str], None] = None, promedio: int = 1, ventana: int = 300,
               ventanas_NBS: list[int] = [-1,1,2]) -> None:
    """Inicializador de atributos de la clase Clasificador_KNN_Binario para la detección de Bow Shocks.
    -El parámetro 'K' es un entero que determina la cantidad de ventanas vecinas que se utilizarán para entrenar al KNN (recomendado (1 a 30)).
    -El parámetro 'variables' es una lista de strings que permite elegir qué magnitudes físicas medidas por el instrumento MAG de la sonda
    MAVEN se desean utilizar para el entrenamiento del algoritmo KNN. Si su valor es None, utiliza en forma predeterminada las variables
    ['B','Xss','Yss','Zss']. Los valores posibles son: ['B','R','Bx','By','Bz','Xpc','Ypc','Zpc','Xss','Yss','Zss'].
    -El parámetro 'promedio' es un entero que determina la suavización (y por lo tanto la rapidez en la lectura) que se utilizará al leer
    los archivos MAG (lee 'promedio'-cantidad de datos, toma la media y continua con los siguientes 'promedio'-datos).
    -El parámetro 'ventana' es un entero que permite ajustar la cantidad de tiempo en segundos que tendrá el ancho de las ventanas para 
    entrenar al KNN (recomendado entre 60 y 600), y cuya finalidad es representar de forma adecuada el ancho de duración de los bow shocks.
    -El parámetro 'ventanas_NBS' es una lista de enteros que representa cuáles ventanas próximas a cada bow shock (BS) se utilizaran como
    región no-bow shock (NBS) para el entrenamiento del KNN.
    """
    if K <= 0:                                                                             # Si K<=0, no es entero válido,
      raise ValueError("'K' debe ser un entero positivo (recomendado 1 ≤ K ≤ 30).")        # => devuelvo un mensaje.
    if variables is None:                                                                  # Si no se definen las variables,
      variables: list[str] = ['B','Xss','Yss','Zss']                                       # utilizo las predeterminadas [|B|,Xss,Yss,Zss]
    if not all(isinstance(v, str) for v in variables):                                     # Si las variables pasadas por parámetro no son
      raise TypeError("'variables' debe ser una lista de strings.")                        # strings => devuelvo un mensaje.
    if promedio < 1:                                                                       # Si el promedio es negativo,
      raise ValueError("'promedio' debe ser un entero positivo.")                          # => devuelvo un mensaje.
    if ventana <= 0:                                                                       # Si la ventana es <= 0 no es un tiempo válido,
      raise ValueError("'ventana' debe ser un entero positivo (recomendado 60 ≤ v ≤ 600).")# => devuelvo un mensaje.
    self.K: int                  = K                                                       # Metaparámetro K para las ventanas vecinas.
    self.variables: list[str]    = list(variables)                                         # Variables a utilizar del vector característico.
    self.promedio: int           = promedio                                                # Promedio a utilizar de las mediciones MAVEN MAG.
    self.ventana: int            = ventana                                                 # Ancho de la ventana de puntos (en segundos).
    self.ventanas_NBS: list[int] = list(ventanas_NBS)                                      # Posición ventanas NBS respecto a BS (a entrenar).
    self.ventana_puntos: int     = max(1,(ventana+promedio-1)//promedio)                   # Calculo puntos reales por ventana (sobre promedio).
    self.entrenado: bool = False                                                           # Booleano del estado del KNN.
    self.scaler = StandardScaler()                                                         # Re-escaleo de variables.
    self.knn    = KNeighborsClassifier(                                                    # Clasificador KNN.
      n_neighbors = self.K,                                                                # Número de vecinos.
      weights     = 'distance',                                                            # Pesos (distancia).
      metric      = 'euclidean',                                                           # Métrica a utilizar (euclídea predeterminada).
      n_jobs      = -1                                                                     # Permite utilizar todo el CPU disponible.
    )

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Vector Característico: Calcula la estadística de una ventana con las variables ingresadas al KNN (por ej.: ['B','Xss','Yss','Zss'])
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def vector_característico(self, data_MAG: pd.DataFrame) -> np.ndarray:
    """La función vector característico recibe un dataframe 'data_MAG' de longitud menor o igual a 'ventana' del KNN, que es del tipo de
    archivo MAG recortado por recorte de Vignes, cuyas columnas son de la forma:
    día_decimal Bx By Bz Xpc Ypc Zpc Xss Yss Zss
    ....        .. .. .. ... ... ... ... ... ...
    y extrae el vector característico por ventana para alimentar al KNN, con las variables pasadas por parámetro al Clasificador_KNN_Binario.
    Devuelve una lista de valores float convertida a array (np.ndarray[list[float]]) donde cada valor representa las magnitudes estadísticas
    (del archivo estadística.py) correspondientes a las 'variables' pasadas por parámetro al KNN."""
    vector: list[float] = []                                                     # Inicializo el vector característico vacío como list[float]
    Bx,By,Bz,Xpc,Ypc,Zpc,Xss,Yss,Zss = [data_MAG[j].to_numpy() for j in range(1,10)]# Extraigo las magnitudes físicas del archivo tipo MAG. 
    pos: dict[str,np.ndarray] = {'Xpc':Xpc, 'Ypc':Ypc, 'Zpc':Zpc,                # Creo un dicc con las posiciones de la sonda en sistema PC
                                 'Xss':Xss, 'Yss':Yss, 'Zss':Zss}                # y en sistema SS.
    mag: dict[str,np.ndarray] = {'Bx' : Bx, 'By' : By, 'Bz' : Bz}                # Creo un dicc con las componentes de campo magnético.
    for var in self.variables:                                                   # Para cada elemento de la lista de variables de clase KNN,
      if var=='B':                                                               # Si la variable es B (módulo de campo magnético),
        vector.extend(estadística_B(módulo(Bx,By,Bz)))                           # uso su estadística especial que contempla gradientes.
      elif var=='R':                                                             # Si no, si es la posición de la sonda (módulo de r),
        vector.extend(estadística_R(módulo(Xss,Yss,Zss, norm=R_m)))              # uso su estadística y NORMALIZO => dim de B es similar a R.
      elif var in mag:                                                           # Si no, si uso las componentes del campo,
        vector.extend(estadística_componentes_B(mag[var]))                       # eso una estadística especial para ellas.
      elif var in pos:                                                           # Si no, si pertenecen a pos,
        vector.extend(estadística_componentes_R((pos[var]/R_m)))                 # uso estadística especial para las posiciones => NORMALIZO!
      else:                                                                      # Si no,
        raise ValueError(f"Variable desconocida: {var}")                         # la variables es inválida => devuelvo un mensaje.
    return np.array(vector)                                                      # Devuelvo el vector convertido a np.ndarray.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Muestras de Entrenamiento BS y NBS: Calcula y devuelve los vectores característicos de las ventanas BS (Fruchtman) y NBS (de MAG).
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def muestras_entrenamiento(self, data_MAG: pd.DataFrame, data_Fruchtman: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    La función muestras_entrenamiento recibe un dataframe 'data_MAG' que contiene los datos de las mediciones recortadas por Vignes del
    instrumento MAG en el intervalo temporal que se haya indicado (1 ó más años) y recibe un np.ndarray 'data_Fruchtman' que contiene los
    tiempos en los que ocurrieron los choques (en día decimal) que Fruchtman ha registrado, pertenecientes al mismo intervalo de tiempo
    indicado en MAG. La función calcula la ventana del BS Fruchtman y las ventanas NBS vecinas a éste para cada día decimal del archivo 
    Fruchtman, y luego calcula los vectores característicos de cada una. Devuelve una tupla cuya primera componente son los vectores 
    característicos BS y NBS (X), y cuya segunda componente son las etiquetas (y) (1 para los BS y 0 para los NBS) en formatos np.ndarray.
    """
    X: list[np.ndarray]            = []                                             # Inicializo una lista de vectores característicos 'X'.
    y: list[int]                   = []                                             # Inicializo una lista de etiquetas (enteros 0 ó 1) 'y'.
    media_ventana: int             = (self.ventana_puntos)//2                       # Calculo el ancho de ventana/2 => división entera! (//).
    t_MAG: np.ndarray = data_MAG[0].to_numpy()                                      # Obtengo días decimales del archivo MAG y convierto a np.
    for t_Fru in data_Fruchtman:                                                    # Para cada BS (día decimal) del archivo Fruchtman:
      j: int  = np.searchsorted(t_MAG, t_Fru)                                       # Busco el j de data_MAG más cercano a t_BS en O(log n).
      if j == 0:                                                                    # Si j es 0,
        j_BS = 0                                                                    # tomo el primero como índice del BS.
      elif j == len(t_MAG):                                                         # Si no, si es el último,
        j_BS = len(t_MAG) - 1                                                       # Tomo el anterior al último.
      else:                                                                         # Si no es ninguno de los casos límite,
        j_BS = j if abs(t_MAG[j] - t_Fru) < abs(t_MAG[j-1] - t_Fru) else j-1        # tomo aquel tal que t_MAG[j] esté + cerca de t_Fru.
      t0_BS: int = j_BS - media_ventana                                             # El inicio de la ventana será dicho j - (ventana/2),
      tf_BS: int = j_BS + media_ventana + (self.ventana_puntos % 2)                 # y el final será j+(ventana/2) (contemplo caso impar).
      if t0_BS < 0 or tf_BS > len(data_MAG):                                        # Si los tiempos se salen de los límites del archivo,
        continue                                                                    # omito esta ventana.
      v_BS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_BS : tf_BS])   # Calculo el vector característico para la ventana BS,
      X.append(v_BS); y.append(1)                                                   # lo agrego a la lista de vectores con etiqueta=1 (BS).
      for desplazamiento in (self.ventanas_NBS):                                    # Para cada desplazamiento de la lista ventanas_NBS,
        j_NBS: int  = j_BS + (desplazamiento*(self.ventana_puntos))                 # el j_NBS (centrado), será el j_BS + dicho desp*ventana.
        t0_NBS: int = j_NBS - media_ventana                                         # Calculo el tiempo inicial,
        tf_NBS: int = j_NBS + media_ventana                                         # y final de la ventana para el nuevo j_NBS.
        if t0_NBS < 0 or tf_NBS > len(data_MAG):                                    # Si los tiempos se salen de los límites del archivo,
          continue                                                                  # omito esta ventana.
        v_NBS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_NBS:tf_NBS])# Si no, calculo el vector de la ventana NBS,
        X.append(v_NBS); y.append(0)                                                # lo agrego a la lista de vectores con etiqueta=0 (NBS).
    return np.array(X), np.array(y)                                                 # Convierto las listas a np.array para el KNN.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Clasificar Muestras: Entrena al clasificador KNN con las mediciones obtenidas de muestras_entrenamiento.
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def clasificar_muestras(self, X: np.ndarray, y: np.ndarray) -> None:
    """La función clasificar_muestras recibe los dos parámetros dados por la tupla de np.ndarrays output de la función muestras_entrenamiento,
    y entrena al clasificador KNN, y devuelve un mensaje con la cantidad de muestras totales que se utilizaron de BS y de NBS y el total."""
    self.scaler.fit(X)
    X_fit = self.scaler.transform(X)
    self.knn.fit(X_fit, y)
    self.entrenado = True
    print(f'El Clasificador_KNN_Binario ha sido entrenado con: {len(X)} muestras, BS={sum(y)}, NBS={len(y)-sum(y)}.')

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Predecir Parámetros de Ventana: Predice las etiquetas (BS=1, NBS=0), probabilidades (%) y índices de ventana (j) de nuevos datos MAG.
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def predecir_ventana(self, data_MAG: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    La función predecir_ventana recibe un dataframe 'data_MAG' con datos del tipo MAG (día_decimal Bx By Bz Xpc Ypc Zpc Xss Yss Zss) del
    recorte de Vignes, y calcula las predicciones de etiquetas (BS=1 ó NBS=0), las probabilidades de los BS y NBS, y los índices j de las
    ventanas correspondientes, devolviendo estos tres parámetros en formato tripla de np.ndarrays. Calcular las predicciones en cada ventana,
    y calcula los vectores característicos con las variables que se hayan indicado al KNN.
      Devuelve:
        prob[:,1] = Probabilidad de la clase BS.
        prob[:,0] = Probabilidad de la clase NBS.
    """
    if not self.entrenado:                                                 # Si todavía no se entrenó al KNN,
      raise RuntimeError('El clasificador KNN no ha sido entrenado.')      # devuelvo un mensaje.
    etiqueta:     list[int]         = []                                   # Inicializo una lista para guardar las etiquetas (1: BS ó 0: NBS),
    probabilidad: list[list[float]] = []                                   # una para guardar probabilidades de bow shocks y no bow shocks,
    j_ventana:    list[int]         = []                                   # y una para guardar los índice de los centros de las ventanas.
    for i in range(0, len(data_MAG), self.ventana_puntos):                 # Para i de 0 al final del archivo MAG:
      j_0: int = i                                                         # obtengo el índice del inicio de la ventana actual,
      j_f: int = i + self.ventana_puntos                                   # y el índice del final de la ventana actual.
      if j_f > len(data_MAG):                                              # 
        break                                                              # 
      ventana: pd.DataFrame = data_MAG[j_0 : j_f]                          # Obtengo solamente los datos MAG de esa ventana,
      v: np.ndarray = self.vector_característico(ventana)                  # y calculo su vector característico y lo guardo en la variable v.
      if v is not None:                                                    # Si el vector característico no es None,
        v_escalado: np.ndarray = self.scaler.transform(v.reshape(1,-1))    # lo re-escalo (funciona mejor pues el KNN trabaja con distancias).
        pred: int              = self.knn.predict(v_escalado)[0]           # Obtengo las predicciones de etiqueta bow shock ó no bow shock.
        prob: np.ndarray       = self.knn.predict_proba(v_escalado)[0]     # Obtengo las probabilidades,
        etiqueta.append(pred)                                              # y agrego ambos a la lista de etiquetas,
        probabilidad.append(prob)                                          # y a la lista de probabilidades.
        j_ventana.append(j_0 + (self.ventana_puntos//2))                   # Obtengo el índice de la ventana como el medio de j_0 y j_f.
    return np.array(etiqueta), np.array(probabilidad), np.array(j_ventana) # Devuelvo listas de etiquetas, probabilidades y j en np.arrays.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Guardado y Exportado del Modelo Clasificador:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def save(self, directorio: str, nombre_archivo: str) -> None:
    """
    Guarda el Clasificador_KNN_Binario entrenado en la ruta final: 'directorio'+'KNN'+'entrenamiento'+'nombre_archivo'.
    """
    ruta_entrenamiento: str = os.path.join(directorio, 'KNN', 'entrenamiento')
    os.makedirs(ruta_entrenamiento, exist_ok=True)
    with open(os.path.join(ruta_entrenamiento, nombre_archivo), 'wb') as archivo:
      pickle.dump(self, archivo)

  @staticmethod
  def load(directorio: str, nombre_archivo: str) -> 'Clasificador_KNN_Binario':
    """
    Carga un Clasificador_KNN_Binario entrenado extraído de la ruta final: 'directorio'+'KNN'+'entrenamiento'+'nombre_archivo'.
    """
    ruta_entrenamiento: str = os.path.join(directorio, 'KNN', 'entrenamiento')
    os.makedirs(ruta_entrenamiento, exist_ok=True)
    with open(os.path.join(ruta_entrenamiento, nombre_archivo), 'rb') as archivo:
      return pickle.load(archivo)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————



#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# entrenar: función para entrenar un modelo KNN con datos de Fruchtman (BS) y MAG (NBS) con libertad para ajustar todos sus parámetros.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def entrenar(
    directorio: str,                                                            # Carpeta donde se encuentran las mediciones Fruchtman y MAG.
    años_entrenamiento: list[str],                                              # Años que se desean entrenar.
    K: int,                                                                     # Cantidad de vecinos más cercanos a utilizar por el KNN.
    variables: list[str] = ['B','Xss','Yss','Zss'],                             # Variables a utilizar para el vector característico del KNN.
    promedio: int = 1,                                                          # Promedio para suavizar las muestras de MAVEN MAG.
    ventana: int = 300,                                                         # Ancho de ventana en segundos a utilizar (representa el BS).
    ventanas_NBS: list[int] = [-1,1,2],                                         # Posiciones de ventanas vecinas al BS para entrenar zona NBS.
    MAG_cache: Union[dict[str, pd.DataFrame], None] = None                      # Contenido de la lectura de archivos MAG.
) -> Clasificador_KNN_Binario:
  """
  La función entrenar recibe un directorio que contiene las mediciones de MAVEN MAG y los archivos de Fruchtman con los bow shock detectados,
  una lista de strings años_entrenamiento que representa los años que se desea entrenar al knn, el metaparámetro K del knn (número de vecinos),
  las variables que se desean utilizar como vector característico, el promedio de las muestras MAG para eliminar el ruido de los datos y
  reducir el tiempo de compilación, el ancho de ventana (en segundos) para modelar el BS, las ventanas NBS que se desean considerar, y la
  variable MAG_cache, que permite extraer rápidamente todos los archivos MAG leídos, si ya fueron leídos previamente (en otras funciones).
  Devuelve el KNN, es decir, la clase Clasificador_KNN_Binario con el algoritmo KNN entrenado habiendo utilizado todas las variables que se
  han pasado por parámetro.
  """
  if años_entrenamiento == ['2014']:                                            # Si el año de entrenamiento es solo el 2014,
    raise ValueError('El año 2014 no tiene suficientes muestras para entrenar.')# => devuelvo un mensaje (son menos de 20 datos).
  knn: Clasificador_KNN_Binario = Clasificador_KNN_Binario(                     # En la variable 'knn' creo la clase Clasificador_KNN_Binario,
    K                     = K,                                                  # con todos los valores que han sido pasados por parámetro a
    variables             = variables,                                          # la función entrenar.
    ventana               = ventana,
    ventanas_NBS          = ventanas_NBS,
  )
  knn.promedio = promedio                                                       # El promedio del knn, es el pasado por parámetro a entrenar.
  X: list[np.ndarray] = []                                                      # Inicializo una lista de vectores característicos 'X'.
  y: list[int]        = []                                                      # Inicializo una lista de etiquetas (enteros 0 ó 1) 'y'.
  for año in años_entrenamiento:                                                # Para cada año de la lista de años_entrenamiento,
    if MAG_cache is not None:                                                   # Si los archivos MAG ya se leyeron, entonces leo
      data_MAG: pd.DataFrame = MAG_cache[año]                                   # los años del dicc MAG_cache y los guardo en data_MAG.
    else:                                                                       # Si no, debo leerlos.
      ruta_MAG: str          = os.path.join(directorio,'recorte_Vignes')        # Obtengo la carpeta donde están todos los archivos MAG.
      t0, tf                 = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'   # Obtengo el intervalo de tiempo de todo el año de MAG, y
      data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, promedio)    # leo todos los archivos MAG del año con el promedio indicado.
    data_Fru: pd.DataFrame = leer_archivo_Fruchtman(directorio, año)            # Leo el archivo Fruchtman del año correspondiente.
    dias_Fru: pd.Series    = data_Fru.iloc[:,0].astype(float)                   # Extraigo días decimales de Fruchtman y convierto a float.
    t_BS = pd.Timestamp(f'{año}-01-01') + pd.to_timedelta(dias_Fru-1, unit='D') # Convierto los tiempos BS a objetos datetime adecuadamente.
    X_año, y_año = knn.muestras_entrenamiento(data_MAG, t_BS.to_numpy())        # Obtengo muestras de entrenamiento del año con data_MAG y BS.
    if len(X_año) == 0:                                                         # Si no hay muestras entrenadas,
      continue                                                                  # pasamos a la siguiente iteración del for (no las agrego).
    X.append(X_año)                                                             # Agrego el primer np.ndarray (vectores característicos) a X,
    y.append(y_año)                                                             # y el segundo np.ndarray (etiquetas) a la lista y.
  knn.clasificar_muestras(np.vstack(X), np.concatenate(y))                      # Clasifico las muestras (X,y) apiladas de todos los años. 
  print('El Clasificador_KNN_Binario se ha entrenado correctamente.')           # Devuelvo un mensaje de que el entrenamiento fue exitoso.
  return knn                                                                    # Devuelvo el knn entrenado para utilizarlo para predecir.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# clasificar: función para clasificar etiquetas BS y NBS a partir de un modelo KNN previamente entrenado.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def clasificar(directorio: str, knn: Clasificador_KNN_Binario, predecir_años: list[str]) -> None:
  """
  La función clasificar recibe un directorio que contiene las carpetas de 'KNN' y subcarpeta 'entrenamiento' donde se encuentra un modelo de
  KNN previamente entrenado para poder cargarlo en la variable 'knn' de tipo Clasificador_KNN_Binario (una clase KNN con sus parámetros
  correspondientes), y recibe una lista de strings que representa los años cuyas mediciones (ventanas) se desean predecir. Devuelve dos
  archivos: 'probabilidades_{año}.txt' y 'tiempos_BS_{año}.txt' para cada año ingresado; que contienen las probabilidades de NBS y BS con la
  predicción encontrada, y los tiempos en formato día decimal de los BS predichos, respectivamente.
  Archivo 'probabilidades_{año}.txt':
    NBS    BS    Predicción
    ..     ..    ...
  Archivo 'tiempos_BS_{año}.txt':
    día_decimal
    ...
  """
  ruta_MAG: str  = os.path.join(directorio,'recorte_Vignes')                    # Obtengo la carpeta donde están los archivos MAG recortados.
  ruta_pred: str = os.path.join(directorio, 'KNN', 'predicción')                # Obtengo la ruta donde guardaré los archivos de predicción.
  os.makedirs(ruta_pred, exist_ok=True)                                         # Si no existe, la creo.
  for año in predecir_años:                                                     # Para cada año de 'predecir_años' cuyos BS deseo detectar,
    t0, tf = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'                     # obtengo el intervalo de tiempo de todo el año de MAG,
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, knn.promedio)  # leo archivos en ese intervalo con el promedio usado en KNN,
    print(f'\nClasificando mediciones del año {año} ...\n')                     # y devuelvo un mensaje de que se está ejecutando el KNN.
    pred, prob, j_ventana = knn.predecir_ventana(data_MAG)                      # Obtengo etiqueta (predicción), probabilidad, j por ventana. 
    print('Clasificación completada.')                                          # Aviso que el KNN terminó la predicción.
    j_BS: np.ndarray = j_ventana[pred == 1]                                     # Recupero los índices j centrales de las ventanas BS,
    t_BS: pd.DatetimeIndex = pd.to_datetime(data_MAG.iloc[:,0].to_numpy()[j_BS])# y obtengo cuándo ocurrieron en formato datetime.
    print(f'Ventanas etiquetadas: {len(pred)}')                                 # Aviso la cantidad de ventanas que se utilizaron,
    print(f'Ventanas BS: {len(t_BS)} ({len(t_BS)/len(pred)*100:.2f} %).')       # y cuántas de ellas se clasificaron como Bow Shock.
    if len(t_BS) > 0:                                                           # Si hay al menos 1 bow shock,
      dias_dec: np.ndarray | list[float] = segundos_a_día(                      # calculo el día decimal del año (1 = 1 de enero)
        (t_BS - pd.Timestamp(f'{año}-01-01')).total_seconds()) + 1              # convirtiendo segundos a formato día.
    else:                                                                       # Si no,
      dias_dec = []                                                             # la lista es vacía.
    probabilidades: pd.DataFrame = pd.DataFrame({'NBS':prob[:,0],'BS':prob[:,1],# Genero un dataframe con las probabilidades BS, NBS,
                                                 'Predicción': pred})           # y con la predicción correspondiente.
    tiempos_BS: pd.DataFrame     = pd.DataFrame({'día_decimal': dias_dec})      # Genero dataframe con tiempos_BS predichos (en día decimal).
    ruta_prob: str = os.path.join(ruta_pred, f'probabilidades_{año}.txt')       # Obtengo la ruta + nombre_completo de los archivos de salida
    ruta_BS: str   = os.path.join(ruta_pred, f'tiempos_BS_{año}.txt')           # para las probabilidades, y los tiempos BS a detectar.
    probabilidades.to_csv(ruta_prob, sep=' ', index=False)                      # Exporto los archivos .txt con los nombres correspondientes
    tiempos_BS    .to_csv(ruta_BS,   sep=' ', index=False)                      # en la carpeta directorio + 'KNN' + 'predicción'.

def diagnosticar_knn(knn: Clasificador_KNN_Binario, directorio: str, año_test: str = '2020'):
  """
  Diagnostic function to check if KNN is working correctly.
  """
  print(f"\n{'='*60}")
  print(f"DIAGNÓSTICO DEL KNN - AÑO {año_test}")
  print(f"{'='*60}")
  ruta_MAG = os.path.join(directorio, 'recorte_Vignes')                         # 1. Load test data
  t0, tf = f'1/1/{año_test}-00:00:00', f'31/12/{año_test}-23:59:59'
  data_MAG = leer_archivos_MAG(ruta_MAG, t0, tf, knn.promedio)
  if len(data_MAG) == 0:
    print("ERROR: No se encontraron datos MAG")
    return
  print(f"1. Datos MAG cargados: {len(data_MAG)} registros")
  print(f"   Columnas: {list(data_MAG.columns)}")
  print(f"\n2. Probando vector característico...")                              # 2. Test vector característico on a sample window
  sample_window = data_MAG.iloc[0:min(300, len(data_MAG))]
  vector = knn.vector_característico(sample_window)
  if vector is not None:
    print(f"   Vector creado: longitud={len(vector)}")
    print(f"   Valores mín/máx: {vector.min():.3f} / {vector.max():.3f}")
    print(f"   ¿Contiene NaN? {np.any(np.isnan(vector))}")
  else:
    print("   ERROR: No se pudo crear el vector")
    return
  print(f"\n3. Probando predicciones...")                                       # 3. Test prediction on first few windows
  pred, prob, j_ventana = knn.predecir_ventana(data_MAG.iloc[0:10000])          # First 10000 points for speed
  if len(pred) > 0:
    print(f"   Predicciones realizadas: {len(pred)} ventanas")
    print(f"   BS detectados: {sum(pred)} ({sum(pred)/len(pred)*100:.1f}%)")
    print(f"   Probabilidad promedio BS: {prob[:,1].mean():.3f}")
    print(f"   Probabilidad promedio NBS: {prob[:,0].mean():.3f}")
    prob_sum = prob.sum(axis=1)                                                 # Check probability consistency
    if np.allclose(prob_sum, 1.0, atol=1e-5):
      print(f"   ✓ Probabilidades suman 1 correctamente")
    else:
      print(f"   ✗ ERROR: Probabilidades no suman 1")
      print(f"     Ejemplo: {prob_sum[:5]}")
  else:
    print("   ERROR: No se realizaron predicciones")
  print(f"\n4. Estadísticas del entrenamiento:")                                # 4. Check training statistics
  print(f"   Entrenado: {knn.entrenado}")
  print(f"   K: {knn.K}")
  print(f"   Variables: {knn.variables}")
  print(f"   Ventana: {knn.ventana}s")
  print(f"   Promedio: {knn.promedio}")
  print(f"\n5. Análisis de características de bow shocks:")                     # 5. Test with known bow shock characteristics
  if len(data_MAG) > 1000:                                                      # Find periods with high B field variability (typical of shocks)
    Bx,By,Bz = [data_MAG.iloc[:,j].to_numpy() for j in [1,2,3]]
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    window_size = knn.ventana                                                 # Calculate moving standard deviation
    if len(B_mag) > window_size:
      B_std = pd.Series(B_mag).rolling(window_size).std().values
      high_var_threshold = np.percentile(B_std[~np.isnan(B_std)], 90)       # Find high variability periods
      high_var_indices = np.where(B_std > high_var_threshold)[0]
      print(f"   Períodos de alta variabilidad (> percentil 90): {len(high_var_indices)}")
      print(f"   Esto debería correlacionar con detecciones BS")
  print(f"\n{'='*60}")
  print(f"DIAGNÓSTICO COMPLETADO")
  print(f"{'='*60}")
  return pred, prob

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————