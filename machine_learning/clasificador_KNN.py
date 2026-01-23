
# Editar

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
from base_de_datos.conversiones   import módulo, R_m
from base_de_datos.lectura        import leer_archivos_MAG
from machine_learning.estadística import estadística, estadística_módulos

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Clasificador_KNN_Binario : para la detección de Bow Shocks mediante mediciones de campo magnético del instrumento MAG de la sonda MAVEN.
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
class Clasificador_KNN_Binario:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Inicializador (Constructor): Inicia las variables características del KNN.
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def __init__(self, K: int, variables: Union[list[str], None] = None, promedio: int = 1, ventana: int = 300,
               ventanas_NBS: list[int] = [-1,1,2], superposición_ventana: int = 50) -> None:
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
    -El parámetro 'superposición_ventana' representa el porcentaje (1 % a 100 %) de superposición entre ventanas utilizado en la predicción.
    """
    if K <= 0:                                                                             # Si K<=0, no es entero válido,
      raise ValueError("'K' debe ser un entero positivo (recomendado 1 ≤ K ≤ 30).")        # => devuelvo un mensaje.
    if variables is None:                                                                  # Si no se definen las variables,
      variables: list[str] = ['B','Xss','Yss','Zss']                                       # utilizo las predeterminadas [|B|,Xss,Yss,Zss]
    if not all(isinstance(v, str) for v in variables):                                     # Si las variables pasadas por parámetro no son
      raise TypeError("'variables' debe ser una lista de strings.")                        # strings => devuelvo un mensaje.
    if ventana <= 0:                                                                       # Si la ventana es <= 0 no es un tiempo válido,
      raise ValueError("'ventana' debe ser un entero positivo (recomendado 60 ≤ v ≤ 600).")# => devuelvo un mensaje.
    if not (0 < superposición_ventana <= 100):                                             # Si la superposición no está entre 1 % y 100 %,
      raise ValueError("'superposición_ventana' debe ser un entero entre 1 y 100.")        # => devuelvo un mensaje.
    self.K: int                     = K                                                    # Metaparámetro K para las ventanas vecinas.
    self.variables: list[str]       = list(variables)                                      # Variables a utilizar del vector característico.
    self.promedio: int              = promedio                                             # Promedio a utilizar de las mediciones MAVEN MAG.
    self.ventana: int               = ventana                                              # Ancho de la ventana de puntos (en segundos).
    self.ventanas_NBS: list[int]    = list(ventanas_NBS)                                   # Posición ventanas NBS respecto a BS (a entrenar).
    self.superposición_ventana: int = superposición_ventana                                # Superposición (%) de ventanas para la predicción.
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
    B, R = módulo(Bx,By,Bz), módulo(Xss,Yss,Zss, norm=R_m)                       # Calculo módulos: NORMALIZO R! => B,R son del mismo orden. 
    dicc: dict[str,np.ndarray] = {                                               # Creo diccionario con todas las variables del archivo MAG.
      'Bx' : Bx , 'By' : By , 'Bz' : Bz ,                                        # Componentes de campo magnético (en sistema PC)
      'Xpc': Xpc, 'Ypc': Ypc, 'Zpc': Zpc, 'Xss': Xss, 'Yss': Yss, 'Zss': Zss}    # Componentes de posición de la sonda (en sistemas PC y SS).
    for var in self.variables:                                                   # Para cada elemento de la lista de variables de la clase KNN,
      if var in ('B','R'):                                                       # si las variables son B ó R (no pertenecen al diccionario),
        vector.extend(estadística_módulos(B if var=='B' else R))                 # calculo sus estadísticas (módulos) y las agrego a vector.
      elif var in dicc:                                                          # si no, si pertenecen al dicc,
        vector.extend(estadística(dicc[var]))                                    # calculo sus estadísticas (para componentes) y las agrego.
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
    media_ventana: int             = (self.ventana) // 2                            # Calculo el ancho de ventana/2 => división entera! (//).
    días_decimales_MAG: np.ndarray = data_MAG[0].to_numpy()                         # Obtengo días decimales del archivo MAG y convierto a np.
    for dia_decimal in data_Fruchtman:                                              # Para cada BS (día decimal) del archivo Fruchtman:
      j_BS: int        = np.searchsorted(días_decimales_MAG, dia_decimal)           # Busco el j de data_MAG más cercano a t_BS en O(log n).
      t0_BS: int       = j_BS - media_ventana                                       # El inicio de la ventana será dicho j - (ventana/2),
      tf_BS: int       = j_BS + media_ventana                                       # y el final será dicho j + (ventana/2).
      if (j_BS - media_ventana) < 0 or (j_BS + media_ventana) > len(data_MAG):      # Si los tiempos se salen de los límites del archivo,
        continue                                                                    # omito esta ventana.
      v_BS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_BS : tf_BS])   # Calculo el vector característico para la ventana BS,
      X.append(v_BS)                                                                # lo agrego a la lista de vectores característicos X,
      y.append(1)                                                                   # y su etiqueta será 1 (bow shock).
      for desplazamiento in (self.ventanas_NBS):                                    # Para cada desplazamiento de la lista ventanas_NBS,
        j_NBS: int        = j_BS + (desplazamiento*(self.ventana))                  # el j_NBS (centrado), será el j_BS + dicho desp*ventana.
        t0_NBS: int       = j_NBS - media_ventana                                   # Calculo el tiempo inicial,
        tf_NBS: int       = j_NBS + media_ventana                                   # y final de la ventana para el nuevo j_NBS.
        if (j_NBS - media_ventana) < 0 or (j_NBS + media_ventana) > len(data_MAG):  # Si los tiempos se salen de los límites del archivo,
          continue                                                                  # omito esta ventana.
        v_NBS: np.ndarray = self.vector_característico(data_MAG.iloc[t0_NBS:tf_NBS])# Si no, calculo el vector de la ventana NBS,
        X.append(v_NBS)                                                             # lo agrego a la lista de vectores característicos X,
        y.append(0)                                                                 # y su etiqueta será 0 (no bow shock).
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
    ventanas correspondientes, devolviendo estos tres parámetros en formato tripla de np.ndarrays. Utiliza el porcentaje (%) de superposición
    entre ventanas que ha sido pasado por parámetro a la clase KNN, como paso para calcular las predicciones en cada ventana, y calcula los
    vectores característicos de cada ventana con las variables que se hayan indicado al KNN.
      Devuelve:
        prob[:,1] = Probabilidad de la clase BS.
        prob[:,0] = Probabilidad de la clase NBS.
    """
    if not self.entrenado:                                                 # Si todavía no se entrenó al KNN,
      raise RuntimeError('El clasificador KNN no ha sido entrenado.')      # devuelvo un mensaje.
    etiqueta:     list[int]         = []                                   # Inicializo una lista para guardar las etiquetas (1: BS ó 0: NBS),
    probabilidad: list[list[float]] = []                                   # una para guardar las probabilidades de bow shocks y no bow shocks,
    j_ventana:    list[int]         = []                                   # y una para guardar los índice de los centros de las ventanas.
    superposición: int = (self.ventana*self.superposición_ventana) // 100  # Calculo la superposición de ventana con el porcentaje dado.
    for i in range(0, len(data_MAG), max(1,superposición)):                # Para i de 0 al final del archivo MAG (con paso = superposición):
      j_0: int = i                                                         # obtengo el índice del inicio de la ventana actual,
      j_f: int = min(i + self.ventana, len(data_MAG))                      # y el índice del final de la ventana actual.
      ventana: pd.DataFrame = data_MAG[j_0 : j_f]                          # Obtengo solamente los datos MAG de esa ventana,
      v: np.ndarray = self.vector_característico(ventana)                  # y calculo su vector característico y lo guardo en la variable v.
      if v is not None:                                                    # Si el vector característico no es None,
        v_escalado: np.ndarray = self.scaler.transform(v.reshape(1,-1))    # lo re-escalo (funciona mejor pues el KNN trabaja con distancias).
        pred: int              = self.knn.predict(v_escalado)[0]           # Obtengo las predicciones de etiqueta bow shock ó no bow shock.
        prob: np.ndarray       = self.knn.predict_proba(v_escalado)[0]     # Obtengo las probabilidades,
        etiqueta.append(pred)                                              # y agrego ambos a la lista de etiquetas,
        probabilidad.append(prob)                                          # y a la lista de probabilidades.
        j_ventana.append(j_0 + ((j_f-j_0) // 2))                           # Obtengo el índice de la ventana como el medio de j_0 y j_f.
    return np.array(etiqueta), np.array(probabilidad), np.array(j_ventana) # Devuelvo listas de etiquetas, probabilidades y j en formato array.

  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  # Guardado y Exportado de Modelo Clasificador:
  #——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
  def save(self, directorio: str) -> None:
    """
    Guarda el Clasificador_KNN_Binario entrenado en disco.
    """
    with open(directorio, 'wb') as archivo:
      pickle.dump(self, archivo)
  @staticmethod
  def load(directorio: str) -> 'Clasificador_KNN_Binario':
    """
    Carga un Clasificador_KNN_Binario entrenado del disco.
    """
    with open(directorio, 'rb') as archivo:
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
    ventana: int = 300,                                                         # Ancho de ventana en segundos a utilizar (representa el BS).
    ventanas_NBS: list[int] = [-1,1,2],                                         # Posiciones de ventanas vecinas al BS para entrenar zona NBS.
    superposición_ventana: int = 50,                                            # Superposición entre ventanas (en %) para la predicción. 
    promedio: int = 1                                                           # Promedio para suavizar las muestras de MAVEN MAG.
) -> Clasificador_KNN_Binario:
  """
  La función entrenar recibe un directorio que contiene las mediciones de MAVEN MAG y los archivos de Fruchtman con los bow shock detectados,
  una lista de strings años_entrenamiento que representa los años que se desea entrenar al knn, el metaparámetro K del knn (número de vecinos),
  las variables que se desean utilizar como vector característico, el ancho de ventana (en segundos) para modelar el BS, las ventanas NBS que
  se desean considerar, la superposición entre ventanas para la predicción y el promedio de las muestras MAG para eliminar el ruido de los
  datos y reducir el tiempo de compilación. Devuelve el KNN, es decir, la clase Clasificador_KNN_Binario con el algoritmo KNN entrenado
  habiendo utilizado todas las variables que se han pasado por parámetro.
  """
  if años_entrenamiento == ['2014']:                                            # Si el año de entrenamiento es solo el 2014,
    raise ValueError('El año 2014 no tiene suficientes muestras para entrenar.')# => devuelvo un mensaje (son menos de 20 datos).
  knn: Clasificador_KNN_Binario = Clasificador_KNN_Binario(                     # En la variable 'knn' creo la clase Clasificador_KNN_Binario,
    K                     = K,                                                  # con todos los valores que han sido pasados por parámetro a
    variables             = variables,                                          # la función entrenar.
    ventana               = ventana,
    ventanas_NBS          = ventanas_NBS,
    superposición_ventana = superposición_ventana
  )
  knn.promedio = promedio                                                       # El promedio del knn, será el pasado por parámetro a entrenar.
  X: list[np.ndarray] = []                                                      # Inicializo una lista de vectores característicos 'X'.
  y: list[int]        = []                                                      # Inicializo una lista de etiquetas (enteros 0 ó 1) 'y'.
  for año in años_entrenamiento:                                                # Para cada año de la lista de años_entrenamiento,
    archivo_F: str = f'hemisferio_N/fruchtman_{año}_merge_hemisferio_N.sts'     # obtengo el nombre del archivo Fruchtman correspondiente,
    t0, tf         = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'             # obtengo el intervalo de tiempo de todo el año de archivo MAG,
    ruta_Fru: str  = os.path.join(directorio, 'fruchtman', archivo_F)           # obtengo las rutas completas del archivo Fruchtman,
    ruta_MAG: str  = os.path.join(directorio,'recorte_Vignes')                  # y de la carpeta donde están todos los archivos MAG.
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, promedio)      # Leo todos los archivos MAG del año con el promedio indicado.
    data_Fru: pd.DataFrame = pd.read_csv(ruta_Fru, sep=' ', header=None)        # Leo todo el archivo Fruchtman del año.
    dias_Fru: pd.Series     = data_Fru.iloc[:,0].astype(float)                  # Extraigo los días decimales de Fruchtman y convierto a float.
    t_BS = pd.Timestamp(f'{año}-01-01') + pd.to_timedelta(dias_Fru-1, unit='D') # Convierto los tiempos BS a objetos datetime adecuadamente.
    X_año, y_año = knn.muestras_entrenamiento(data_MAG, t_BS.to_numpy())        # Obtengo muestras de entrenamiento del año con data_MAG y BS.
    if len(X_año) == 0:                                                         # Si no hay muestras entrenadas,
      continue                                                                  # pasamos a la siguiente iteración del for (no las agrego).
    X.append(X_año)                                                             # Agrego el primer np.ndarray (vectores característicos) a X,
    y.append(y_año)                                                             # y el segundo np.ndarray (etiquetas) a la lista y.
  knn.clasificar_muestras(np.vstack(X), np.concatenate(y))                      # Clasifico las muestras (X,y) apiladas de todos los años. 
  print('El Clasificador_KNN_Binario se ha entrenado correctamente.')           # Devuelvo un mensaje de que el entrenamiento fue exitoso.
  #knn.save('bowshock_knn_model.pkl')                                           # Guardo el modelo.
  return knn                                                                    # Devuelvo el knn entrenado para utilizarlo para predecir.

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# clasificar: 
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
def clasificar(directorio: str, knn: Clasificador_KNN_Binario, predecir_años: list[str]) -> None:
  """
  Docstring
  """
  for año in predecir_años:                                                     #
    t0, tf         = f'1/1/{año}-00:00:00', f'31/12/{año}-23:59:59'             # obtengo el intervalo de tiempo de todo el año de archivo MAG,
    ruta_MAG: str  = os.path.join(directorio,'recorte_Vignes')                  # y de la carpeta donde están todos los archivos MAG.
    data_MAG: pd.DataFrame = leer_archivos_MAG(ruta_MAG, t0, tf, knn.promedio)  #
    print(f"\nAnalyzing {año}...")                                              #
    etiqueta, probabilidad, j_ventana = knn.predecir_ventana(data_MAG)          #
    j_BS = j_ventana[etiqueta == 1]                                             # Step 3: Pick only BS etiqueta
    t_BS = pd.to_datetime(data_MAG.iloc[:,0].to_numpy()[j_BS])                  # Recover actual datetime Convert np datetime64 to pd Timestamp array
    año_0 = pd.Timestamp(f'{t_BS[0].year}-01-01')                               # Now you can access the year
    días_dec = (t_BS-año_0).total_seconds()/86400 + 1                           # Decimal day-of-year
    print(f'Total windows: {len(etiqueta)}')                                    #
    print(f'Bow Shock detections: {len(t_BS)}')                                 #
    print(f'Percentage: {((len(t_BS)/len(etiqueta))*100):.2f}%')                #
    results = pd.DataFrame({'prediction': etiqueta, 'prob_BS': probabilidad[:,1], 'prob_NBS': probabilidad[:,0]})
    bs_results = pd.DataFrame({'datetime': t_BS, 'decimal_day': días_dec}) # Add predicted BS datetime and decimal day
    results.to_csv(f'bowshock_predictions_{año}.csv', index=False)
    bs_results.to_csv(f'bowshock_times_{año}.csv', index=False)
    #resultados[resultados["prediccion"] == 1][["dia_decimal"]].to_csv(f'bowshock_times_{año}.csv', index=False)









class Validación_Cruzada:
  def __init__():
    return



#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————