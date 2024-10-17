# ====================================================
# PARTE I: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ====================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:


# ----------------------------------
# I.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....
         
#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Los atributos que tiene nuestra clase naive bayes los definimos en el constructor y son el constante de suavizado que especifiquemos en el mismo, por defecto 1, las clases posibles a clasificar(los valores únicos de la variable objetivo), las probabilidades a priori de esas clases posible y las probabilidades condicionada de cada valor de atributo en función de la clase objetivo. También una variable que indica si se ha realizado el enrtenamiento o no, ya que si es False, algunos métodos de la clase generan una excepcion.

# El método entrena realiza el entrenamiento calculando las probabilidades a priori y condicionadas, teniendo en función la matriz de variables predictoras y el array de la variable objetivo. Con esto posteriormente podemos realizar predicciones y calcular la probabilidad de que la muestra pertenezca a cada una de las posibles clases.

# El metodo calculate_class_prior, calcula las probs a priori de cada clase, contando cuantas veces aparece cada clase y dividiendolo entre todas las muestras que forman el conjunto entrenamiento.

# El metodo calculate_feature_likelihoods, calcula las probabilidades condicionadas de cada atributo según la clase, contando el número que aparece ese atributo para las muestras de esa clase.

# El métodos clasifica_log_prob calcula las log-probabilidades de pertenecer una muestra a cada clase mediante el calculo del logaritmo de la priori de la clasess sumada a las probs condicionadas de los atributos de la muestra a esa clase

# El métodos clasifica_prob transforma las log_probabilidades en probabilidades que sumen 1, haciendo la exponencial de las log probabilidades y normalizandolas.

# El método clasifica calcula la clase con mayor log-probabilidades y devuelve esa clase.

import numpy as np

class NaiveBayes:
    def __init__(self, k=1):
        #definimos los atributos d la clase
        self.k = k 
        self.classes = None 
        self.class_priors = None 
        self.feature_likelihoods = None 
        self.trained = False 

    def entrena(self, X, y): 
        self.classes = np.unique(y) #
        self.class_priors = self.calculate_class_priors(y) 
        self.feature_likelihoods = self.calculate_feature_likelihoods(X, y) 
        self.trained = True 

    def clasifica_log_prob(self, example):
        if not self.trained:
#             raise ValueError("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica_log_prob'.")
             raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica_log_prob'.")         
        log_probabilities = {}
        for cls in self.classes:
            log_probabilities[cls] = np.log(self.class_priors[cls])
            numero = 0
            for i in example:
                log_probabilities[cls]+=np.log(self.feature_likelihoods[(cls,f'var{numero}',i)])
                numero += 1
        return log_probabilities
    
    def clasifica_prob(self, example):
        if not self.trained:
#             raise ValueError("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica_prob'.") #Opcion 1
             raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica_prob'.")
            
        log_probabilities = self.clasifica_log_prob(example)
        probabilities_no_norm = np.exp(list(map(lambda x: x, log_probabilities.values())))
        probabilities = probabilities_no_norm / np.sum(probabilities_no_norm)
        probabilidades = {}
        for categoria, probabilidad in zip(log_probabilities.keys(), probabilities):
            probabilidades[categoria]=probabilidad
        return probabilidades

    def clasifica(self, example):
        if not self.trained:
#             raise ValueError("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica'.") Opción 1
            raise ClasificadorNoEntrenado("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica'.")
            
        log_probabilities = self.clasifica_log_prob(example)
        predicted_class = max(log_probabilities, key=log_probabilities.get)
        return predicted_class

    def calculate_class_priors(self, y):
        priors={}
        for i in np.unique(y):
            cont = sum(y == i)
            priors[i]=cont/len(y)
        return priors

    def calculate_feature_likelihoods(self, X, y):
        feature_likelihoods = {}
        for cls in self.classes:
            cls_indices = np.where(y == cls)[0]
            cls_samples = X[cls_indices]
            for feature in range(X.shape[1]):
                features = np.unique(X[:,feature])
                cardinalFeature = len(features)
                for f in features:
                    cont=sum(X[cls_indices,feature]==f)
                    cond=(cont+self.k) / (len(cls_indices) + self.k*cardinalFeature)
                    feature_likelihoods[(cls,f'var{feature}',f)]=cond
        return feature_likelihoods

class ClasificadorNoEntrenado(Exception): 
    def __init__(self, message):
        super().__init__(message)
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): 
    def __init__(self, message):
        super().__init__(message)
  
# Ejemplo "jugar al tenis":


# >>> nb_tenis=NaiveBayes(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'

import jugar_tenis #para cargar los datos
nb_tenis=NaiveBayes(k=0.5)
nb_tenis.entrena(jugar_tenis.X_tenis,jugar_tenis.y_tenis)
ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])

print("Las log-probabilidades: " ,nb_tenis.clasifica_log_prob(ej_tenis))

print("Las probabilidades normalizadas: " ,nb_tenis.clasifica_prob(ej_tenis))
print("Este ejemplo pertenece a la categoria: ", nb_tenis.clasifica(ej_tenis))

# ----------------------------------------------
# I.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

def rendimiento(clasificador, X, y):
    predicciones = [clasificador.clasifica(x) for x in X] #llamamos al metodo clasifica de cada ejemplo de X y lo metemos en un array
    aciertos = sum(prediccion == valor_esperado for prediccion, valor_esperado in zip(predicciones, y)) #vemos cuanto son correctos
    accuracy = aciertos / len(y) #calculamos tasa acierto
    return accuracy

rendimiento(nb_tenis,jugar_tenis.X_tenis,jugar_tenis.y_tenis)

# --------------------------
# I.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US

import votos
from sklearn.model_selection import train_test_split
(X_votos_train,X_votos_test,y_votos_train,y_votos_test) = train_test_split(votos.datos,votos.clasif,test_size=0.40,shuffle=True)

MejorKvotos=0
MejorTAcierto=0
for i in np.arange(0,1.1,0.1):
    nb_votos=NaiveBayes(k=i)
    nb_votos.entrena(X_votos_train,y_votos_train)
    tacierto=rendimiento(nb_votos,X_votos_test,y_votos_test)
    if tacierto > MejorTAcierto:
        MejorTAcierto = tacierto
        MejorKvotos = i
    print(f"Rendimiento con k = {i} es(tasa acierto): {tacierto}")

    
print(f"El mejor modelo de naive bayes para los datos de test es el que tiene un suavizado de : {MejorKvotos}" )


# =====================================================
# PARTE II: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# =====================================================

# En esta SEGUNDA parte se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística. 



# ---------------------------------------------
# II.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#     def __init__(self,clases=[0,1],normalizacion=False,
#                  rate=0.1,rate_decay=False,batch_tam=64)
#         .....
        
#     def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):

#         .....        

#     def clasifica_prob(self,ejemplo):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

        

# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificación, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista sería
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamaño de los mini batches (por defecto 64) que se
#    usan para calcular cada actualización de pesos.



# * El método entrena tiene los siguientes parámetros de entrada:

#  + X e y son los datos del conjunto de entrenamiento y su clasificación
#    esperada, respectivamente. El primero es un array con los ejemplos, y el
#    segundo un array con las clasificaciones de esos ejemplos, en el mismo
#    orden.

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + reiniciar_pesos: si es True, cada vez que se llama a entrena, se
#    reinicia al comienzo del entrenamiento el vector de pesos de
#    manera aleatoria (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior, excepto que se diera
#    explícitamente el vector de pesos en el parámetro peso_iniciales.  

#  + pesos_iniciales: si no es None y el parámetro anterior reiniciar_pesos 
#    es False, es un array con los pesos iniciales. Este parámetro puede ser
#    útil para empezar con unos pesos que se habían obtenido y almacenado como
#    consecuencia de un entrenamiento anterior.



# * Los métodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los métodos de clasificación antes de
#   entrenar. 

# Se recomienda definir la función sigmoide usando la función expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)

# Los atributos que tiene nuestra clase regresión logistica los definimos en el constructor y son las clases de etiqueta de la variable objetivo, si se debe realizar normalización, la tasa de aprendizaje(que mide el cambio que se va a realizar en cada iteración, el rate_decay que si es True hace quela tasa vaya bajando poco a poco, y el tamaño del lote.

# El metodo calcula_param_normalizar calcula la media y la desviaciónd de cada caracteristica del conjunto sobre el que se entrenan los datos para poder realizar posteriormente la normalización mediante el metodo normalizar_datos.

# El metodo entrena recibe los datos de entrenamiento y si se deben reiniciar los pesos, así como el número de epocas. Hace los calculos, calcula el gradiente en cada iteración con las medidas pertenecientes a cada bacth y actualiza los pesos para mejor el rendimiento.

# El metodo sigmoide es la función de activación y calcula la probabilidad de pertenecer una muestra a la clase positiva(1) según los calculos del sesgo + peso*variable predictora.

# El métodos clasifica_prob calcula la probabilidad de pertenercer a esa clase.La salida de usar la sigmoide da la prob de pertenecer clase positiva y por lo tanto 1 - probPos = probNeg.

# El método clasifica calcula la clase con mayor probabilidad y devuelve esa clase.


import numpy as np
from scipy.special import expit

class RegresionLogisticaMiniBatch():
    def __init__(self, clases=[0, 1], normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.pesos = None
        self.trained=False
        self.media = None
        self.desviacion = None
    
    def calcula_param_normalizar(self,X):
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)
        
    def sigmoide(self, x):
        return expit(x)
    
    def normalizar_datos(self, X):
        X_norm = (X - self.media) / self.desviacion
        return X_norm
    
    def entrena(self, X, y, n_epochs, reiniciar_pesos=False, pesos_iniciales=None):
        if self.normalizacion:
            self.calcula_param_normalizar(X)
            X = self.normalizar_datos(X)
        
        n_ejemplos, n_caracteristicas = X.shape
        
        if reiniciar_pesos or self.pesos is None:
            self.pesos = np.random.uniform(-1, 1, size=(n_caracteristicas + 1,))
        elif pesos_iniciales is not None:
            self.pesos = pesos_iniciales
        
        for epoch in range(n_epochs):
            if self.rate_decay:
                self.rate = self.rate / (1 + epoch)
                
            
            indices = np.random.permutation(n_ejemplos)
            X_shuffle = X[indices]
            y_shuffle = y[indices]
            
            for i in range(0, n_ejemplos, self.batch_tam):
                X_batch = X_shuffle[i:i + self.batch_tam]
                y_batch = y_shuffle[i:i + self.batch_tam]
                
                X_batch = np.column_stack((np.ones((X_batch.shape[0], 1)), X_batch))
                logits = np.dot(X_batch, self.pesos)
                y_pred = self.sigmoide(logits)
                
                gradiente = np.dot((y_batch-y_pred),X_batch)
                
                self.pesos += self.rate * gradiente
        self.trained = True
    
    def clasifica_prob(self, ejemplo):
        if not self.trained:
            raise ValueError("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica_prob'.")
        
        if self.normalizacion:
            ejemplo = self.normalizar_datos(ejemplo)
        
        ejemplo = np.insert(ejemplo, 0, 1)
        logits = np.dot(ejemplo, self.pesos)
        probabilidad = self.sigmoide(logits)
        
        probabilidad_clase0 = 1 - probabilidad
        probabilidad_clase1 = probabilidad
        
        return {self.clases[0]: probabilidad_clase0, self.clases[1]: probabilidad_clase1}
    
    def clasifica(self, ejemplo):
        if not self.trained:
            raise ValueError("El modelo no ha sido entrenado. Llama al método 'entrena' antes de utilizar 'clasifica'.")
        probabilidades = self.clasifica_prob(ejemplo)
        clase_predicha = self.clases[np.argmax(list(probabilidades.values()))]
        return clase_predicha
# ----------------------------------------------------------------


# Ejemplo de uso, con los datos del cáncer de mama, que se puede cargar desde
# Scikit Learn:

# >>> from sklearn.datasets import load_breast_cancer
# >>> cancer=load_breast_cancer()

# >>> X_cancer,y_cancer=cancer.data,cancer.target


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer,10000)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9906103286384976
# >>> rendimiento(lr_cancer,Xt_cancer,yt_cancer)
# 0.972027972027972

def rendimientoRL(clasificador, X, y):
    predicciones = [clasificador.clasifica(x) for x in X]
    aciertos = sum(prediccion == valor_esperado for prediccion, valor_esperado in zip(predicciones, y))
    accuracy = aciertos / len(y)
    return accuracy

# -----------------------------------------------------------------







# -----------------------------------
# II.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US

import votos
from sklearn.model_selection import train_test_split
(X_votos_train,X_votos_test,y_votos_train,y_votos_test) = train_test_split(votos.datos,votos.clasif,test_size=0.40,shuffle=True)


rl_votos=RegresionLogisticaMiniBatch() # no es necesario normalización ya que son variables númericas discretas(categóricas)
rl_votos.entrena(X_votos_train,y_votos_train,1000)
print(rendimientoRL(rl_votos,X_votos_test,y_votos_test))

# - Cáncer de mama 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

X_cancer,y_cancer=cancer.data,cancer.target

(X_cancer_train,X_cancer_test,y_cancer_train,y_cancer_test) = train_test_split(X_cancer,y_cancer,test_size=0.30,shuffle=True)


rl_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)  #necesaria normalización ya que son variables continuas

rl_cancer.entrena(X_cancer_train,y_cancer_train,1000)

print(rendimientoRL(rl_cancer,X_cancer_test,y_cancer_test))

# ===================================
# PARTE III: CLASIFICACIÓN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.


# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs):

#        .......

#     def clasifica(self,ejemplo):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

# Los atributos que tiene nuestra clase One vs Rest de regresión logística los definimos en el constructor y son las clases de etiqueta de la variable objetivo, si se debe realizar normalización, la tasa de aprendizaje(que mide el cambio que se va a realizar en cada iteración, el rate_decay que si es True hace quela tasa vaya bajando poco a poco, y el tamaño del lote.

# El metodo entrena recibe los datos de entrenamiento y si se deben reiniciar los pesos, así como el número de epocas.Primero selecciona una de las clases posibles y computa que si la etiqueta de ese valor coincide sea 1(valor positivo) y cualquier otro valor es 0(valor negativo), convirtiendo el problema en uno de clasificación binaria como teniamos en la de regresión logistica definida anteriormente. Hace los calculos, calcula el gradiente en cada iteración con las medidas pertenecientes a cada bacth y actualiza los pesos para mejor el rendimiento. Así para todas las clases entrena el clasificador para cada clase poniendola como positiva.

# El métodos clasifica_prob calcula la probabilidad de pertenercer a cada clase según su clasificador binario entrenado y le asigna la probabilidad positiva a esa clase. Así para cada posible clase de la variable objetivo, teniendo todas las probabilidades para cada clase vs el resto. 

# El método clasifica calcula la clase con mayor probabilidad y devuelve esa clase.

class RL_OvR():
    def __init__(self, clases, normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        self.clases = clases
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clasificadores = {}
    
    def entrena(self, X, y, n_epochs, reiniciar_pesos=False, pesos_iniciales=None):
        for clase in self.clases:
            y_clase = np.where(y == clase, 1, 0)
            clasificador = RegresionLogisticaMiniBatch([0, 1], self.normalizacion, self.rate, self.rate_decay, self.batch_tam)
            clasificador.entrena(X, y_clase, n_epochs, reiniciar_pesos, pesos_iniciales)
            self.clasificadores[clase] = clasificador
    
    def clasifica_prob(self, ejemplo):
        probabilidades = {}
        
        for clase, clasificador in self.clasificadores.items():
            probabilidad = clasificador.clasifica_prob(ejemplo)[1]  # Probabilidad de la clase positiva
            probabilidades[clase] = probabilidad
        
        return probabilidades
    
    def clasifica(self, ejemplo):
        probabilidades = self.clasifica_prob(ejemplo)
        clase_predicha = max(probabilidades, key=probabilidades.get)
        return clase_predicha

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> from sklearn.datasets import load_iris
# >>> iris=load_iris()
# >>> X_iris=iris.data
# >>> y_iris=iris.target
# >>> Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

def rendimientoRL_OVR(clasificador, X, y):
    predicciones = [clasificador.clasifica(x) for x in X]
    aciertos = sum(prediccion == valor_esperado for prediccion, valor_esperado in zip(predicciones, y))
    accuracy = aciertos / len(y)
    return accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris=load_iris()
X_iris=iris.data
y_iris=iris.target
Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)


rendimientoRL_OVR(rl_iris,Xt_iris,yt_iris)

# ------------------------------------------------------------
# III.2) Clasificación de imágenes de dígitos escritos a mano
# ------------------------------------------------------------


#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. Si el
#  tiempo de cómputo en el entrenamiento no permite terminar en un tiempo
#  razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

def cargaImágenes(fichero,ancho,alto): #tendra cada imagen 784 caracteristicas ya que la imagen de 28 x 28 pixeles esta aplanada
    def convierte_0_1(c):
        if c==" ":
            return 0
        else:
            return 1

    with open(fichero) as f:
        lista_imagenes=[]
        ejemplo=[]
        cont_lin=0
        for lin in f:
            ejemplo.extend(list(map(convierte_0_1,lin[:ancho])))
            cont_lin+=1
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)  
                ejemplo=[]
                cont_lin=0
    return np.array(lista_imagenes)

def cargaClases(fichero):
    with open(fichero) as f:
        return np.array([int(c) for c in f])

#cargamos los datos    
trainingdigits="digitdata/trainingimages"
validationdigits="digitdata/validationimages"
testdigits="digitdata/testimages"
trainingdigitslabels="digitdata/traininglabels"
validationdigitslabels="digitdata/validationlabels"
testdigitslabels="digitdata/testlabels"


X_train_dg=cargaImágenes(trainingdigits,28,28)
y_train_dg=cargaClases(trainingdigitslabels)
X_valid_dg=cargaImágenes(validationdigits,28,28)
y_valid_dg=cargaClases(validationdigitslabels)
X_test_dg=cargaImágenes(testdigits,28,28)
y_test_dg=cargaClases(testdigitslabels)


#calculamos el rendimiento sobre prueba y validación entrenando sobre datos entrenamiento para algunos posibles hiperparámetros y vemos cuales son los mejores.
batch_tam=[32,64,128]
rate=[0.01,0.001,0.0005]
rate_decay=[False,True]

MejorBatch=0
MejorTAcierto=0
RateDecay=False
MejorRate=0
for i in batch_tam:
    for j in rate:
        for z in rate_decay:
            rl_digit=RL_OvR(np.unique(y_train_dg).tolist(),rate=j,batch_tam=i,rate_decay=z)
            rl_digit.entrena(X_train_dg,y_train_dg,n_epochs=100)
            taciertoValid = rendimientoRL_OVR(rl_digit,X_valid_dg,y_valid_dg)
            taciertoTest = rendimientoRL_OVR(rl_digit,X_test_dg,y_test_dg)
            if taciertoValid > MejorTAcierto:
                MejorTAcierto = taciertoValid
                MejorBatch = i
                RateDecay=z
                MejorRate=j
            print(f"Rendimiento validacion con rate = {j}, batch_tam = {i} y rate_decay = {z} es(tasa acierto): {taciertoValid}")
            print(f"Rendimiento test con rate = {j}, batch_tam = {i} y rate_decay = {z} es(tasa acierto): {taciertoTest}")


    
print(f"El mejor modelo de naive bayes para los datos de test es con batch_tam = {MejorBatch}, rate_decay = {RateDecay} y rate = {MejorRate}" )



