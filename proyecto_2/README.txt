Red Multicapa Feedforward con Backpropagation


Integrantes:
    Erick Silva         11-10969
    Leslie Rodrigues    10-10613
    Georvic Tur         12-11402


Requerimientos
    
    Lenguaje: Python3
    Librerías: numpy y matplotlib para python3
    

Organización de los Módulos

    generarPatrones.py : define varias funciones auxiliares, además del generador
                         de puntos. Cuando es corrido como programa, entrena una
                         red neuronal que intenta clasificar puntos dentro de 
                         un círculo. Los parámetros de la red se pueden cambiar
                         al llamar a su función. También se puede cambiar el 
                         nombre del conjunto de datos de entrada así como el
                         tamaño de los puntos por área para la prueba.
    
    iris.py :            Entrena una red neuronal para discriminar a la Iris Setosa
                         del resto de flores.
    
    iris_todas.py:       Entrena una red neuronal para discriminar tres tipos de
                         flores.
    
    mlp.py:              Define una red neuronal general para clasificación binaria
    
    mlpMulticlase.py:    Define una red neuronal general para clasificación multiclase
    
    datosP2EM2017/datos_P2_EM2017_N500.txt
    datosP2EM2017/datos_P2_EM2017_N1000.txt
    datosP2EM2017/datos_P2_EM2017_N2000.txt
    
    datosP2EM2017/iris_setosa.data: datos preprocesados de la iris setosa para
                                    discriminar sólo a esta especie
    datosP2EM2017/iris_todas.data: datos preprocesados de la iris setosa para
                                   discriminar a las tres especies


Instrucciones
    
    Para obtener los resultados de la clasificación de patrones hay que
    ejecutar la siguiente orden:
        
        python3 generarPatrones.py
        
        Dentro de dicho script se puede cambiar el conjunto de entrenamiento
        a través del archivo de entrada. También se puede cambiar el conjunto de
        prueba alterando los argumento de la función generadora.
        
    
    Para obtener los resultados de la clasificación de iris setosa hay que
    ejecutar la siguiente orden:
    
        python3 iris.py
        
        Dentro de dicho script se puede cambiar la proporción del conjunto de
        entrenamiento.
        
    Para obtener los resultados de la clasificación de iris multiclase hay que
    ejecutar la siguiente orden:
    
        python3 iris_todas.py
        
        Dentro de dicho script se puede cambiar la proporción del conjunto de
        entrenamiento.
