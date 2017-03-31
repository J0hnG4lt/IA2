Análisis de Clusters Aplicado a Repositorios Públicos de Github
----------------------------------

### Integrantes:
* Leslie Rodrigues
* Erick Silva
* Georvic Tur

---------------------------------------------------------------

### Resumen

Buscamos determinar si las relaciones que existen entre los 
lenguajes de programación usados en repositorios de GitHub
permiten crear algún tipo de perfilamiento de sus programdores.

Para lograr esto, creamos un *web scraper* y lo usamos para
recolectar los datos que guardamos en la carpeta *datos_originales*.

Luego usamos [scikit-learn](http://scikit-learn.org/) y [Weka](http://www.cs.waikato.ac.nz/ml/weka/) con varios algoritmos
de agrupamiento. También calculamos las correlaciones entre 
los lenguajes siguiendo las indicaciones de [arXiv:1603.00431 [cs.PL]](https://arxiv.org/abs/1603.00431).

---------------------------------------------------------------

### Corridas

El directorio de corridas está compuesto por subdirectorios con
el nombre del algoritmo o técnica usada. Cada uno contiene los
resultados de su aplicación.

En el caso de los algoritmos aplicados en Weka, se aumentó
el archivo de vectores de atributos con la dimensión 
adicional del *cluster* al que pertenece cada instancia.

En el caso de los algoritmos aplicados con *scikit-learn*,
se muestran archivos cuyos nombres están prefijados por 
la palabra *cluster*.

Para generar los vectores de atributos a partir del archivo
JSON con los datos de los repositorios hay que ejecutar
los scripts colocados en el mismo directorio cuyos
nombres están sufijados por *generateClusters.py*.

Si se ejecuta generateCluster.py se generan los vectores
de atributos del experimento primero, en el cual se colapsan
los datos en lenguajes. Es decir, los lenguajes son las 
instancias y los atributos indican la cantidad de veces 
que dicho lenguaje aparece con otro.

Si se ejecuta perRepo_generateClusters.py, se generan las 
instancias de cada repositorio. Los atributos de estas 
instancias indican la cantidad de bytes de cada lengueje 
en ellos.

Si se ejecuta perUserGenerateDendogram.py, se genera un 
dendrograma a partir de una matriz de similitud calculada
por nosotros y un algoritmo de agrupamiento jerárquico.

Si se ejecuta perUserLanguageCorrelations.py, se genera 
una matriz de correlaciones entre los lenguajes. En este 
caso, la data original se colapsó en los usuarios.

---------------------------------------------------------------

### Scraper

Para correr el *scraper* es necesario usar *Python3* e indicar 
en el cuerpo del script el usuario raíz desde donde van a ser 
recolectados los usuarios atendiendo al árbol de *followers*. 
También hay que indicar la profundidad de este grafo desde la raíz. 
Si ya se tienen suficientes usuarios, se procede a recolectar 
la información de sus repositorios. Este script fue implementado de 
tal manera de que no se pierda la información recolectada si 
ocurre un *rate limit*. Una vez que se haya reseteado la restricción, 
se puede volver a invocar el *scraper* con Python3.