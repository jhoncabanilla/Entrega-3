from functools import lru_cache
import random
import numpy as np
from numpy.core.fromnumeric import cumprod
from numpy.lib import stride_tricks
from numpy.lib.function_base import select
import seaborn as sb
import matplotlib.pyplot as plt
import time
import sys
from SyncRNG import SyncRNG
sys.setrecursionlimit(10**6)


"Lab 3.1. Recorriendo el laberinto"

"********************************************************VERSION_1: LISTA DE ADYACENCIA***********************************"
class Nodo:
    """
    Clase Nodo: Clase empleada para identificar cada nodo perteneciente al grafo.
    """
    def __init__(self, i):
        """
        Constructor:
            > id: identificar de cada nodo. Es de tipo int y es inicializado con el valor del parametro "i"
            > visitado: variable de tipo boolean mediante la cual indico si el nodo ha sido visitado o no
            > vecinos: lista que utilizo para almacenar los distintos vecinos que tiene un nodo en concreto
        """
        self.id = i
        self.visitado = False
        self.vecinos = []

    def agregaVecino(self, n):
        """
        Funcion agregaVecino: funcion mediante la cual agregamos un nuevo nodo a la lista de vecinos del nodo correspondiente
        · Parametros
            > self
            > n: nodo a incluir en la lista de vecinos. Antes de todo comprobamos que el nodo no se encuentre ya en la lista
        """
        if n not in self.vecinos:
            self.vecinos.append(n)


class Grafo: 
    """
    Clase Grafo: clase empleada para representar el grafo
    """
    def __init__(self):
        """
        Constructor:
            > nodos: diccionario empleado para almacenar el conjunto de nodos que componen el grafo
        """
        self.nodos = {}

    def agregaNodo(self, n):
        """
        Procede a añadir en el diccionario de nodos un nuevo nodo que no se encuentre previamente guardado.
        · Parametros:
            > self
            > n: Nodo que se desea añadir al grafo
        """
        if n not in self.nodos:
            self.nodos[n] = Nodo(n) #Almacena objeto de tipo Nodo con su id

    def agregaEjes(self, n1, n2):
        """
        Funcion agregaEjes: Funcion que crea los ejes entre los distintos nodos del grafo. Al tratarse de un grafo no direccionado, los ejes se crearan en un sentido
        y en el otro.
        Para llevar a cabo dicha tarea, lo que hago es llamar a la funcion "agregaVecino" de la clase Nodo y le paso como argumento el nodo con el que se quiere crear el eje.
        · Parametros:
            > self
            > n1: Nodo recibido al que se le agregara como vecino el otro nodo enviado
            > n2: Nodo recibido al que se le agregara como vecino el otro nodo enviado
        """
        #Grafo no direccionado
        if n1 not in self.nodos:
            self.agregaNodo(n1)

        if n2 not in self.nodos:
            self.agregaNodo(n2)

        self.nodos[n1].agregaVecino(n2)
        self.nodos[n2].agregaVecino(n1)

    def DFS(self, nodo):
        """
        Funcion que implementa el algoritmo de busqueda: Primero en Profundidad
        · Parametros:
            > nodo: nodo desde el cual se comienza el recorrido en profundidad
        """
        #Marcamos el nodo como visitado
        self.nodos[nodo].visitado = True

        #Recorremos los vecinos del nodo y si no han sido visitados llamamos de manera recursiva a DFS con el nuevo nodo.
        for n in self.nodos[nodo].vecinos:
            if not self.nodos[n].visitado:
                self.DFS(n)


def creaArray(filas, columnas, g):
    """
    Funcion que asocia a cada nodo/habitacion un valor entero.
    Empleo una valor "id" que servira como identificador de cada nodo.
    Tambien utilizo una matriz "array" de tamaño filas*columnas que inicializo a 0, en la que ire guardando los id's de los nodos.
    Utilizo 2 bucles for para iterar, y empleo la funcion "agregaNodo" de la clase Grafo para añadir un nuevo nodo en el grafo. Como comenté, tambien voy guardando en cada
    posicion de la matriz creada, el id de cada nodo.
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones
        > g: Objeto de tipo grafo

    · Output:
        > array: matriz que contiene los distintos id's de los nodos que componen el grafo
    """
    id = 0
    array = np.zeros((filas, columnas))
    for i in range(filas):
        for c in range(columnas):
            g.agregaNodo(id)
            array[i][c] = id
            id += 1
    return array


def ide(matriz, f, c):
    """
    Funcion que retorna el id correspondiente a la habitacion indicada a partir de la fila y la columna
    · Inputs:
        > matriz: matriz que contiene los id's de los nodos que componen el grafo
        > f: valor de una fila de la matriz indicada previamente
        > c: valor de una columna de la matriz indicada previamente

    · Output:
        > Id correspondiente al nodo situado en las posiciones indicadas dentro de la matriz de id's
    """
    return matriz[f][c]


def generaLaberinto(filas, columnas, semilla, pro, g):
    """
    Funcion que genera un laberinto a partir de una semilla
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones
        > semilla: semilla utilizada
        > pro: probabilidad empleada entre 0 y 1
        > g: Objeto de tipo grafo

    · Output:
        > array: matriz que contiene los distintos id's de los nodos que componen el grafo
    """

    #Creamos el array de nodos V
    array = creaArray(filas, columnas, g)

    #Inicializar rand-float con semilla
    random.seed(semilla)

    for i in range (filas):
        for j in range(columnas):
            if i > 0 and semilla.rand() < pro:
                """
                Si se cumplen las condiciones, establecemos un eje entre el nodo en el que nos encontramos y el nodo que esta encima
                y viceversa (conmutatividad)
                """
                #Agremamos los vecinos de cada nodo
                g.agregaEjes(int (ide(array,i,j)) , int (ide(array,i-1,j)) )
                g.agregaEjes(int (ide(array,i-1,j)) , int (ide(array,i,j)) )

            if j > 0 and semilla.rand() < pro:
                """
                Si se cumplen las condiciones, establecemos un eje entre el nodo en el que nos encontramos y el nodo que esta a su izquierda
                y viceversa (conmutatividad)
                """
                #Agremamos los vecinos de cada nodo
                g.agregaEjes(int (ide(array,i,j) ), int (ide(array,i,j-1)) )
                g.agregaEjes(int (ide(array,i,j-1) ), int (ide(array,i,j)) )

    return array


def traspasarGrafo(filas, columnas, g, array):
    """
    Funcion que traspasa el grafo a una matriz para su dibujo con un mapa de calor.
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones
        > g: Objeto de tipo grafo
        > array: matriz que contiene los distintos id's de los nodos que componen el grafo

    · Output:
        > matrix: matriz de mapa de calor
    """
    #Para que se puedan ver las habitaciones, paredes y pasillos, utilizaremos una matriz de dimensiones: filas*2+1, columnas*2+1
    #Inicializamos la matriz de mapa de calor a ceros que representan las paredes
    matrix = np.zeros((filas*2+1, columnas*2+1))
    
    for i in range(filas):
        for j in range(columnas):
            #Obtenemos el identificador del nodo actual y comprobamos ha sido visitado durante la DFS
            nodoActual = int(ide(array,i,j))
            if g.nodos[nodoActual].visitado == True:
                matrix[i*2+1][j*2+1] = 20 #En ese caso, guardamos en la casilla el valor 20 para indicar que el nodo ha sido visitado
            else:
                #Ponemos una habitacion no visitada
                #Para indicarlo ponemos el valor 10 en la matriz
                matrix[i*2+1][j*2+1] = 10

            "Para comprobar si existe un eje, lo que hago es obtener una lista de los vecinos del nodo en el que nos encontremos"
            "y compruebo si el nodo a su derecha o debajo se encuentran en dicha lista"
            lista_vecinos = g.nodos[nodoActual].vecinos

            #Ponemos pasillos/conexiones hacia abajo y derecha
            if i < filas-1 and int(ide(array, i+1, j)) in lista_vecinos:
                "Si se cumple esta condicion ponemos un pasillo que conecta con la habitacion de debajo"
                "Tambien evitamos poner un pasillo hacia debajo en una habitacion que se encuentre en la ultima fila"

                if g.nodos[ int(ide(array, i+1, j)) ].visitado == True:
                    matrix[i*2+2][j*2+1] = 20 #Pasillo visitado
                else:
                    matrix[i*2+2][j*2+1] = 15 #Pasillo no visitado


            if j < columnas-1 and int(ide(array, i, j+1)) in lista_vecinos: 
                "Si se cumple esta condicion ponemos una pasillo que conecta con la habitacion de la derecha"
                "Tambien evitamos poner un pasillo hacia la derecha en una habitacion que se encuentre en la ultima columna"

                if g.nodos[ int(ide(array, i, j+1)) ].visitado == True:
                    matrix[i*2+1][j*2+2] = 20
                else:
                    matrix[i*2+1][j*2+2] = 15

    return matrix


"********************************************************VERSION_2: MATRIZ DE ADYACENCIA***********************************"
def creaArray_matriz(filas, columnas):
    """
    Asocia a cada nodo/habitacion un valor entero
    Empleo una valor "id" que servira como identificador de cada nodo.
    Tambien utilizo una matriz "array" de tamaño filas*columnas que inicializo a 0, en la que ire guardando los id's de los nodos.
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones

    · Output:
        > array: matriz que contiene los distintos id's de los nodos que componen el grafo
    """

    id = 0
    array = np.zeros((filas, columnas))
   
    for i in range(filas):
        for c in range(columnas):
            array[i][c] = id
            id += 1      

    return array


def matrizAdyacencia(n):
    """
    Funcion que devuelve la matriz de adyacencia correspondiente al grafo
    · Inputs:
        > n: Dimension de la matriz de adyacencia. Se trata del numero de nodos que resulta tras multiplicar las filas por las columnas

    · Output:
        > matriz: matriz de adyacencia llenas de 0's
    """
    matriz = np.zeros((n,n))

    return matriz


def generaEjes(matriz, i, j):
    """
    Funcion que crea los ejes entre las habitaciones/nodos indicados
    · Inputs:
        > matriz: matriz de adyacencia original
        > i: posicion de la fila correspondiente a la habitacion en la matriz de adyacencia
        > j: posicion de la columna correspondiente a la habitacion en la matriz de adyacencia

    · Output:
        > matriz: matriz de adyacencia modificada con los nuevo ejes
    """

    matriz[ int(i) ][ int(j) ] = 1 #Para ello ponemos el valor 1, el cual indica un eje/conexion

    return matriz


def generaLaberinto_matriz(filas, columnas, semilla, pro):
    """
    Funcion que genera un laberinto a partir de una semilla"
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones
        > semilla: semilla utilizada
        > pro: probabilidad empleada entre 0 y 1

    · Output:
        > matrizAdy: matriz de adyacencia tras crear los posibles ejes
        > nodos: matriz que contiene los distintos id's de los nodos que componen el grafo
    """

    #Creamos el array de  nodos V
    nodos = creaArray_matriz(filas, columnas)

    #La matriz de adyacencia tiene dimensiones n x n: numero de nodos
    n = filas*columnas 
    matrizAdy = matrizAdyacencia(n)

    "Inicializar rand-float con semilla"
    random.seed(semilla)

    for i in range (filas):
        for j in range(columnas):
            """
            Si se cumplen las condiciones, establecemos un eje entre el nodo en el que nos encontramos y el nodo que esta encima
            y viceversa (conmutatividad)
            """
            if i > 0 and semilla.rand() < pro:
                matrizAdy = generaEjes(matrizAdy, ide(nodos,i,j), ide(nodos,i-1,j) ) 
                matrizAdy = generaEjes(matrizAdy, ide(nodos,i-1,j), ide(nodos,i,j) ) 

            """
            Si se cumplen las condiciones, establecemos un eje entre el nodo en el que nos encontramos y el nodo que esta a su izquierda
            y viceversa (conmutatividad)
            """
            if j > 0 and semilla.rand() < pro:
                matrizAdy = generaEjes(matrizAdy, ide(nodos,i,j), ide(nodos,i,j-1) ) 
                matrizAdy = generaEjes(matrizAdy, ide(nodos,i,j-1), ide(nodos,i,j) ) 

    return matrizAdy, nodos

    
def traspasarGrafo_matriz(filas, columnas, matrizAdy, nodos, visitados):
    """
    Funcion que traspasa el grafo a una matriz para su dibujo con un mapa de calor.
    · Inputs:
        > filas: filas utilizadas para representar el laberinto en 2-dimensiones
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones
        > matrizAdy: matriz de adyacencia tras crear los posibles ejes
        > nodos: matriz que contiene los distintos id's de los nodos que componen el grafo

    · Output:
        > matrix: matriz de mapa de calor
    """

    #Para que se puedan ver las habitaciones, paredes y pasillos, utilizaremos una matriz de dimensiones: filas*2+1, columnas*2+1
    #Inicializamos la matriz de mapa de calor a ceros que representan las paredes
    matrix = np.zeros((filas*2+1, columnas*2+1))

    for i in range(filas):
        for j in range(columnas):

            "Procedo a obtener el identificador del nodo en el que me encuentro y el de los que estan a la derecha y debajo"
            "Una vez los tenga, compruebo si en la matrizAdy las posiciones matrizAdy[nodoActual][NodoDer or NodoDeb] == 1.0, lo que indicaria que existe un eje"
            "entre los nodos"
            nodoActual = ide(nodos, i, j)

            #Compruebo si el nodo actual ha sido visitado
            if nodoActual in visitados:
                 matrix[i*2+1][j*2+1] = 20 #Habitacion visitada
            else:
                matrix[i*2+1][j*2+1] = 10 #Habitacion no visitada


            #Ponemos pasillos/conexiones hacia abajo y derecha
            if i < filas-1:
                #Evitamos poner un pasillo hacia debajo en una habitacion que se encuentre en la ultima fila
                nodoDebajo = ide(nodos, i+1, j)
                if matrizAdy[int(nodoActual)][int(nodoDebajo)] == 1.0:
                    "Si se cumple esta condicion existe una relacion entre el nodo actual y el que se encuentra debajo"

                    #Comprobamos si el nodo ha sido visitado
                    if nodoDebajo in visitados:
                        matrix[i*2+2][j*2+1] = 20 #Pasillo visitado
                    else:
                        matrix[i*2+2][j*2+1] = 15 #Pasillo no visitado

            if j < columnas-1:
                #Evitamos poner un pasillo hacia la derecha en una habitacion que se encuentre en la ultima columna
                nodoDerecha = ide(nodos, i, j+1)
                if matrizAdy[int(nodoActual)][int(nodoDerecha)] == 1.0:
                    "Si se cumple esta condicion existe una relacion entre el nodo actual y el que se encuentra a su derecha"

                    #Comprobamos si el nodo ha sido visitado
                    if nodoDerecha in visitados:
                        matrix[i*2+1][j*2+2] = 20
                    else:
                        matrix[i*2+1][j*2+2] = 15

    return matrix


def DFS_matriz(nodo, visitado, matrizAdy, columnas):
    """
    Funcion que implementa la busqueda primero en profundidad.
    · Inputs:
        > nodo: nodo que se esta visitando en la busqueda
        > visitado: lista de los nodos visitados durante la busqueda
        > matrizAdy: matriz de adyacencia
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones

    · Output:
        > visitado: lista final con todos los nodos visitados al final de la busqueda
    """

    #Añadimos a la lista de visitados el nodo actual
    visitado.append(nodo)

    #Creo un lista de vecinos vacia. En ella ire incluyendo los vecinos que tenga cada nodo
    vecinos = []
    for j in range(columnas):
        #Procedo a recorrer la matriz de adyacencia de manera que exploro unicamente las columnas para una fila dada, que sera la del nodo correspondiente
        if matrizAdy[int(nodo)][j] == 1.0:
            vecinos.append(j) #Vecino de nuestro nodo actual

    for n in vecinos:
        #Comprobamos si los vecinos del nodo han sido visitados, sino llamamos recursivamente a DFS_matriz con los respectivos vecinos
        if n not in visitado:
            DFS_matriz(n, visitado, matrizAdy, columnas)
    
    return visitado


def fs(nodo, matrizAdy, columnas):
    """
    Funcion que crea una lista vacia que utilizare para añadir los nodos que han sido visitados durante la busqueda DFS
    · Inputs:
        > nodo: nodo por el que se empieza a realizar la busqueda primero en profundidad
        > matrizAdy: matriz de adyacencia
        > columnas: columnas utilizadas para representar el laberinto en 2-dimensiones

    · Output:
        > visitado: lista final con todos los nodos visitados al final de la busqueda 
    """
    visitado = []
    visitado = DFS_matriz(nodo, visitado, matrizAdy, columnas)

    return visitado


def main():
    """
    *******************************************************************************
                                        MAIN
    *******************************************************************************
    """
    print("Implementacion lab 3.1")
    print()

    #Valores del array
    filas, columnas = 25,25
    s = 5
    semilla = SyncRNG(seed=s)
    #Probabilidad entre 0 y 1
    pro = 0.5

    """
    Tarea 1: Laberinto y Grafo
    """

    #Escogemos version
    version = int(input("Escoger version: [1]Con lista de adyacencia o [2]Con matriz de adyacencia: "))
    if version == 1:
        #Version 1: E como lista de adyacencia"
        g = Grafo()
        t0 = time.time()
        arrayIndNodos = generaLaberinto(filas, columnas, semilla, pro, g)

        #Impresion lista de adyacencia
        for v in g.nodos:
            print(v, g.nodos[v].vecinos)

        """
        Tarea 2: Recorrido en Profundidad
        """
        #Realizamos un recorrido primero en profundidad
        g.DFS(0) #El nodo de la esquina superior izquierda se corresponde con el id=0
        t1 = time.time()
        print("Tiempo de ejecucion con lista de adyacencia para filas=",filas,",columnas",columnas,",semilla:",s,"y probabilidad",pro,"es: -->",t1-t0)

        """
        Implementacion Dibujar laberintos mediante un mapa de calor
        """
        #Obtenemos la matriz de mapa de calor
        m = traspasarGrafo(filas, columnas, g, arrayIndNodos)
        

    else:
        #Version 2: E como matriz de adyacencia"
        t0 = time.time()
        matrizAdy, nodos = generaLaberinto_matriz(filas, columnas, semilla, pro)

        #Impresion matriz de adyacencia
        print(matrizAdy)

        """
        Tarea 2: Recorrido en Profundidad
        """
        #Realizamos un recorrido primero en profundidad
        nodo = nodos[0][0] #Obtenemos el identificador del nodo de la esquina superior izquierda
        visitados = fs(nodo, matrizAdy, columnas*columnas)
        t1 = time.time()
        print("Tiempo de ejecucion con matriz de adyacencia para filas=",filas,",columnas",columnas,",semilla:",s,"y probabilidad",pro,"es: -->",t1-t0)

        """
        Implementacion Dibujar laberintos mediante un mapa de calor
        """
        #Obtenemos la matriz de mapa de calor
        m = traspasarGrafo_matriz(filas, columnas, matrizAdy, nodos, visitados)



    #Dibujar laberinto. Uso de la libreria de visualizacion de datos seaborn
    sb.heatmap(m,cmap='hot')
    plt.show()


if __name__ == "__main__":
    main()