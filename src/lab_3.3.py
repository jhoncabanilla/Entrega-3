from functools import lru_cache
import random
import numpy as np
from numpy.core.fromnumeric import cumprod
from numpy.lib import stride_tricks
from numpy.lib.function_base import select
import seaborn as sb
import matplotlib.pyplot as plt
import copy
import sys
from SyncRNG import SyncRNG
sys.setrecursionlimit(10**6)


"Lab 3.3. Detectando ciclos en el laberinto"

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
            > profundidad: valor con el que se indica la profundidad de la busqueda
            > ciclo: lista en la que guardo los nodos con los que se formaría un ciclo desde el nodo en el que me encuentre
        """
        self.id = i
        self.visitado = False
        self.vecinos = []
        self.profundidad = 0
        self.ciclo = []

    def agregaVecino(self, n):
        """
        Funcion agregaVecino: funcion mediante la cual agregamos un nuevo nodo a la lista de vecinos del nodo correspondiente
        · Parametros
            > self
            > n: nodo a incluir en la lista de vecinos. Antes de todo comprobamos que el nodo no se encuentre ya en la lista
        """
        if n not in self.vecinos:
            self.vecinos.append(n)

    def agregaCiclo(self, nodo):
        """
        Funcion agregaCiclo: funcion mediante la cual agregamos un nuevo nodo a la lista de ciclo del nodo correspondiente
        · Parametros
            > self
            > n: nodo a incluir en la lista de ciclo. Antes de todo comprobamos que el nodo no se encuentre ya en la lista
        """
        if nodo not in self.ciclo:
            self.ciclo.append(nodo)


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

    def DFS(self, nodo, profundidad):
        """
        Funcion que implementa el algoritmo de busqueda: Primero en Profundidad
        · Parametros:
            > self
            > nodo: nodo desde el cual se comienza el recorrido en profundidad
            > profundidad: valor de la profundidad de la busqueda
        """
        #Marcamos el nodo como visitado y establecemos la profundidad del nodo a la indicada
        self.nodos[nodo].visitado = True
        self.nodos[nodo].profundidad = profundidad

        #Recorremos los vecinos del nodo y si no han sido visitados llamamos de manera recursiva a DFS con el nuevo nodo.
        for n in self.nodos[nodo].vecinos:
            if not self.nodos[n].visitado:
                self.DFS(n, profundidad+1) #En la llamada recursiva la profundidad será la recibida +1

    def RecorridoDFS(self):
        """
        Bucle que realiza una busqueda DFS desde todas y cada una de las habitaciones que no esten visitadas por una busqueda anterior
        """
        for n in self.nodos:
            if not self.nodos[n].visitado:
                self.DFS(n, profundidad=1)


    def BusquedaCiclo(self, nodo, padre, visitados):
        """"
        Funcion que determina si se ha encontrado un ciclo durante la busqueda y guarda en la lista ciclo el nodo con el que se forma dicho ciclo
        · Inputs:
            > self
            > nodo: nodo actual de la busqueda
            > padre: padre del nodo actual
            > visitados: lista de los nodos visitados durante la busqueda de ciclos
        """
        visitados[nodo] = True

        for vecino in self.nodos[nodo].vecinos:
            if vecino != padre:
                if visitados[vecino] or self.BusquedaCiclo(vecino, nodo, visitados):
                    self.nodos[vecino].agregaCiclo(nodo)
        return False

    def DetectaCiclo(self):
        """
        Funcion en la cual voy recorriendo los nodos visitados durante la busqueda para buscar los posibles ciclos llamando a la funcion BusquedaCiclo.
        """
        visitados = [False for v in self.nodos]

        for nodo in self.nodos:
            if self.nodos[nodo].visitado:
                if not visitados[nodo]:
                    self.BusquedaCiclo(nodo, None, visitados)


    def BFS(self, n, profundidad):
        """
        Funcion que implementa el algoritmo de busqueda: Primero en Anchura
        · Parametros:
            > self
            > n: nodo desde el cual se comienza el recorrido en anchura
            > profundidad: valor de la profundidad de la busqueda
        """

        if not self.nodos[n].visitado:
            cola = []
            cola.append(n)
            self.nodos[n].visitado = True
            self.nodos[n].profundidad = profundidad
 
            while cola: #Mientras la cola no este vacia
                n = cola.pop(0) #Quitamos el primer nodo de la cola de nodos
                for node in self.nodos[n].vecinos:
                    if not self.nodos[node].visitado:
                        cola.append(node)
                        self.nodos[node].visitado = True
                        profundidad += 1
                        self.nodos[node].profundidad = profundidad


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

    #ESTABLECEMOS ORDEN DE BUSQUEDA --> ABAJO-DERECHA-IZQUIERDA-ARRIBA
    for v in g.nodos:
        g.nodos[v].vecinos.reverse()

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

    #Inicializamos la matriz a -100 que representan las paredes
    matrix = np.full((filas*2+1, columnas*2+1), -100)
    
    for i in range(filas):
        for j in range(columnas):
            #Obtenemos el identificador del nodo actual y comprobamos ha sido visitado durante la DFS
            nodoActual = int(ide(array,i,j))
            if g.nodos[nodoActual].visitado == True:
                matrix[i*2+1][j*2+1] = g.nodos[nodoActual].profundidad #En ese caso, guardamos en la casilla el valor de la profundidad del nodo

            else:
                matrix[i*2+1][j*2+1] = 0 #Habitacion no visitada

            "Para comprobar si existe un eje, lo que hago es obtener una lista de los vecinos del nodo en el que nos encontremos"
            "y compruebo si el nodo a su derecha o debajo se encuentran en dicha lista"
            lista_vecinos = g.nodos[nodoActual].vecinos

            #Ponemos pasillos/conexiones hacia abajo y derecha
            #Añadimos la deteccion de ciclos y los dibujamos en caso de encontrarlos
            if i < filas-1 and int(ide(array, i+1, j)) in lista_vecinos:
                "Si se cumple esta condicion ponemos un pasillo que conecta con la habitacion de debajo"
                "Tambien evitamos poner un pasillo hacia debajo en una habitacion que se encuentre en la ultima fila"

                if g.nodos[ int(ide(array, i+1, j)) ].visitado == True:
                    if int(ide(array, i+1, j)) in g.nodos[nodoActual].ciclo:
                        matrix[i*2+2][j*2+1] = 250 #Valor especial para indicar que cierra un ciclo
                    else:
                        matrix[i*2+2][j*2+1] = g.nodos[ int(ide(array,i,j)) ].profundidad  #Asi ponemos en el pasillo el color de la habitacion de la que sale
                else:
                    matrix[i*2+2][j*2+1] = 0 #Pasillo no recorrido


            if j < columnas-1 and int(ide(array, i, j+1)) in lista_vecinos: 
                "Si se cumple esta condicion ponemos una pasillo que conecta con la habitacion de la derecha"
                "Tambien evitamos poner un pasillo hacia la derecha en una habitacion que se encuentre en la ultima columna"

                if g.nodos[ int(ide(array, i, j+1)) ].visitado == True:
                    if int(ide(array, i, j+1)) in g.nodos[nodoActual].ciclo:
                        matrix[i*2+1][j*2+2] = 250
                    else:
                        matrix[i*2+1][j*2+2] = g.nodos[ int(ide(array,i,j)) ].profundidad                
                else:
                    matrix[i*2+1][j*2+2] = 0 #Pasillo no recorrido

    return matrix

def anotaciones(filas, columnas, m):
    """
    Funcion que utilizo para dibujar las profundidades sobre la matriz de mapa de calor
    """
    puntos =  np.full((filas*2+1, columnas*2+1), -100)
    for i in range(filas*2+1):
        for j in range(columnas*2+1):
            if m[i][j] == 250:
                 puntos[i][j] = 0
                 plt.text(j,i,'CC', fontsize=8, color='black',  horizontalalignment='left', verticalalignment='top')
            else:
                puntos[i][j] = m[i][j]

    return puntos


def main():
    """
    *******************************************************************************
                                        MAIN
    *******************************************************************************
    """
    print("Implementacion lab 3.3")
    print()

    print("ORDEN DE BUSQUEDA: ABAJO-DERECHA-IZQUIERDA-ARRIBA")
    print()

    #Valores del array
    filas, columnas = 25, 25
    semilla = SyncRNG(seed=5)

    #Probabilidad entre 0 y 1
    pro = 0.5

    """
    Tarea 1. Detectando y dibujando ciclos
    """
    #Version 1: E como lista de adyacencia"
    g = Grafo()
    arrayIndNodos = generaLaberinto(filas, columnas, semilla, pro, g)

    #Realizamos una búsqueda en profundidad desde cada habitación del laberinto que no esté ya visitada por una búsqueda anterior 
    g.RecorridoDFS()
    #Deteccion de ciclos
    g.DetectaCiclo()

    """
    Implementacion Dibujar laberintos mediante un mapa de calor
    """
    #Obtenemos la matriz de mapa de calor
    m = traspasarGrafo(filas, columnas, g, arrayIndNodos)

    #Para conseguir que las paredes resalten con respecto a los otros valores introducidos por el recorrido hago lo siguiente:
    cmap=copy.copy(plt.get_cmap("hot"))
    cmap.set_under("gray")
    cmap.set_bad("blue")

    #Con esto conseguimos dibujar la profundidad de las habitaciones y pasillo
    an = np.vectorize(lambda x: '' if x <= 0 else str(round(x))) (anotaciones(filas, columnas, m))

    #Dibujar laberinto
    sb.heatmap(m,vmin=0,cmap=cmap,cbar_kws={'extend': 'min', 'extendrect': True}, annot=an, fmt="", mask=(m==0))
    plt.show()


if __name__ == "__main__":
    main()