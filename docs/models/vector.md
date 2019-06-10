# Vector entero
#### `models.Vector`

Clase que representa a un vector entero y todas las operaciones relacionadas con él, como la suma de dos vectores, de un
vector y un número, multiplicación, division, obtención de valores, establecer valores, longitud, comparaciones, etc.

* **Atributos**
    * `decimals` - Variable que indica el número por el que tenemos que multiplicar un vector flotante para convertirlo
    en entero y eliminar los decimales restantes. Dicho de otra forma, si tenemos el número `12.345` y `decimals` vale
    `100`, obtendremos el entero `1234`.
    * `relative_tolerance` - Tolerancia relativa que se usará para comparar la igualdad entre vectores.
    * `absolute_tolerance` - Tolerancia absoluta que se usará para comparar la igualdad entre vectores.
    * `components` - Atributo de la clase `np.array` donde almacenaremos los números para que las operaciones sean más 
    eficientes.
    
    * **Propiedades**
        * **Estas propiedades son perezosas, para evitar el cálculo si no es necesario** 
        * `magnitude` - Devuelve la magnitud del vector.
        * `zero_vector` - Devuelve un vector de la misma longitud y tipo que el vector actual, pero con todas las
        componentes a cero.
    
* **Métodos**
    * **Métodos mágicos**
        * _**Como son métodos de mágicos de Python vamos a explicar la función que le hemos aplicado**_.
        * `__getitem__(self, item)`
            * Obtiene de la variable `components` el objeto de la posición `item`.
        * `__setitem__(self, key, value)`
            * Establece el valor `value` a la clave `key` en el array `components`.
        * `__len__(self)`
            * Devuelve la longitud del array `components`.
        * `__str__(self)`
            * Convierte el array `components` en una cadena de texto para mostrar por pantalla.
        * `__repr__(self)`
            * Convierte el array `components` en una cadena de texto para representar.
        * `__hash__(self)`
            * Convierte el array en un `tuple` del que se pueda hacer un `hash` y devuelve el `hash` de la misma.
        * `__add__(self, other)`
            * Devuelve un nuevo vector con la suma de dos vectores o la suma de un vector con un número entero.
        * `__sub__(self, other)`
            * Devuelve un nuevo vector con la resta de dos vectores, o la resta de un vector con un número entero.
        * `__mul__(self, other)`
            * Devuelve un nuevo vector con la multiplicación de dos vectores, o la multiplicación de un vector con un 
            número entero.
        * `__truediv__(self, other)`
            * Devuelve un nuevo vector con la división de dos vectores, o la división de un vector con un número entero.
        * `__pow__(self, power, modulo=None)`
            * Devuelve un nuevo vector con la potencia de dos vectores, o la potencia de un vector con un número entero.
        * `__ge__(self, other)`
            * Para que un vector sea igual o mayor a otro, todas las componentes, deben ser mayores o iguales a su 
            pareja.
        * `__gt__(self, other)`
            * Para que un vector sea estrictamente mayor a otro, todas las componentes, deben ser mayores a su pareja.
        * `__lt__(self, other)`
            * Para que un vector sea igual o menor a otro, todas las componentes, deben ser mayores o iguales a su
            pareja.
        * `__le__(self, other)`
            * Para que un vector sea estrictamente menor a otro, todas las componentes, deben ser menores a su pareja. 
    * `tolist(self)`
        * **Descripción**
            * Devuelve una lista con los componentes del vector.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un `iterable`, que puede ser tratado como una lista de los componentes.
    * `copy(self)`
        * **Descripción**
            * Devuelve una copia del vector.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un `Vector`, instancia nueva del vector actual.
    * `all_close(self, v2, relative=None) -> bool`
        * **Descripción**
            * Devuelve un booleano que indica si dos vectores son cercanos, con una cierta tolerancia.
        * **Parámetros**
            * `v2` - Vector con el cual vamos a comparar el nuestro.
            * `relative=None` - Tolerancia para la comparación.
        * **Salida**
            * Devuelve un `bool`, booleano que indica si son parecidos o no.
    * `dominance(self, v2, relative=None) -> Dominance`
        * **Descripción**
            * Devuelve un objeto de la clase [`Dominance`](dominance.md), con la dominancia entre vectores, con cierta
            tolerancia.
        * **Parámetros**
            * `v2` - Vector con el cual vamos a comprar el nuestro.
            * `relative=None` - Tolerancia para la comparación.
        * **Salida**
            * Devuelve un `Dominance`, objeto que indica la dominancia entre vectores.
    * **Estáticos**
        * `m3_max(vectors: list) -> list`
            * **Descripción**
                * Dada una lista de vectores, devuelve de forma eficiente una lista con los vectores no dominados.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
            * **Salida**
                * Devuelve un `list`, lista de vectores no dominados.
        * `m3_max_2_sets(vectors: list) -> (list, list)`
            * **Descripción**
                * Dada una lista de vectores, devuelve dos listas, la primera con vectores no dominados y la segunda con
                los dominados, basándose en el algoritmo `m3`. La segunda lista puede contener vectores duplicados.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
            * **Salida**
                * Devuelve un `tuple` con dos `list`, tupla con dos lista que representa a los vectores no dominados y 
                dominados.
        * `m3_max_2_sets_not_duplicates(vectors: list) -> (list, list)`
            * **Descripción**
                * Tiene la misma función que el anterior, pero evita vectores duplicados en la segunda lista.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
            * **Salida**
                * Devuelve un `tuple` con dos `list`, tupla con dos lista que representa a los vectores no dominados y 
                dominados.
        * `m3_max_2_sets_with_repetitions(vectors: list) -> (list, list, list)`
            * **Descripción**
                * Dada una lista de vectores, devuelve una tupla de tres elementos, donde el primero es la lista de
                vectores no dominados sin repeticiones, la segunda es la lista de vectores dominados, y una tercera 
                lista con los vectores no dominados repetidos.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
            * **Salida**
                * Devuelve un `tuple` con tres `list`, tupla con tres lista que representa a los vectores no dominados
                 únicos, dominados y no dominados repetidos.
                 
[< Volver](index.md)