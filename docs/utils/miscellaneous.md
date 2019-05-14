# Utilidades varias

Estas funciones no están agrupadas, ya que, son pequeñas funciones que son usadas en varios puntos del proyecto.

* `lists_to_tuples(x)`
    * **Descripción**
        * Esta función recibe una lista de listas y las convierte a una tupla de tuplas mediante llamadas recursivas.
    * **Parámetros**
        * `x` - Puede ser una lista de lista o un elemento simple (número, str...).
    * **Salida**
        * Una tupla de tuplas.
    
* `sum_a_vector_and_a_list_of_vectors(v: Vector, v_list: list)`
    * **Descripción**
        * Recibe un vector y una lista de vectores, y devuelve una lista con la suma del vector a cada uno de los
        vectores de la lista.
    * **Parámetros**
        * `v: Vector` - Vector a sumar
        * `v_list: list` - Lista de vectores
    * **Salida**
        * Una lista de cada uno de los vectores con el vector `v` sumado.
        
* `euclidean_distance(a: Vector, b: Vector) -> float`
    * **Descripción**
        * Recibe dos vectores y devuelve la distancia euclídea entre ellos.
    * **Parámetros**
        * `a: Vector` - Primer vector
        * `b: Vector` - Segundo vector
    * **Salida**
        * La distancia euclídea entre los dos vectores en número flotante.

* `manhattan_distance(a: Vector, b: Vector) -> float`
    * **Descripción**
        * Recibe dos vectores y devuelve la distancia manhattan entre ellos.
    * **Parámetros**
        * `a: Vector` - Primer vector
        * `b: Vector` - Segundo vector
    * **Salida**
        * La distancia manhattan entre los dos vectores en número flotante.
        
* `distance_to_origin(a: Vector) -> float`
    * **Descripción**
        * Recibe un vector y devuelve la distancia euclídea desde ese vector hasta el origen **0**.
    * **Parámetros**
        * `a: Vector` - Un vector
    * **Salida**
        * Un flotante que indica la distancia euclídea desde el vector indicado hasta el vector origen.
        
* `order_vectors_by_origin_nearest(vectors: list) -> list`
    * **Descripción**
        * Recibe una lista de vectores y calcula la distancia de cada vector con respecto al origen **0**.
    * **Parámetros**
        * `vectors: Vector` - Lista de vectores
    * **Salida**
        * Una lista de los vectores indicados en orde creciente con respecto a la distancia con el origen.
        
* `str_to_snake_case(text: str) -> str`
    * **Descripción**
        * Recibe un texto y lo devuelve en formato `snake_case`
    * **Parámetros**
        * `text: str` - Texto a transformar
    * **Salida**
        * El texto transformado en `snake_case`
        
[< Volver](index.md)