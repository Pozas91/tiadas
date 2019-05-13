# Vector acción
#### `models.ActionVector(Vector)`

Esta clase es un "wrapper" de la clase [`Vector`](vector.md), que permite asociar una acción al vector dado, de esta
forma se pueden hacer unas operaciones más eficientes en algunos puntos del proyecto. 

* **Atributos**
    * `action` - Representa la acción asociada al vector.
    * `vector` - Objeto de la clase [`Vector`](vector.md) para realizar las operaciones.
* **Métodos**
    * **Métodos mágicos**
        * `__str__(self)`
            * Devuelve una cadena de texto para mostrar por pantalla la acción y el vector.
        * `__repr__(self)`
            * Devuelve una cadena de texto para representar a la acción y al vector.
    * `dominance(self, v2) -> `[`Dominance`](dominance.md)
        * **Descripción**
            * Devuelve un objeto de la clase [`Dominance`](dominance.md), con la dominancia entre los atributos `vector`
            de ambas clases.
        * **Parámetros**
            * `v2` - Vector con el cual vamos a comprar el nuestro.
        * **Salida**
            * Devuelve un [`Dominance`](dominance.md), objeto que indica la dominancia entre vectores.
    * **Estáticos**
        * `actions_occurrences_based_m3_with_repetitions(vectors: list, number_of_actions: int) -> list`
            * **Descripción**
                * Método para obtener el número de vectores no dominados que tiene cada acción asociada. Este algoritmo
                está basado en el `m3_max_2_sets_with_repetitions` de la clase [`Vector`](vector.md) para hacer más
                eficiente el cáculo de la cardinalidad.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
                * `number_of_actions` - Número de posibles acciones diferentes.
            * **Salida**
                * Devuelve un `list`, lista de longitud `number_of_actions`, que contiene el número de vectores no 
                dominados que tiene cada acción.
                 
[< Volver](index.md)