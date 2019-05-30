# Vector indexado
#### `models.IndexVector(Vector)`

Esta clase es un "wrapper" de la clase [`Vector`](vector.md), que permite asociar una acción al vector dado, de esta
forma se pueden hacer unas operaciones más eficientes en algunos puntos del proyecto. 

* **Atributos**
    * `index` - Representa al índice asociado al vector.
    * `vector` - Objeto de la clase [`Vector`](vector.md) para realizar las operaciones.
* **Métodos**
    * **Métodos mágicos**
        * `__str__(self)`
            * Devuelve una cadena de texto para mostrar por pantalla la acción y el vector.
        * `__repr__(self)`
            * Devuelve una cadena de texto para representar a la acción y al vector.
        * `__mul__(self, other)`
            * Devuelve un nuevo vector con la multiplicación de dos vectores, o la multiplicación de un vector con un 
            número entero.
    * `dominance(self, v2) -> `[`Dominance`](dominance.md)
        * **Descripción**
            * Devuelve un objeto de la clase [`Dominance`](dominance.md), con la dominancia entre los atributos `vector`
            de ambas clases.
        * **Parámetros**
            * `v2` - Vector con el cual vamos a comprar el nuestro.
        * **Salida**
            * Devuelve un [`Dominance`](dominance.md), objeto que indica la dominancia entre vectores.
    * **Estáticos**
        * `actions_occurrences_based_m3_with_repetitions(vectors: list, actions: list) -> dict`
            * **Descripción**
                * Método para obtener el número de vectores no dominados que tiene cada acción asociada. Este algoritmo
                está basado en el `m3_max_2_sets_with_repetitions` de la clase [`Vector`](vector.md) para hacer más
                eficiente el cáculo de la cardinalidad.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
                * `actions` - Lista de acciones posibles.
            * **Salida**
                * Devuelve un `dict`, diccionario donde cada clave representa a una acción y el valor representa al
                número de vectores no dominados que tiene cada acción.
                 
[< Volver](index.md)