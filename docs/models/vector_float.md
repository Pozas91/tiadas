# Vector flotante
#### `models.VectorFloat(Vector)`

Clase que representa a un vector flotante, hereda de un vector entero [`Vector`](vector.md), por lo que tiene las mismas
funcionalidades, con algunas diferencias que presentamos a continuación. 
    
* **Métodos**
    * **Métodos mágicos**
        * _**Como son métodos de mágicos de Python vamos a explicar la función que le hemos aplicado**_.
        * `__ge__(self, other)`
            * Para que un vector sea igual o mayor a otro, todas las componentes deben ser mayores o iguales a su 
            pareja, pero en esta caso usamos 
            [`numpy.isclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html), para dotar a la 
            comparación de un pequeño margen correspondiente con la variable `self.relative` heredada.
        * `__gt__(self, other)`
            * Para que un vector sea mayor a otro, todas las componentes deben ser mayores a su  pareja, pero en esta 
            caso usamos [`numpy.isclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html), para
            dotar a la comparación de un pequeño margen correspondiente con la variable `self.relative` heredada.
        * `__lt__(self, other)`
            * Para que un vector sea menor a otro, todas las componentes deben ser menores a su pareja, pero en esta 
            caso usamos  [`numpy.isclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html), 
            para dotar a la comparación de un pequeño margen correspondiente con la variable `self.relative` heredada.
        * `__le__(self, other)`
            * Para que un vector sea igual o menor a otro, todas las componentes deben ser menores o iguales a su 
            pareja, pero en esta caso usamos 
            [`numpy.isclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html), para dotar a la 
            comparación de un pequeño margen correspondiente con la variable `self.relative` heredada. 
    * `to_int(self) -> Vector`
        * **Descripción**
            * Devuelve un vector entero, con todas las componentes multiplicadas por `self.decimals`, para eliminar
            los decimales que no deseamos.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un `Vector`, vector que representa a un vector entero.
    * **Estáticos**
        * `m3_max_2_sets_not_duplicates(vectors: list) -> (list, list)`
            * **Descripción**
                * A diferencia que la versión original en la clase [`Vector`](vector.md), este método usa una
                comprobación previa para ver si dos vectores son parecidos, en caso de ser cierto, descarta las demás
                comprobaciones, evitando así a los vectores duplicados.
            * **Parámetros**
                * `vectors: list` - Lista de vectores.
            * **Salida**
                * Devuelve un `tuple` con dos `list`, tupla con dos lista que representa a los vectores no dominados y 
                dominados.
                 
[< Volver](index.md)