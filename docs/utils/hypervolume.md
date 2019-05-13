# Utilidades para el hipervolumen

Este fichero contiene una única función que calcula el hipervolumen, la función se llama `calc_hypervolume`, y recibe
como primer parámetro una lista con los vectores (`list_of_vectors`) sobre los cuales se calculará el hipervolumen, y
como segundo parámetro opcional, se podrá pasar un vector de referencia (`reference`). En caso de no pasar un vector de
referencia se tomará como referencia el vector compuesto por las componentes más pequeñas de todos los vectores de la
lista de vectores, menos una unidad.

* `calc_hypervolume(list_of_vectors: list, reference: Vector = None) -> float`
    * **Parámetros**
        * `list_of_vectors: list` - Lista de elementos de la clase `Vector`
        * `reference: Vector` - Un vector de referencia
    * **Salida**
        * Devuelve un número flotante correspondiente al hipervolumen calculado.
    
La función comentada es un "wrapper" de la función `hypervolume` de la librería `pygmo`, este wrapper es necesario para
nuestro proyecto, ya que la función original está diseñada para problemas de minimización, y en nuestro proyecto
buscamos la maximización.

[< Volver](index.md)