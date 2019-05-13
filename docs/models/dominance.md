# Dominancia
#### `models.Dominance(Enum)`

Esta clase hereda de `Enum` y por tanto, es una clase enumerada que tiene 4 posibles valores, y es usada como referencia
para ver la dominancia entre dos vectores:
* `dominate = 1` - Indica si el vector que estamos tratando domina a otro vector.
* `is_dominated = 2` - Indica si el vector que estamos tratando es dominado por otro vector.
* `equals = 3` - Indica si ambos vectores son iguales
* `otherwise = 4` - Indica que ningún vector domina o es dominado por otro.

[< Volver](index.md)