# Espacios básicos de `gym`
#### `gym.Env`

Los entornos tienen definidas las variables `action_space` y `observation_space`. Estas variables pertenecen a la clase 
`Space`, la cual, tiene una serie de atributos y métdos útiles, que se describirán a continuación:

* Space(object)
    * Atributos
        * **shape**
        * **dtype**
    * Métodos
        * **sample:** Coge un elemento aleatorio uniforme de este espacio.
        * **seed(seed):** Establece la semilla `seed`
        * **contains(x):** Devuelve `True` si `x` pertenece al espacio.
        * **to_jsonable(sample_n):** Convierte `samples_n` de este espacio en un tipo JSONable.
        * **from_jsonable(sample_n):** Operación inversa al anterior.

* Box(Space)
    * **Una Caja en R^n**
    * Atributos
        * **low**
        * **high**
        * **shape**
        * **dtype**
    * Métodos
        * **__init__(low, high, shape, dtype):** Con el constructor tenemos dos opciones:
            * Bien `low` y `high`, son escalares y le pasamos unas dimensiones, para que se redimensionen.
            * O bien `low` y `high` son arrays de las mismas dimensiones. 
        
* Dict(Space)
    * **Un diccionario de espacios simples**
    * Atributos
        * **spaces:** El diccionario de espacios.
    
* Discrete(Space)
    * **{0, 1, ..., n - 1}**
    * Atributos
        * **n**
        
* MultiBinary(Space)
    * Atributos
        * **n**
    
* MultiDiscrete(Space)
    * Atributos
        * **nvec:** vector de contadores de cada variable categórica.
        
* Tuple(Space)
    * **Una tupla de espacios simples**
    * Atributos
        * **spaces:** La tupla con los dos espacios.

[< Volver](index.md)