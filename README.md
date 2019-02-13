# TIADAS
### Espacios
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
        
### Entornos
* gym.Env
    * Atributos
        * **action_space:** Acciones posibles a realizar.
        * **observation_space:** Estados posibles del entorno.
    * Métodos
        * **step(action):** Realiza la acción `action`
        * **reset:** Vuelve el entorno al comienzo de su ejecución.
        * **render:** Muestra el estado del entorno.

#### Seeding
El fichero seeding usado para las semillas dentro de GYM, crea básicamente un np.random.RandomState() con la semilla pasada, pero hace un hash previo con la semilla.
Esto se hace para evitar la correlación de las salidas cuando se usan varios procesos con las mismas semillas, en entorno sigue siendo replicable pero las salidas no están correlacionadas.

Fuente:
```text
Any given evaluation is likely to have many PRNG's active at
once. (Most commonly, because the environment is running in
multiple processes.) There's literature indicating that having
linear correlations between seeds of multiple PRNG's can correlate
the outputs:
http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
http://dl.acm.org/citation.cfm?id=1276928
Thus, for sanity we hash the seeds before using them. (This scheme
is likely not crypto-strength, but it should be good enough to get
rid of simple correlations.)
```

### Instalar como paquete local
Para instalar los entornos como paquete locales, basta abrir la consola, ir al directorio gym_tiadas dentro de TIADAS y poner:
```
python -m pip install .
```
Esto lo instalará como paquete del entorno dentro del python usado.

### Gráficos
Dentro del directorio plots, se almacenan los gráficos más relevantes que hemos ido generando.

### Espacio Malla
Para diseñar el espacio de malla, hemos optado por dos opciones, la primera y más sencilla (y con mejores resultados en
tiempo), es crear un espacio **Tupla(Discrete, Discrete)**, donde el primer elemento representa el eje de coordenadas X,
y el segundo representa el eje de coordenadas Y.

La segunda opción consiste en usar un espacio discreto, y pasar de tupla a estados discretos con la función usada en la
clase **RussellNorvigDiscrete** .

### Ejemplos
Se han creado dos entornos (uno con tupla y otro discreto), así como dos agentes para probarlos (Agent y AgentDiscrete),
en el fichero ``main.py``, se pueden ver varias funciones con diferentes pruebas.