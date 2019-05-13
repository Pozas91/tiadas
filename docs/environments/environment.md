# Entorno base
#### `gym_tiadas.gym_tiadas.envs.Environment(gym.Env)`

Tenemos un entorno llamado `Environment` que es una abstracción de la clase 
[`gym.Env`](http://gym.openai.com/docs/#environments), proporcionada por `gym`. Este entorno tiene una serie de métodos 
y atributos para que sea compatible con `gym` y a su vez tenga métodos comunes y estructurados.

* **Atributos**
    * `_actions: dict` - Las acciones posibles que se pueden realizar en el entorno, donde la clave del diccionario
    es un texto descriptivo y el valor un número entero.
    * `_icons: dict` - Carácteres usados para representar los entornos, donde la clave es un texto descriptivo y el
    valor es el caracter.
    * `action_space: gym.spaces.Discrete` - Es un objeto de la clase `Discrete` que representa un vector de números
    enteros.
    * `observation_space: gym.spaces` - Es un objeto de cualquier clase del módulo `spaces` de `gym`, estos son los
    estados posibles con los que cuenta el entorno.
    * `np_random: `
    [`numpy.random.RandomState()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html) - 
    Es una instancia de la clase `RandomState` de `numpy`, y es usado para generar número aleatorios en el entorno.
    * `seed: int` - Número entero que representa a la semilla inicial con la que se inicializa el generador de 
    números aleatorios.
    * `initial_state: object` - Es un objeto que representa el estado inicial del entorno, en nuestro caso puede ser
    una tupla o un entero.
    * `current_state: object` - Estado actual del entorno.
    * `finals: dict` - Diccionario con todos los estados finales del entorno.
    * `obstacles: frozenset` - Conjunto inmutable con los estados que el agente no puede alcanzar (es un obstáculo).
    * `default_reward: `[`Vector`](../models/vector.md) - Vector recompensa devuelto por defecto.

* **Propiedades**
    * `actions -> dict` - Devuelve el diccionario `_actions`, conviertiéndolo en solo lectura.
    * `icons -> dict` - Devuelve el diccionario `_icons`, conviertiéndolo en solo lectura.
    
* **Métodos**
    
    * `step(self, action: int) -> (object, `[`Vector`](../models/vector.md)`, bool, dict)`
        * **Descripción**
            * Recibe un número entero como acción, realiza un movimiento en el entorno y devuelve la información
            conseguida.
        * **Parámetros**
            * `action: int` - Una acción perteneciente al espacio de acciones, y definida en el diccionario de acciones.
            Indica el movimiento que va a realizar el agente.
        * **Salida**
            * Una tupla donde:
                * El primer elemento es de tipo `object`, e indica el estado alcanzado.
                * El segundo elemento es de tipo [`Vector`](../models/vector.md), e indica la recompensa obtenida al realizar el movimiento.
                * El tercer elemento es de tipo `bool`, e indica si el estado alcanzado es un estado final o no.
                * El cuarto y último elemento es de tipo `dict`, y es un diccionario con información adicional.
                
    * `seed(self, seed: int = None) -> list`
        * **Descripción**
            * Recibe como parámetro un número entero correspondiente a la semilla inicial que se le pasará al sistema
            generador.
        * **Parámetros**
            * `seed: int` - Semilla inicial.
        * **Salida**
            * Devuelve como salida una lista que tiene una semilla modificada por la función [`seeding`](../utils/seeding.md) de 
            `gym`.
            
    * `reset(self) -> object`
        * **Descripción**
            * Devuelve al entorno a un estado inicial, modificando el valor de las variables necesarias.
        * **No recibe parámetros**
        * **Salida**
            * Suele devolver el estado inicial del entorno.
            
    * `render(self, mode: str = 'human') -> None`
        * **Descripción**
            * Su finalidad es devolver de manera visual el estado en el que se encuentra el entorno.
        * **Parámetros**
            * `mode='human': ` - Texto que indica el tipo de salida que queremos obtener.
        * **Salida**
            * Información por consola sobre el entorno.
    
    *  `next_state(self, action: int, state: object = None) -> object`
        * **Descripción**
            * Devuelve el siguiente estado que se alcanza al aplicar la acción recibida en el estado indicado. Si no se
            indica un estado, se toma el actual.
        * **Parámetros**
            * `action: int` - Una acción perteneciente al espacio de acciones, y definida en el diccionario de acciones.
            Indica el movimiento que va a realizar el agente.
            * `state: object = None` - Un objeto que representa a un estado, si no se indica, se usará el estado actual
            del entorno.
        * **Salida**
            * Devuelve el siguiente estado.
            
    * `get_dict_model(self) -> dict`
        * **Descripción**
            * Obtiene las variables del modelo y las convierte en un diccionario que podrá ser usado para almacenar los 
            datos del modelo.
        * **No recibe parámetros**
        * **Salida**
            * Devuelve un diccionario con las variables más importantes del entorno.
            
    * `is_final(self, state: object = None) -> bool`
        * **Descripción**
            * Comprueba si un estado indicado es final o no.
        * **Parámetros**
            * `state: object = None` - Indica el estado sobre el que queremos hacer la comprobación, si no se indica, se
            tomará el estado actual del entorno.
        * **Salida**
            * Un valor booleano que indica si el estado es final o no.

[< Volver](index.md)