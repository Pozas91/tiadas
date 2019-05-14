# Agente Base
#### `agents.Agent`

Los agentes irán iterando por los diferentes estados del entorno indicado, realizando una acción sobre él, con una 
política `e-greedy`, que ayudará a que los agentes prueben varias opciones para favorecer la exploración del entorno y 
no ceñirse a la primera recompensa encontrada.

Este agente se llama `Agent`, y es una abstracción para los agentes definidos en nuestro proyecto, estos agentes tienen 
la  siguiente estructura básica común:

* **Atributos**
    * `_icons` - Diccionario de los posibles iconos usados por el agente, donde la clave es un texto identificable y el
    valor es un caracter que representa al icono.
    * `gamma` - Factor de descuento.
    * `epsilon` - Epsilon usado para la exploración del agente (política e-greedy).
    * `environment` - Entorno de la clase [`Environment`](../environments/environment.md), sobre el que el agente
    interactuará.
    * `max_iterations` - Número máximo de iteraciones hasta decidir que hemos llegado a un estado final, si este valor
    es `None` se considerará sin límite.
    * `iterations` - Contador de iteraciones realizadas.
    * `states_to_observe` - Diccionario donde las claves son los estados que están siendo observados y los valores, son 
    las observaciones realizadas.
    * `state` - Estado actual del agente.
    * `seed` - Semilla con el que el generador de números aleatorios se iniciará.
    * `generator` - Generador de números aleatorios, es una instancia de
    [`numpy.random.RandomState()`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html).
    
* **Métodos**
    * `select_action(self, state: object = None) -> int`
        * **Descripción**
            * Dado un estado, un acción basado en una política e-greedy. Si se aplica el epsilon se cogerá una acción 
            aleatoria (**exploración**), si no se cogerá la mejor acción posible (**explotación**).
        * **Parámetros**
            * `state: object = None` - Un objeto que representa un estado, si este valor es `None` se tomará el estado
            actual del agente.
        * **Salida**
            * Devuelve un `int`, entero que representa a una acción.
    * `episode(self) -> None`
        * **Descripción**
            * Realiza un episodio completo en el entorno hasta encontrar un estado final. 
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `reset(self) -> None`
        * **Descripción**
            * Devuelve al agente a un estado inicial.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `best_action(self, state: object = None) -> int`
        * **Descripción**
            * Dado un estado devuelve la mejor acción asociada a ese estado, en caso de que varias acciones sean
            "óptimas", se cogerá una al azar entre ellas. Si no se indica un estado, se cogerá el estado actual.
        * **Parámetros**
            * `state: object = None` - Un objeto que representa un estado, si este valor es `None` se tomará el estado
            actual del agente.
        * **Salida**
            * Devuelve un `int`, entero que representa a una acción. 
    * `reset_iterations(self) -> None`
        * **Descripción**
            * Resetea el número de iteraciones a 0.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `reset_rewards_history(self) -> None`
        * **Descripción**
            * Borra el historial de recompensas acumulado.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `show_observed_states(self) -> None`
        * **Descripción**
            * Muestra la gráfica de los estados observados a lo largo de las iteraciones.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra un gráfico con los estados observados.
    * `print_information(self) -> None`
        * **Descripción**
            * Muestra por consola información básica del agente, como la semilla que tiene, el gamma, etc.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por consola la información
    * `train(self, epochs: int = 1000) -> None`
        * **Descripción**
            * Realiza sobre el agente el un número de episodios indicado como `epochs`.
        * **Parámetros**
            * `epochs: int = 1000` - Número de episodios ejecutados
        * **Salida**
            * **No ofrece salida**
 
[< Volver](index.md)