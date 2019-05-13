# Agente Q-Learning mono objectivo
#### `agents.Agent`

La misión de este agente es resolver problemas de un solo objetivo, mediante aprendizaje por refuerzo basado en
diferencias temporales, más concretamente Q-Learning.

El agente irá iterando por los diferentes estados del entorno indicado, realizando una acción sobre él, con una política
`e-greedy`, que ayudará a que el agente pruebe varias opciones para favorecer la exploración del entorno y no ceñirse a
la primera recompensa encontrada.

La fórmula del Q-Learning es la siguiente:

```text
Q(St, At) <- (1 - alpha) * Q(St, At) + alpha * (r + y * Q(St_1, action))
```

Donde, `Q(St, At)` es el valor esperado al estar en el estado `St` y aplicar la acción `At` que se irá actualizando en
cada iteración. `alpha` es la tasa de aprendizaje del agente, `r` es la recompensa obtenida por el entorno al usar la
acción `At` en el estado `St`. `y` es una tasa de descuento, que evita que el agente se encuentre dando ciclos mientras
aprende. Y por último tenemos `Q(St_1, action)`, que es el valor esperado por el siguiente estado alcanzado, para
cualquier acción que se pueda realizar.

* **Atributos**
    * `_icons` - Diccionario de los posibles iconos usados por el agente, donde la clave es un texto identificable y el
    valor es un caracter que representa al icono.
    * `alpha` - Tasa de aprendizaje del agente.
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
    * `q` - Diccionario de diccionarios, donde guardaremos los valores esperados de Q. La estructura del diccionario es 
    la siguiente:
        ```json
        {
            state_1: {action_1: reward, action_2: reward, action_3: reward, ...},
            state_2: {action_1: reward, action_2: reward, action_3: reward, ...},
            state_3: {action_1: reward, action_2: reward, action_3: reward, ...},
            ...
        }
        ```
    * `rewards_history` - Lista de las recompensas obtenidas por el agente a lo largo de su ejecución.
    * **Propiedades**
        * `v -> `[`Vector`](../models/vector.md) - Obtiene el mejor vector del estado inicial.
    
* **Métodos**
    * `select_action(self, state: object = None) -> int`
        * **Descripción**
            * Dado un estado, un acción basado en una política e-greedy. Si se aplica el epsilon se cogerá un valor 
            aleatorio (**exploración**), si no se cogerá la mejor acción posible (**explotación**).
        * **Parámetros**
            * `state: object = None` - Un objeto que representa un estado, si este valor es `None` se tomará el estado
            actual del agente.
        * **Salida**
            * Devuelve un `int`, entero que representa a una acción.
    * `walk(self) -> list`
        * **Descripción**
            * Una vez que el agente ha sido entrenado y ha aprendido una política, esta función la recorre.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un `list`, lista de las recompensas recibidas.
    * `episode(self) -> None`
        * **Descripción**
            * Realiza un episodio completo en el entorno hasta encontrar un estado fina. A medida que se realizan 
            episodios, los valores esperados `Q` se van actualizando en función de las recompensas recibidas por el
            entorno.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `_update_q_dictionary(self, reward: `[`Vector`](../models/vector.md)`, action: int, next_state: object) -> None`
        * **Descripción**
            * Este método aplica la fórmula del `Q-Learning` y actualiza los valores esperados en `Q`.
        * **Parámetros**
            * `reward: `[`Vector`](../models/vector.md) - Vector de recompensa recibido por el entorno.
            * `action: int` - Acción que ha sido escogida para realizarse.
            * `next_state: object` - Nuevo estado que se alcanzará al realizar la acción anterior al estado actual.
        * **Salida**
            * **No ofrece salida**
    * `show_q(self) -> None`
        * **Descripción**
            * Muestra el diccionario de valores esperados `Q`.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Escribe por pantalla los valores de `Q`
    * `show_policy(self) -> None`
        * **Descripción**
            * Muestra las políticas que se van a tomar para cada estado en forma de tabla.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por pantalla la matriz de estados con la política tomada.
    * `show_raw_policy(self) -> None`
        * **Descripción**
            * Muestra la política en forma de lista, asociando a cada estado la acción que se va a tomar.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por pantalla la lista de estados con la acción asociada.
    * `reset(self) -> None`
        * **Descripción**
            * Devuelve al agente a un estado inicial, borrando los valores de `Q` obtenidos hasta el momento, el
            historial de recompensas, las iteraciones realizadas y reinicializando el estado actual.
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
    * `_best_reward(self, state: object) -> `[`Vector`](../models/vector.md)
        * **Descripción**
            * Devuelve la mejor recompensa que puede ofrecer el estado indicado.
        * **Parámetros**
            * `state: object` - Un objecto que representa a un estado.
        * **Salida**
            * Devuelve un [`Vector`](../models/vector.md), vector que representa a la recompensa encontrada.
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
    * `process_reward(self, reward: `[`Vector`](../models/vector.md)`) -> `[`Vector`](../models/vector.md)
        * **Descripción**
            * Procesa la recompensa recibida antes de usarla
        * **Parámetros**
            * `reward: `[`Vector`](../models/vector.md) - Vector que representa a la recompensa
        * **Salida**
            * Devuelve un [`Vector`](../models/vector.md), vector que representa la recompensa procesada.
    * `show_observed_states(self) -> None`
        * **Descripción**
            * Muestra la gráfica de los estados observados a lo largo de las iteraciones.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra un gráfico con los estados observados.
    * `show_v_values(self) -> None`
        * **Descripción**
            * Muestra una lista de cada estado con el valor máximo encontrado.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por consola una lista de todos los estados con el `V` máximo.
 
[< Volver](index.md)