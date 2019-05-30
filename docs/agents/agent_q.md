# Agente Q-Learning mono objectivo
#### `agents.AgentQ(`[`Agent`](agent.md)`)`

La misión de este agente es resolver problemas de un solo objetivo, mediante aprendizaje por refuerzo basado en
diferencias temporales, más concretamente Q-Learning.

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
    * `alpha` - Tasa de aprendizaje del agente.
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
    * `_update_q_values(self, reward: `[`Vector`](../models/vector.md)`, action: int, next_state: object) -> None`
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
    * `show_v_values(self) -> None`
        * **Descripción**
            * Muestra una lista de cada estado con el valor máximo encontrado.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por consola una lista de todos los estados con el `V` máximo.
 
[< Volver](index.md)