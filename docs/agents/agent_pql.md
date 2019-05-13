# Agent PQL
#### `agents.AgentPQL(Agent)`

Está basado en el artículo "Multi-Objective Reinforcement Learning using Set of
Pareto Dominating Policies" de Kristof Van Moffaert and Ann Nowé.

Este agente mantiene tres diccionarios, uno de la media de las recompensas encontradas, otro de vectores no dominados y 
por último uno de ocurrencias. Estos diccionarios son usados para hacer cálculos en tiempo de ejecución, y así, poder
aprender varias políticas simultáneamente.

#### Diccionario de ocurrencias
Mantiene el número de veces que ha pasado un agente por un estado y ha realizado una acción concreta, la estructura es:

La diferencia principal con los otros agentes, es que este agente realmente no aprende, va sobreescribiendo los valores
encontrados.

* **Atributos**
    * `json_indent` - Tabulación que tendrán las líneas en los ficheros json.
    * `dumps_path` - Ruta relativa hacia el directorio de volcado de datos.
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
    * `r` - Diccionario de diccionario, donde se almacena la recompensa inmediata media observada:
        ```json
        {
            state_1: {action_1: average_reward, action_2: average_reward, ...},
            state_2: {action_1: average_reward, action_2: average_reward, ...},
            ...
        }
        ```
    * `nd` - Diccionario de diccionario, donde se almacena una lista de los vectores no dominados que ha ido encontrado 
    el agente:
        ```json
        {
            state_1: {action_1: [vector_no_dominado_1, vector_no_dominado_2, ...], action_2: [vector_no_dominado_1, vector_no_dominado_2, ...], ...},
            state_2: {action_1: [vector_no_dominado_1, vector_no_dominado_2, ...], action_2: [vector_no_dominado_1, vector_no_dominado_2, ...], ...},
            ...
        }
        ```
    * `n` - Diccionario de diccionario, donde se almacena el número de veces que el agente ha pasado por un estado y ha 
    realizado una acción concreta:
        ```json
            {
                state_1: {action_1: occurrences, action_2: occurrences, ...},
                state_2: {action_1: occurrences, action_2: occurrences, ...},
                ...
            }
        ```
    * `hv_reference` - Vector de referencia a partir del cual se calcula el hipervolumen.
    * `evaluation_mechanism` - Texto que indica un tipo de mecanismo de evaluación concreto. Este texto puede ser:
        * `'HV-PQL'` - Evaluación por hipervolumen, el agente escoge el mayor hipervolumen
        * `'PO-PQL'` - Evaluación por pareto, el agente escoge entre todas las acciones que tengan al menos un vector no
         dominado.
        * `'C-PQL'` - Evaluación por cardinalidad, el agente escoge la acción con más vectores no dominados.
    
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
    * `episode(self) -> None`
        * **Descripción**
            * Realiza un episodio completo en el entorno hasta encontrar un estado final. A medida que se realizan 
            episodios, se van almacenando los vectores no dominados que el agente vaya encontrando, actualizando la
            recompensa media y el contador de visitas del estado.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `get_and_update_n_s_a(self, state: object, action: int) -> int`
        * **Descripción**
            * Actualiza y obtiene el número de veces que el agente ha estado en el estado `state`, realizando la acción
            `action`.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
            * `action: int` - Acción realizada.
        * **Salida**
            * Devuelve un `int`, entero que simboliza el número de veces.
    * `update_r_s_a(self, state: object, action: int, reward: `[`Vector`](../models/vector.md)`, occurrences: int) -> None`
        * **Descripción**
            * Actualiza la recompensa media para el estado `state`, realizando la acción `action`.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
            * `action: int` - Acción realizada.
            * `reward: `[`Vector`](../models/vector.md) - Recompensa que ha devuelto el entorno.
            * `occurrences: int` - Número de veces que el agente ha pasado.
        * **Salida**
            * **No ofrece salida**
    * `update_nd_s_a(self, state: object, action: int, next_state: object) -> None`
        * **Descripción**
            * Actualiza los vectores no dominados que el agente vaya encontrando.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
            * `action: int` - Acción realizada.
            * `next_state: object` - Siguiente estado que se va a alcanzar.
        * **Salida**
            * **No ofrece salida**
    * `q_set(self, state: object, action: int) -> list`
        * **Descripción**
            * Obtiene el conjunto `Q` calculado con la recompensa media y los vectores no dominados.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
            * `action: int` - Acción realizada.
        * **Salida**
            * Devuelve un `list`, lista de vectores `Q`.
    * `reset(self) -> None`
        * **Descripción**
            * Devuelve al agente a un estado inicial, borrando las recompensas, los vectores no dominados, las
            ocurrencias, etc.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `reset_iterations(self) -> None`
        * **Descripción**
            * Resetea el número de iteraciones a 0.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * **No ofrece salida**
    * `best_action(self, state: object = None) -> int`
        * **Descripción**
            * Dado un estado devuelve la mejor acción asociada a ese estado de acuerdo al método de evaluación 
            especificado, en caso de que varias acciones sean "óptimas", se cogerá una al azar entre ellas. Si no se 
            indica un estado, se cogerá el estado actual.
        * **Parámetros**
            * `state: object = None` - Un objeto que representa un estado, si este valor es `None` se tomará el estado
            actual del agente.
        * **Salida**
            * Devuelve un `int`, entero que representa a una acción.
    * `show_observed_states(self) -> None`
        * **Descripción**
            * Muestra la gráfica de los estados observados a lo largo de las iteraciones.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra un gráfico con los estados observados.
    * `track_policy(self, state: object, target: `[`Vector`](../models/vector.md)`) -> list`
        * **Descripción**
            * Ejecuta un episodio usando el objetivo `target` en búsqueda de una de las políticas aprendidas, desde el 
            estado `state`. En caso de que no se haya encontrado la política adecuada coge el camino que tenga menor
            distancia euclídea con respecto a la indicada. Devuelve los estados visitados.
        * **Parámetros**
            * `state: object` - Estado inicial para comenzar la búsqueda.
            * `target: `[`Vector`](../models/vector.md) - Vector objetivo que queremos encontrar.
        * **Salida**
            * Devuelve un `list`, lista de estados visitados.
    * `_best_hypervolume(self, state: object) -> float`
        * **Descripción**
            * Calcula el mejor hipervolumen posible dado un estado para todas las acciones posibles.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
        * **Salida**
            * Devuelve un `float`, flotante con el mejor hipervolumen.
    * `hypervolume_evaluation(self, state: object) -> int`
        * **Descripción**
            * Calcula la mejor acción posible usando el método de evaluación `HV-PQL`. En caso de empate, elige de forma
            aleatoria entre las mejores acciones. Este método consiste en coger la acción que ofrezca el mayor
            hipervolumen. 
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
        * **Salida**
            * Devuelve un `int`, entero con la acción tomada.
    * `cardinality_evaluation(self, state: object) -> int`
        * **Descripción**
            * Calcula la mejor acción posible usando el método de evaluación `C-PQL`. En caso de empate, elige de forma
            aleatoria entre las mejores acciones. Este método consiste en coger la acción que ofrezca el mayor número de
            vectores no dominados.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
        * **Salida**
            * Devuelve un `int`, entero con la acción tomada.
    * `pareto_evaluation(self, state: object) -> int`
        * **Descripción**
            * Calcula la mejor acción posible usando el método de evaluación `PO-PQL`. En caso de empate, elige de forma
            aleatoria entre las mejores acciones. Este método consiste en coger una acción cualquiera que ofrezca al 
            menos un vector no dominado.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
        * **Salida**
            * Devuelve un `int`, entero con la acción tomada.
    * `get_dict_model(self) -> dict`
        * **Descripción**
            * Obtiene las variables del modelo y las convierte en un diccionario que podrá ser usado para almacenar los 
            datos del modelo.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un diccionario con las variables más importantes del agente y el entorno asociado.
    * `to_json(self) -> str`
        * **Descripción**
            * Convierte al agente en un texto json, obteniendo los valores más importantes.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Devuelve un `str`, texto en formato json.
    * `save(self, filename: str = None) -> None`
        * **Descripción**
            * Guarda un fichero en formato `json` con la información más relevante del agente. Si no se indica un nombre
            de fichero, se tomará el nombre del entorno en `snake_case`, junto con el algoritmo de evaluación usado y un
            sello temporal.
        * **Parámetros**
            * `filename: str = None` - Nombre del fichero, por defecto `None`.
        * **Salida**
            * **No ofrece salida**
    * `non_dominated_vectors_from_state(self, state: object) -> list`
        * **Descripción**
            * Obtiene una lista de todos los vectores no dominados en el estado `state`.
        * **Parámetros**
            * `state: object` - Estado que estamos examinando.
        * **Salida**
            * Devuelve un `list`, lista con todos los vectores no dominados.       
    * `print_information(self) -> None`
        * **Descripción**
            * Muestra por consola información básica del agente, como la semilla que tiene, el gamma, etc.
        * **Parámetros**
            * **No recibe parámetros**
        * **Salida**
            * Muestra por consola la información
    * `load(filename: str = None, environment: `[`Environment`](../environments/environment.md)` = None, 
    evaluation_mechanism: str = None) -> object`
        * **Descripción**
            * Carga un modelo del agente con los datos proporcionados. Cargará el nombre de fichero indicado si existe, 
            en caso de no indicar un nombre del fichero, habrá que indicar el entorno y el mecanismo de evaluación y 
            cargará el último almacenado.
        * **Parámetros**
            * `filename: str = None` - Nombre del fichero.
            * `environment: `[`Environment`](../environments/environment.md)` = None` - Entorno del que obtendrá el nombre del fichero.
            * `evaluation_mechanism: str = None` - Mecanismo de evaluación del que obtendrá el nombre del fichero.
        * **Salida**
            * Devuelve un `object`, objeto que representa al modelo `AgentPQL`
    * `dumps_file_path(filename: str) -> str`
        * **Descripción**
            * Obtiene la ruta de volcado de modelos, para el nombre de fichero dado.
        * **Parámetros**
            * `filename: str` - Nombre del fichero a partir del cual se generará la ruta.
        * **Salida**
            * Devuelve un `str`, cadena de texto que indica la ruta.
    
 
[< Volver](index.md)