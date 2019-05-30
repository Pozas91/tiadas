# Agente Q-Learning multi-objectivo escalarizado
#### `agents.AgentMOSP(`[`AgentQ`](agent_q.md)`)`

Este agente hereda el agente [`AgentQ`](agent_q.md), la diferencia con el anterior es que se le asignan unos pesos, y 
puede ser usado en problemas multi-objectivo, ya que multiplicará el vector pesos, por el vector recompensas, obteniendo
un flotante que es una recompensa escalarizada. 

* **Atributos**
    * `weights` - Vector de pesos, para realizar la multiplicación escalarizada.
    
* **Métodos**
    * `_update_q_values(self, reward: Vector, action: int, next_state: object) -> None`
        * **Descripción**
            * Este método aplica la fórmula del `Q-Learning` y actualiza los valores esperados en `Q`.
        * **Parámetros**
            * `reward: `[`Vector`](../models/vector.md) - Vector de recompensa recibido por el entorno.
            * `action: int` - Acción que ha sido escogida para realizarse.
            * `next_state: object` - Nuevo estado que se alcanzará al realizar la acción anterior al estado actual.
        * **Salida**
            * **No ofrece salida**
    * `process_reward(self, reward: `[`Vector`](../models/vector.md)`) -> `[`Vector`](../models/vector.md)
        * **Descripción**
            * Multiplica el vector recompensa por el vector de pesos, para obtener una recompensa escalarizada.
        * **Parámetros**
            * `reward: `[`Vector`](../models/vector.md) - Vector que representa a la recompensa
        * **Salida**
            * Devuelve un [`Vector`](../models/vector.md), vector que representa la recompensa procesada.
    * `find_c_vector(self, w1: float, w2: float, solutions_known: list = None) ->  `[`Vector`](../models/vector.md)
        * **Descripción**
            * Dados dos pesos calcula cual es el Vector de dos dimensiones máximo que se puede alcanzar.
            Este método ofrece la posibilidad de pasarle una lista de soluciones para agilizar el proceso y parar cuando
            encuentre la mejor solución.
        * **Parámetros**
            * `w1: float` - Peso para la primera componente de la solución
            * `w2: float` - Peso para la segunda componente de la solución.
            * `solutions_known: list = None` - Lista de soluciones conocidas (frontera de pareto)
        * **Salida**
            * Devuelve un [`Vector`](../models/vector.md), vector 2D óptimo encontrado.
    
    * `calc_frontier_scalarized(p: `[`Vector`](../models/vector.md)`, q: `[`Vector`](../models/vector.md)`, 
    solutions_known: list = None) -> list`
        * **Descripción**
            * Dado dos puntos iniciales, devuelve por búsqueda dicotómica todos los puntos que forman la
            frontera de pareto, este método hace uso del anterior y sólo es válido para vectores 2D. Si se conoce los 
            vectores óptimos se pueden proporcionar para agilizar el proceso.
        * **Parámetros**
            * `p: `[`Vector`](../models/vector.md) - Primer vector de referencia.
            * `q: `[`Vector`](../models/vector.md) - Segundo vector de referencia.
            * `soluctions_known: list = None` -  Soluciones conocidas para agilizar el proceso
        * **Salida**
            * Devuelve un `list`, lista de todos los vectores que conforman la frontera de pareto.
 
[< Volver](index.md)