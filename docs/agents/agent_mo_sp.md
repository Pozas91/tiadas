# Agente Q-Learning multi-objectivo escalarizado
#### `agents.AgentMOSP(`[`AgentQ`](agent_q.md)`)`

Este agente hereda el agente [`AgentQ`](agent_q.md), la diferencia con el anterior es que se le asignan unos pesos, y 
puede ser usado en problemas multi-objectivo, ya que multiplicará el vector pesos, por el vector recompensas, obteniendo
un flotante que es una recompensa escalarizada. 

* **Atributos**
    * `weights` - Vector de pesos, para realizar la multiplicación escalarizada.
    
* **Métodos**
    * `_update_q_dictionary(self, reward: Vector, action: int, next_state: object) -> None`
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
 
[< Volver](index.md)