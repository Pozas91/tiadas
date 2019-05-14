# Pareto

Este fichero contiene métodos útiles para calcular la frontera de pareto, para vectores de 2 dimensiones.

* `optimize(agent: `[`AgentQ`](../agents/agent_q.md)`, w1: float, w2: float, solutions_known: list = None) -> 
`[`Vector`](../models/vector.md)
    * **Descripción**
        * Dados dos pesos, y un agente calcula cual es el Vector de dos dimensiones máximo que se puede alcanzar.
        Este método ofrece la posibilidad de pasarle una lista de soluciones para agilizar el proceso y parar cuando
        encuentre la mejor solución.
    * **Parámetros**
        * `agent: `[`AgentQ`](../agents/agent_q.md) - Cualquiera de los agentes definidos en el módulo `agents`.
        * `w1: float` - Peso para la primera componente de la solución
        * `w2: float` - Peso para la segunda componente de la solución.
        * `solutions_known: list = None` - Lista de soluciones conocidas (frontera de pareto)
    * **Salida**
        * Devuelve un [`Vector`](../models/vector.md), vector 2D óptimo encontrado.

* `calc_frontier_scalarized(p: `[`Vector`](../models/vector.md)`, q: `[`Vector`](../models/vector.md)`, problem, 
solutions_known: list = None) -> list`
    * **Descripción**
        * Dado dos puntos iniciales y un agente, devuelve por búsqueda dicotómica todos los puntos que forman la
        frontera de pareto, este método hace uso del anterior y sólo es válido para vectores 2D. Si se conoce los 
        vectores óptimos se pueden proporcionar para agilizar el proceso.
    * **Parámetros**
        * `p: `[`Vector`](../models/vector.md) - Primer vector de referencia.
        * `q: `[`Vector`](../models/vector.md) - Segundo vector de referencia.
        * `problem: `[`AgentQ`](../agents/agent_q.md) - Agente que sacará los vectores.
        * `soluctions_known: list = None` -  Soluciones conocidas para agilizar el proceso
    * **Salida**
        * Devuelve un `list`, lista de todos los vectores que conforman la frontera de pareto.

[< Volver](index.md)