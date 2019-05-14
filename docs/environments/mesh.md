# Entorno malla
#### `gym_tiadas.gym_tiadas.envs.EnvMesh(Environment)`

Este entorno hereda del entorno [`Environment`](environment.md), la principal diferencia es que su espacio de
observación se compone por una tupla de dos valores correspondientes al eje x y al eje y. Por tanto, los estados que
trataremos serán del tipo `tuple`.

* **Atributos**
    * **Mismos atributos que la clase [`Environment`](environment.md).**
    
* **Métodos**
            
    * `render(self, mode='human')`
        * **Descripción**
            * Su finalidad es devolver de manera visual el estado en el que se encuentra el entorno.
        * **Parámetros**
            * `mode='human': ` - Texto que indica el tipo de salida que queremos obtener.
        * **Salida**
            * Devuelve el estado del entorno malla en forma de tabla.
    
    *  `next_state(self, action: int, state: tuple = None) -> tuple`
        * **Descripción**
            * Descompone el estado recibido (o actual), y en función de la acción indicada actualiza una de las dos
            componentes, comprobando que no coincidan con un obstáculo.
        * **Parámetros**
            * `action: int` - Una acción perteneciente al espacio de acciones, y definida en el diccionario de acciones.
            Indica el movimiento que va a realizar el agente.
            * `state: tuple = None` - Una tupla que representa a un estado, si no se indica, se usará el estado actual
            del entorno.
        * **Salida**
            * Devuelve el siguiente estado.

[< Volver](index.md)