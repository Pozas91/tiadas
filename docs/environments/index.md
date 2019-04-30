# Entornos

Los entornos que usamos en nuestro proyecto heredan de la clase `gym.Env` proporcionada por la librería `gym`, los
atributos y métodos más importantes que tienen estas clases son los siguientes:

* gym.Env
    * Atributos
        * **action_space:** Acciones posibles a realizar.
        * **observation_space:** Estados posibles del entorno.
    * Métodos
        * **step(action):** Realiza la acción `action`
        * **reset:** Vuelve el entorno al comienzo de su ejecución.
        * **render:** Muestra el estado del entorno.

**Para ver una información detallada de cada entorno, puede acceder a los siguientes enlaces:**
* [Malla](mesh.md)
* [Espacios](spaces.md)
* [Seeding](seeding.md)