# Espacios definidos

Para algunos de nuestros problemas hemos detectado la necesidad de tener entorno cuyas acciones sean dinámicas,
es decir, puede ser que en algunos estados del entorno el agente tenga limitadas algunas acciones. Para ello, hemos
creado nuevos espacios para cubrir esta necesidad:

* `IterableDiscrete(Discrete)`
    * **Descripción**
        * Esta clase funciona igual que la clase [`Discrete`](gym.md) de `gym`, pero le hemos introducido la posiblidad
        de iterar sobre él.

* `DynamicSpace(Space)`
    * **Descripción**
        * Esta clase se usa en los entornos con espacios dinámicos. Para ello, le pasamos la lista de posibles acciones.
        
[< Volver](index.md)