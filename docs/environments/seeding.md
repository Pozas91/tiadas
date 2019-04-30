# Seeding

La librería `gym` hace uso de un fichero llamado `seeding`, este fichero es usado para las semillas dentro de `gym`,
crea una instancia de la clase `np.random.RandomState()` con la semilla proporcionada, haciendo previamente un hash de
la semilla. Esto se hace con la intención de evitar la correlación de las salidas cuando se usan varios procesos con las
mismas semillas, el entorno sigue siendo replicable pero las salidas no están correlacionadas.

Fuente:
 ```text
Any given evaluation is likely to have many PRNG's active at
once. (Most commonly, because the environment is running in
multiple processes.) There's literature indicating that having
linear correlations between seeds of multiple PRNG's can correlate
the outputs:
http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
http://dl.acm.org/citation.cfm?id=1276928
Thus, for sanity we hash the seeds before using them. (This scheme
is likely not crypto-strength, but it should be good enough to get
rid of simple correlations.)
```