# Modelos

Contiene funciones útiles para los modelos del proyecto

* `lazy_property(fn) -> object`
    * **Descripción**
        * Recibe una función que simula una propiedad y la carga de forma perezosa, está pensada para ser usadada como
        un `decorator` de Python 
    * **Parámetros**
        * `fn` - Función a la que se le aplica
    * **Salida**
        * Devuelve un `object`, objeto que representa al atributo que estamos llamando.

[< Volver](index.md)