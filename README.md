# TIADAS

Parrafo para la descripción del proyecto

## Empezando

Instrucciones para hacer la copia del proyecto y ejecutarlo en una máquina local para propósitos de desarrollo y testing.
Ver notas de despliegue para ver como desplegarlo en un sistema real.

### Requisitos

Antes de usar el proyecto es necesario la instalación de una serie de librerías. Estas librerías están definidas en el
fichero `requirements.txt`, para instalar las librerías es necesario ejecutar el siguiente comando.

```text
python -m pip install -r requirements.txt
```

### Instalación

El proyecto se ha preparado para instalar los entornos como paquete independiente, para poder realizar varias pruebas
con ellos siguiendo el esquema propuesto por `gym`, el paquete generado e instanlado en el equipo se llamará `gym_tiadas`
para hacer esto es necesario, acceder a la raíz del proyecto y ejecutar el comando:

```text
python -m pip install .
```

Esto lo instalará como paquete independiente dentro de python, aunque no es un paso necesario si se quiere ejecutar
fichero del propio proyecto.

Una vez realizado el paso anterior bastará con usar el paquete importándolo:

```python
import gym_tiadas
```

## Ejecución de las pruebas.
Explicar como se ejecutan los test automáticos para el sistema

## Despliegue

Pasos adicionales que hay que seguir para desplegar el proyecto en un entorno real

## Construido con
* [gym](https://gym.openai.com/) - Entornos y utilidades para el aprendizaje por refuerzo.
* [numpy](https://www.numpy.org/) - Paquete para computación ciéntifica.

## Contribuir
Por favor leer el fichero [CONTRIBUTING.md](contributing.md) para obtener detalles de conducta en nuestro código, y el proceso para
enviar peticiones de pull.

## Wiki
Puedes encontrar mucho más sobre como utilizar este proyecto en nuestra [Wiki](docs/index.md).

## Versionado

Usamos [SemVer](http://semver.org/) para el versionado. Para ver las versiones disponibles, ver 
[tags on this repository](https://github.com/your/project/tags).

## Autores
* Autor uno - Trabajo - Enlace
* Autor dos - Trabajo - Enlace

## Licencia
ESte proyecto está bajo licencia tal tal tal, para ver licencia visitar el archivo [LICENSE.md](LICENSE.md)

## Agradecimientos
Agradecimientos a:
* Uno
* Dos
* Tres