# Agente MultiObjetivo

### Recompensas
Recibe un vector de recompensas del estilo __R__(s, a) = (R<sub>1</sub>(s, a), ..., R<sub>m</sub>)), donde m es el número
de objetivos que estamos tratando. Por ejemplo:

```python
rewards = [0.2, 0.3, 0.1, 30.0]
```

### Q-Valores
Convertirmos el diccionario de Q, en un vector de valores, __Q__(s, a) = (Q<sub>1</sub>(s, a), ..., Q<sub>m</sub>)), en
nuestro caso vamos a usar la siguiente estructura:

Estructura anterior
```json
{
    state_1: {action_1: reward, action_2: reward, ..., action_m: reward},
    ...
    state_n: {action_1: reward, action_2: reward, ..., action_m: reward},
}
```

Estructura multi-objetivo

```json
{
    state_1: {
      action_1: [reward_1, .., reward_o],
      ...,
      action_m: [reward_1, .., reward_o],
    },
    ...,
    state_n: {
      action_1: [reward_1, .., reward_o],
      ...,
      action_m: [reward_1, .., reward_o],
  },
}
```