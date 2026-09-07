# Generador de Estadísticas de Jugadores (PlayerStatsGenerator)

## Descripción

El `PlayerStatsGenerator` genera estadísticas sintéticas de jugadores para partidos simulados. A diferencia del motor de Markov que simula el marcador punto a punto, este generador no simula cada acción individual: en su lugar, muestrea de distribuciones estadísticas pre-ajustadas a los datos históricos de cada jugador y normaliza los totales al marcador del set. Es un generador de **stats post-hoc**, no un simulador play-by-play.

*Código: `src/models/player_stats_generator.py` · Parámetros guardados: `models/player_stats_params.json`*

---

## 1. Pipeline de Entrenamiento y Uso

```
┌─────────────────────┐
│  FIT (entrenamiento)│
│                     │
│  player_stats.csv   │
│  (por equipo/temp)  │
│         │           │
│         ▼           │
│  Por cada equipo:   │
│  - Filtrar temp.    │
│    más reciente     │
│  - Filtrar jug.     │
│    con ≥5 sets      │
│  - Calcular         │
│    media/std por    │
│    stat por jugador │
│  - Participation    │
│    rate             │
│         │           │
│         ▼           │
│  team_profiles = {  │
│    Trento: {        │
│      Michieletto: { │
│        puntos:      │
│          {mean,std, │
│           total,    │
│           sets}     │
│        aces: {...}  │
│        ...          │
│      }              │
│    }                │
│  }                  │
│  team_rosters = {   │
│    Trento: [jug1,   │
│             jug2,   │
│             ...]    │
│  }                  │
│         │           │
│         ▼           │
│  save() →           │
│  player_stats_      │
│  params.json        │
└─────────┬───────────┘
          │ (una vez al entrenar)
          │
┌─────────▼───────────┐
│  GENERATE (runtime) │
│                     │
│  load() desde JSON  │
│         │           │
│  Por set simulado:  │
│  generate_set_stats │
│  (team, score,      │
│   opponent_score)   │
│         │           │
│         ▼           │
│  Por jugador:       │
│  - Participation    │
│    check (aleatorio)│
│  - Para cada stat:  │
│    sample normal    │
│    trunc(mean,std)  │
│  - Recalcular pts = │
│    atq+aces+bloq    │
│         │           │
│         ▼           │
│  [{jugador: "...",  │
│    puntos: 13,      │
│    aces: 1,         │
│    ...}]            │
└─────────────────────┘
```

---

## 2. Ajuste de Distribuciones (`fit()`)

### 2.1. Entrada

El método `fit(player_stats, team_stats)` recibe dos DataFrames del pipeline de datos:

- **`player_stats`**: salida de `load_player_stats()` — stats individuales de jugadores por equipo/temporada, con columnas como `jugador`, `sets`, `puntos`, `aces`, `ataques_ganados`, `bloqueos`, `recepciones_exc`, `errores_saque`.
- **`team_stats`**: salida de `load_team_season_stats()` — usado indirectamente (el parámetro existe pero no se usa activamente en el ajuste).

### 2.2. Procesamiento por Equipo

Para cada `equipo_id` en los datos:

1. **Seleccionar la temporada más reciente** disponible para ese equipo (orden descendente por temporada, tomar la primera).
2. **Filtrar jugadores con ≥5 sets jugados** (umbral mínimo de datos).
3. **Para cada jugador**, calcular stats por set:

```python
for stat_key in STAT_KEYS:  # ["puntos", "aces", "ataques_ganados", "bloqueos",
                            #  "recepciones_exc", "errores_saque"]
    per_set = row[stat_key] / sets_played
    stats[stat_key] = {
        "mean": per_set,
        "std": max(per_set * 0.4, 0.1),  # 40% de la media como std, mínimo 0.1
        "total": row[stat_key],           # total histórico
        "sets": sets_played,              # sets jugados
    }
```

4. **Calcular participación**: `sets_del_jugador / max_sets_del_equipo` — define qué proporción de sets juega cada jugador.

### 2.3. Mapeo de IDs a Nombres Canónicos

Los CSVs de `stats_por_equipo_completo/` usan IDs crípticos como `"TN-ITAS"`, `"APG"`, `"MI-POWER"`. El método `_extract_team_name()` usa un mapa hardcodeado para traducirlos:

| ID en CSV | Equipo | ID en CSV | Equipo |
|---|---|---|---|
| TN-ITAS | Trento | APG | Perugia |
| MC | Lube | MI-POWER | Milano |
| VRI | Verona | MIVER | Monza |
| MO | Modena | PIACENZAYOU | Piacenza |
| CIS-VOLLEY | Cisterna | PD | Padova |
| TA | Taranto | BASTIA | Grottazzolina |

---

## 3. Generación de Stats por Set (`generate_set_stats()`)

Cuando se invoca al final de cada set simulado:

```python
stats = player_stats_gen.generate_set_stats(
    team_name="Trento",
    team_score=25,
    opponent_score=20,
)
```

### 3.1. Algoritmo por Jugador

```python
for jugador in roster:
    # 1. Decidir si participa (basado en participation rate)
    if random() > participation_rate:
        continue  # este jugador no juega este set

    # 2. Para cada stat: muestrear de N(mean, std) truncada ≥ 0
    for stat_key in STAT_KEYS:
        mean = profile[jugador][stat_key]["mean"]
        std = profile[jugador][stat_key]["std"]
        value = max(0, round(normal(mean, std)))
        stats[stat_key] = value

    # 3. Recalcular puntos según fórmula de voleibol
    stats["puntos"] = ataques_ganados + aces + bloqueos
```

La desviación estándar se fija en `0.4 × media` (mínimo 0.1), lo que genera una dispersión realista: un jugador que promedia 3 puntos/set tendrá típicamente entre 1 y 5 puntos en un set dado.

### 3.2. Stats Generadas

| Stat | Tipo | Descripción |
|---|---|---|
| `puntos` | calculado | `ataques_ganados + aces + bloqueos` (fórmula estándar de voleibol) |
| `aces` | muestreado | Saques directos |
| `ataques_ganados` | muestreado | Ataques que terminan en punto |
| `bloqueos` | muestreado | Bloqueos ganados |
| `recepciones_exc` | muestreado | Recepciones excelentes (pase perfecto) |
| `errores_saque` | muestreado | Errores de saque |

---

## 4. Persistencia

### 4.1. Guardado (`save()`)

Serializa `team_profiles` y `team_rosters` a `models/player_stats_params.json` (formato JSON, ~200-400 KB según el número de equipos):

```json
{
  "team_profiles": {
    "Trento": {
      "Alessandro Michieletto": {
        "puntos": {"mean": 5.2, "std": 2.1, "total": 415, "sets": 80},
        "aces": {"mean": 0.4, "std": 0.2, "total": 32, "sets": 80},
        "ataques_ganados": {"mean": 4.3, "std": 1.7, ...},
        "bloqueos": {"mean": 0.5, "std": 0.2, ...},
        "recepciones_exc": {"mean": 1.8, "std": 0.7, ...},
        "errores_saque": {"mean": 0.6, "std": 0.2, ...},
        "participation": 0.85
      }
    }
  },
  "team_rosters": {
    "Trento": ["Alessandro Michieletto", "Riccardo Sbertoli", ...]
  }
}
```

### 4.2. Carga (`load()`)

`PlayerStatsGenerator.load()` reconstruye el generador desde el JSON. Usado por la API al arrancar (`src/api/main.py`).

---

## 5. Integración en la Simulación

### 5.1. En Partido Individual

El endpoint `POST /api/simular/partido` activa el generador si `generar_stats_jugadores = true`. Al final de cada set simulado se invoca `generate_set_stats()` para ambos equipos.

Ver `prediccion_partidos.md` (sección 5) para más detalles.

### 5.2. En Temporada

El `SeasonSimulator` genera stats al final de cada partido y las acumula a lo largo de la temporada en `player_season_stats`. Al final de la simulación, cada jugador tiene stats agregadas para toda la temporada: puntos totales, aces, ataques, bloqueos, recepciones.

Ver `prediccion_temporadas.md` (sección 9.2) para más detalles.

---

## 6. Limitaciones

1. **Muestreo, no simulación**: las stats no son el resultado de una simulación play-by-play. No hay coherencia entre el marcador del set y las acciones individuales más allá de la normalización implícita por las medias históricas.

2. **Distribución normal truncada**: se asume que las stats por set siguen una distribución normal, cuando en realidad son datos de conteo discretos que se aproximan mejor con Poisson o Negative Binomial.

3. **Desviación estándar fija**: `std = max(mean * 0.4, 0.1)` es una heurística simple. No captura la variabilidad real de cada jugador (algunos son más consistentes que otros).

4. **Una temporada por equipo**: solo se usa la temporada más reciente de cada equipo para ajustar distribuciones. Jugadores que cambiaron de equipo o temporada no tienen representación en su nuevo contexto.

5. **Sin correlación entre stats**: las stats de un jugador se muestrean independientemente. En realidad, un jugador que tiene un gran partido en ataque probablemente también tenga más aces o bloqueos.

6. **IDs de equipo frágiles**: el mapa de IDs de CSV a nombres canónicos está hardcodeado. Si se añade un equipo nuevo, hay que actualizar `_extract_team_name()` manualmente.

7. **Sin diferenciación por posición**: opuesto, central, colocador y líbero tienen distribuciones de stats muy diferentes, pero el generador trata a todos los jugadores con el mismo proceso.

---

## 7. Conclusión

El `PlayerStatsGenerator` es un generador ligero de estadísticas sintéticas que utiliza distribuciones estadísticas pre-ajustadas para producir perfiles de jugadores realistas en partidos simulados. Su propósito es decorativo: añade riqueza a la salida de la simulación sin necesidad de un motor de simulación física de cada acción. Las stats generadas son consistentes a nivel agregado (un buen anotador promedia más puntos) pero no capturan correlaciones entre acciones ni la dinámica real de un set de voleibol. Es una solución pragmática para un proyecto de TFG que prioriza la simulación a nivel de equipo sobre el detalle individual.
