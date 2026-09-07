# Capa de Datos

> **⚠️ ACTUALIZACIÓN 2026-07-08.** Se añadió `src/data/rolling_features.py`,
> que reconstruye las features a nivel de partido desde `sets_partidos.csv`
> **sin leakage temporal** (recorrido cronológico, solo información previa) e
> incorpora un **Elo con margen de victoria**. Reemplaza en producción a las
> features enriquecidas (`enrich_with_team_stats`, `compute_roster_features`),
> que tenían leakage de temporada completa. Expone `get_historical_team_elo` y
> `get_historical_team_strengths`, usadas por el API para sembrar fuerzas y Elo
> runtime. Contexto completo en
> [`mejora_precision_2026-07.md`](mejora_precision_2026-07.md).

## Descripción

La capa de datos es el cimiento del sistema. Se encarga de cargar, limpiar, normalizar y preparar los 22 archivos CSV de la carpeta `DB/`, que contienen datos históricos de la SuperLega italiana desde 2014 hasta 2024. Estos datos alimentan tanto el entrenamiento de los modelos ML como, indirectamente, la simulación en tiempo real (a través de los perfiles estáticos de equipo cargados por el `RuntimeFeatureBuilder`).

*Código: `src/data/data_pipeline.py` (275 líneas), `src/data/feature_store.py` (356 líneas), `src/data/team_mapper.py` (258 líneas), `src/data/team_id_mapper.py` (mapa `ID_Equipo → canónico` para pre-temporada features como roster churn), `src/data/rolling_features.py` (246 líneas, produce las features de partido vía Elo cronológico + EWMA + H2H desde `sets_partidos.csv`), `src/data/team_sideout.py` (109 líneas, sideout por equipo proxy), `src/data/roster_continuity.py` (computa la fracción de puntos T−1 anotados por jugadores que siguen en T, **B5 NEGATIVE**, no se integra a producción)*

---

## 1. Arquitectura General

La capa de datos se compone de tres módulos con responsabilidades bien definidas:

```
                    ┌─────────────────────────────────────┐
                    │         team_mapper.py               │
                    │  Normalización de nombres de equipos │
                    │  TEAM_ALIASES → _ALIAS_LOOKUP        │
                    │  get_all_viable_teams()              │
                    └──────────┬──────────────────────────┘
                               │ nombres canónicos
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    data_pipeline.py                           │
│                                                              │
│  DB/sets_partidos.csv        → load_sets_partidos()          │
│  DB/features/match_features.csv  → load_match_features()     │
│  DB/features/set_features.csv    → load_set_features()       │
│  DB/enfrentamientos_directos/*.csv → load_enfrentamientos()  │
│  DB/Comparacion_equipos_10_años.csv → load_team_season_stats()│
│  DB/stats_por_equipo_completo/*.csv → load_player_stats()    │
│                                                              │
│  run_pipeline() → dict[str, pd.DataFrame]                    │
└──────────────────────┬───────────────────────────────────────┘
                       │ DataFrames limpios
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  feature_store.py                             │
│                                                              │
│  prepare_match_data()  → splits train(2016-22)/val(2023)/    │
│                           test(2024) para partidos           │
│  prepare_set_data()    → splits temporales para sets         │
│  enrich_with_team_stats() → añade stats de equipo por temp.  │
│  compute_roster_features() → añade features de roster        │
│  save_splits() / load_splits() → cache en models/feature_cache│
└──────────────────────────────────────────────────────────────┘
```

| Módulo | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Normalización | `team_mapper.py` | Unificar nombres de equipos (MonzaMonza → Monza) |
| Mapeo ID | `team_id_mapper.py` | Mapa `ID_Equipo → canónico` (22 entradas); evita la colisión LT↔CIS-VOLLEY y PC↔PIACENZAYOU al calcular features pre-temporada como roster churn |
| Pipeline | `data_pipeline.py` | Cargar y limpiar los 6 grupos de CSVs |
| Features offline | `feature_store.py` | Construir splits temporales y enriquecer features |
| Features rolling | `rolling_features.py` | Features de partido sin leakage (Elo+EWMA+H2H) desde `sets_partidos.csv` |
| Sideout proxy | `team_sideout.py` | Sideout estimado por equipo a partir de datos históricos |
| Roster continuity | `roster_continuity.py` | Continuidad T−1 → T por equipo (B5 NEGATIVE; queda como artefacto documentado, no integrado a producción) |

---

## 2. `data_pipeline.py` — Pipeline de Carga y Limpieza

El pipeline se ejecuta con `python -m src.data.data_pipeline`. Lee seis grupos de fuentes, normaliza los nombres de equipos con `team_mapper.py`, calcula columnas derivadas y devuelve un diccionario con los DataFrames resultantes.

### 2.1. `load_sets_partidos()`

Lee `DB/sets_partidos.csv`. Contiene los marcadores punto a punto de cada set de la SuperLega (2014-2024).

| Operación | Descripción |
|---|---|
| Normalizar nombres | Aplica `normalize_team_name()` a `equipo_local` y `equipo_visitante` |
| Extraer temporada | Convierte `"2024/2025"` → `2024` (año de inicio) |
| Ganador del set | Columna derivada: si `ganador_set_local == 1` → `equipo_local`, si no → `equipo_visitante` |
| Diferencia de puntos | `diff_puntos_set = puntos_local - puntos_visitante` |

### 2.2. `load_match_features()`

Lee `DB/features/match_features.csv`. Contiene features pre-calculadas a nivel de partido (win rates, Elo, ratios, etc.). Es la tabla principal para entrenar el `MatchPredictor`. Contiene una fila por partido.

| Operación | Descripción |
|---|---|
| Normalizar nombres | Aplica `normalize_team_name()` a `local` y `visitante` |
| Extraer temporada | Convierte `"2024/2025"` → `2024` |
| Asegurar tipo | `gana_local` se convierte a `int` |

### 2.3. `load_set_features()`

Lee `DB/features/set_features.csv`. Contiene dos filas por set: la primera con datos **pre-set** (para predecir), la segunda con datos **post-set** (con los resultados del set como verdad terreno). El pipeline filtra solo las filas pre-set.

| Operación | Descripción |
|---|---|
| Extraer temporada | Desde `partido_id` (formato: `"2024/2025_1"`) |
| Separar pre/post | Usa `groupby(["partido_id", "set_num"]).cumcount()`: índice 0 = pre-set, índice 1 = post-set |
| Asegurar tipo | `ganador_set_local` → `int` |

### 2.4. `load_enfrentamientos_directos()`

Lee todos los CSVs con patrón `DB/enfrentamientos_directos/enfrentamientos_directos_*.csv`. Unifica los archivos y calcula el resultado del partido (`gana_local`, `resultado`).

### 2.5. `load_team_season_stats()`

Lee `DB/Comparacion_equipos_10_años.csv`. Contiene estadísticas agregadas por equipo y temporada: ataques, recepciones, saques, bloqueos. Renombra las columnas (originalmente con nombres compuestos como `ATTACK_Exc. %`) a nombres legibles y calcula métricas derivadas:

| Métrica derivada | Fórmula |
|---|---|
| `puntos_por_set` | `puntos_totales / sets_jugados` |
| `aces_ratio` | `aces / saques_totales` |
| `ataque_eficacia` | `ataques_ganados / ataques_totales` |
| `recepcion_eficacia` | `recepciones_excelentes / recepciones_totales` |

### 2.6. `load_player_stats()`

Lee todos los CSVs con patrón `DB/stats_por_equipo_completo/*_historial_10_años.csv`. Son estadísticas individuales de jugadores por equipo y temporada.

| Operación | Descripción |
|---|---|
| Separar totales | Detecta filas `"Team Totals"` y las excluye |
| Renombrar columnas | `Player_Player` → `jugador`, `POINTS_Tot` → `puntos`, etc. |
| Convertir numéricas | 8 columnas numéricas con `pd.to_numeric(..., errors="coerce")` |
| Calcular por set | `puntos_por_set`, `aces_por_set` |

### 2.7. `run_pipeline()`

Ejecuta las 6 funciones de carga en orden e imprime un resumen con filas/columnas de cada DataFrame.

```
============================================================
PIPELINE DE DATOS — SuperLega Volleyball Simulator
============================================================

[1/6] Cargando sets_partidos...
[2/6] Cargando match_features...
[3/6] Cargando set_features...
...
============================================================
RESUMEN DEL PIPELINE
============================================================
  sets                   →  12345 filas,   8 columnas
  match_features         →   2345 filas,  87 columnas
  ...
```

---

## 3. `feature_store.py` — Gestión de Features y Splits Temporales

### 3.1. Split Temporal

Se usa un split temporal estricto para evitar fuga de datos futuros:

| Split | Temporadas | Uso |
|---|---|---|
| **Train** | 2016, 2017, 2018, 2019, 2020, 2021, 2022 | Entrenamiento |
| **Val** | 2023 | Validación (selección de hiperparámetros) |
| **Test** | 2024 | Evaluación final (datos no vistos) |

```python
TEMPORAL_SPLITS = {
    "train": [2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "val": [2023],
    "test": [2024],
}
```

**IMPORTANTE**: este split es estrictamente temporal. Nunca se barajan temporadas ni se usa información futura en el entrenamiento.

### 3.2. Features base para partido (57 columnas)

`MATCH_FEATURE_COLS` define 57 features base, que se agrupan en categorías:

| Categoría | Features | Descripción |
|---|---|---|
| Win rates | `h_win_rate_global`, `h_win_rate_last5`, `h_win_rate_home`, etc. | Rendimiento histórico del equipo |
| Diferencias | `diff_win_rate_global`, `diff_win_rate_last5` | Brecha entre local y visitante |
| Set metrics | `h_set_win_rate`, `h_set_diff_exp`, `diff_set_win_rate` | Rendimiento en sets |
| Points | `h_pts_fav_exp`, `h_pts_con_exp` | Puntos esperados |
| Forma | `h_forma_home`, `h_forma_away`, `diff_forma_efectiva` | Forma reciente con desglose local/visitante |
| H2H | `h_h2h_win_rate`, `h_h2h_set_diff_exp` | Historial de enfrentamientos directos |
| Momentum | `h_racha`, `a_racha`, `diff_ultimo_set_diff` | Rachas y diferencia del último set |
| Descanso | `h_descanso`, `a_descanso`, `diff_descanso` | Días de descanso |
| Ranking | `h_rank_season`, `a_rank_season`, `diff_rank_season` | Posición en la tabla |
| Elo | `elo_h`, `elo_a`, `elo_diff`, `elo_win_prob_h`, `elo_h_home`, `elo_a_away` | Sistema de rating Elo |
| Ratios | `set_ratio_h`, `set_ratio_a`, `diff_set_ratio`, `point_ratio_h`, `point_ratio_a`, `dominancia_h`, `dominancia_a`, `diff_dominancia` | Ratios de rendimiento |
| SOS | `sos_h`, `sos_a`, `diff_sos` | Strength of schedule |
| Jornada | `jornada_num` | Número de jornada |

A estas 57 se suman:
- **21 features enriquecidas** de `enrich_with_team_stats()` (7 stats de equipo × 3 columnas: local, visitante, diferencia)
- **9 features de roster básico** de `compute_roster_features()`: top_scorer_avg, roster_depth, ace_threat + diferencias local-visitante
- **87 total** para el MatchPredictor legacy (57+21+9)

### 3.3. Features para SetPredictor (21 columnas)

`SET_FEATURE_COLS` define 21 features que capturan el contexto del set:

| Feature | Descripción |
|---|---|
| `strength_h`, `strength_a`, `strength_diff` | Fuerza de equipo (pre-match) |
| `elo_diff` | Diferencia de Elo |
| `set_wr_h`, `set_wr_a`, `diff_set_wr` | Win rate en sets durante la temporada |
| `forma_h`, `forma_a`, `diff_forma` | Forma reciente |
| `pts_fav_h`, `pts_fav_a` | Puntos a favor en el partido actual |
| `h2h_diff` | Historial de enfrentamientos directos |
| `diff_set_ratio`, `diff_dominancia` | Ratios derivados |
| `set_num_norm` | Número de set normalizado (0..1) |
| `sets_h_antes`, `sets_a_antes`, `diff_sets_antes` | Sets ganados antes de este set |
| `momentum_h` | Momentum del local en el set actual |
| `es_desempate` | Flag: 1 si es 5º set |

### 3.4. Enriquecimiento con Team Stats

`enrich_with_team_stats(match_df, team_stats)` cruza `match_features` con `Comparacion_equipos_10_años.csv` para añadir stats de equipo por temporada:

```python
stat_cols = {
    "puntos_por_set": "pts_set",
    "aces_por_set": "aces_set",
    "pct_ataque": "atq_pct",
    "ataque_eficacia": "atq_eff",
    "recepcion_eficacia": "rec_eff",
    "bloqueos_por_set": "bloq_set",
    "aces_ratio": "ace_ratio",
}
```

Para cada columna se crean tres versiones: `h_{short_name}`, `a_{short_name}`, `diff_{short_name}`.

### 3.5. Enriquecimiento con Roster Features

`compute_roster_features(match_df, player_stats)` calcula métricas agregadas del roster de cada equipo a partir de las estadísticas históricas de jugadores:

| Feature | Cálculo | Significado |
|---|---|---|
| `top_scorer_avg` | Max(puntos/set entre todos los jugadores) | Mejor anotador |
| `roster_depth` | Entropía de la distribución de puntos | Cuán repartido está el ataque |
| `ace_threat` | Max(aces/set) | Mejor sacador |
| `block_power` | Mean(bloqueos/set) de top 3 bloqueadores | Potencia de bloqueo |
| `rec_quality` | Max(% recepción excelente) de jugadores con >10 recepciones | Mejor receptor |

Las `ROSTER_BASIC_COLS` (9 columnas: top_scorer, depth, ace_threat + diferencias local-visitante) son las que se usan en el entrenamiento oficial del MatchPredictor legacy. Las `ROSTER_FULL_COLS` (6 adicionales: block_power, rec_quality + diferencias, total 15) se usan solo en benchmarks para evaluar si añadir bloqueo/recepción mejora el modelo.

### 3.6. Nota sobre el camino activo de producción

`enrich_with_team_stats()` y `compute_roster_features()` **no están en el camino activo de producción**. Permanecen en el código y se usan en `benchmark_roster.py` y en el `match_predictor.joblib` legacy (fallback en disco), pero la ruta de producción usa `rolling_features.build_rolling_match_features()` desde `sets_partidos.csv`, que produce features sin leakage (Elo+EWMA+H2H).

### 3.7. Persistencia

Los splits se guardan en `models/feature_cache/` para reutilización rápida:

```
models/feature_cache/
├── match_X_train.csv  /  match_y_train.csv
├── match_X_val.csv    /  match_y_val.csv
├── match_X_test.csv   /  match_y_test.csv
├── set_X_train.csv    /  set_y_train.csv
├── set_X_val.csv      /  set_y_val.csv
└── set_X_test.csv     /  set_y_test.csv
```

---

## 4. `team_mapper.py` — Normalización de Nombres de Equipos

### 4.1. Problema

Cada fuente de datos usa nombres distintos para el mismo equipo: lo que en un CSV es `"Monza"`, en otro es `"MonzaMonza"` (nombre duplicado), y en un tercero `"Gi Group Monza"` (con patrocinador). El `team_mapper` unifica todos los alias a un nombre canónico.

### 4.2. Diccionario Maestro

`TEAM_ALIASES` mapea nombre canónico → lista de alias. Cada equipo tiene entre 1 y 6 alias registrados:

```python
TEAM_ALIASES = {
    "Modena": ["Modena", "Azimut Modena", "Leo Shoes Modena", "ModenaModena", ...],
    "Trento": ["Trento", "Trentino", "Diatec Trentino", "Itas Trentino", ...],
    "Perugia": ["Perugia", "Sir Safety Conad Perugia", "Sir Susa Vim Perugia", ...],
    ...
}
```

Hay **22 equipos canónicos** en total, incluyendo tanto los 12 de la temporada actual como equipos históricos (Siena, Ravenna, Cuneo, etc.).

### 4.3. Algoritmo de `normalize_team_name()`

```python
def normalize_team_name(raw_name: str) -> str:
    # 1. Búsqueda directa en _ALIAS_LOOKUP
    # 2. Detectar nombres duplicados (MonzaMonza → Monza)
    # 3. Búsqueda por subcadena (alias contenido en name o viceversa)
    # 4. Fallback: devolver tal cual (sin normalizar)
```

El paso 2 usa `_try_dedup()` que detecta cadenas donde la primera mitad coincide con la segunda mitad: `"MonzaMonza"` → detecta que `"Monza" * 2 == "MonzaMonza"` y devuelve `"Monza"`.

### 4.4. Equipos por Temporada

`get_superliga_teams(season)` devuelve los equipos que jugaron en la SuperLega (no Serie A2) para cada temporada desde 2019/2020 hasta 2024/2025:

| Temporada | Equipos | Nota |
|---|---|---|
| 2024/2025 | 12 | Trento, Perugia, Piacenza, Verona, Lube, Milano, Modena, Monza, Cisterna, Padova, Taranto, Grottazzolina |
| 2023/2024 | 12 | Idem |
| 2022/2023 | 12 | Acicastello en lugar de Grottazzolina |
| 2021/2022 | 12 | Cisterna Top Volley, Siena |
| 2020/2021 | 13 | Ravenna, Vibo Valentia, Taranto |
| 2019/2020 | 13 | Ravenna, Vibo Valentia, Sora |

### 4.5. Equipos Viables

`get_all_viable_teams()` devuelve 16 equipos que cumplen tres condiciones:
- **≥20 partidos** en `match_features.csv`
- **Stats de equipo** en `Comparacion_equipos_10_años.csv`
- **Datos de jugadores** en `stats_por_equipo_completo/`

De estos 16, 12 son "actuales" (temporada 2024/2025) y 4 son "históricos" (Siena, Ravenna, Acicastello, Cuneo).

```python
_ALL_VIABLE_TEAMS = [
    # Actuales (2024/2025)
    "Trento", "Perugia", "Piacenza", "Verona", "Lube",
    "Milano", "Modena", "Monza", "Cisterna", "Padova",
    "Taranto", "Grottazzolina",
    # Históricos
    "Siena", "Ravenna", "Acicastello", "Cuneo",
]
```

---

## 5. Limitaciones

1. **Sin normalización estandarizada**: `normalize_team_name()` usa búsqueda por subcadena como último recurso, lo que puede producir falsos positivos si un alias es subcadena de otro nombre no relacionado.

2. **Dependencia de CSV con nombres compuestos**: las columnas de `Comparacion_equipos_10_años.csv` tienen nombres como `"ATTACK_Exc. %"` con espacios, puntos y caracteres especiales. Si el CSV cambia de formato, el renombrado en `load_team_season_stats()` se rompe.

3. **IDs de equipo hardcodeados**: `_extract_team_name()` en `player_stats_generator.py` tiene un mapa fijo de 20 IDs de CSV a nombres canónicos. Nuevos equipos requieren actualización manual.

4. **Dos equipos "Cisterna"**: hay `"Cisterna"` (2024/2025) y `"Cisterna Top Volley"` (equipo históricamente distinto, antes llamado Latina). Comparten alias y `normalize_team_name("Cisterna Volley")` podría mapear incorrectamente.

5. **Sin validación post-pipeline**: no hay tests que verifiquen que todos los equipos en los CSVs se normalizan correctamente, ni que los splits tienen la distribución esperada de clases.

6. **Carga completa en memoria**: `run_pipeline()` carga todos los datos en memoria. Con 10 años de datos (~5000 partidos, ~50000 sets, ~800 jugadores) no es problema, pero escalaría mal con datasets más grandes.

---

## 6. Conclusión

La capa de datos resuelve el problema central de la heterogeneidad de fuentes: unifica nombres de equipos con `team_mapper.py`, carga y limpia los 22 CSVs con `data_pipeline.py`, y construye los splits temporales y features con `feature_store.py`. La arquitectura de tres módulos separa bien las responsabilidades, y el sistema de alias permite manejar un ecosistema de datos donde cada fuente nombra los equipos de forma diferente. La principal deuda técnica es la ausencia de validación automática de la normalización y la fragilidad de los parsers de CSV con nombres de columna no estandarizados.
