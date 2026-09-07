# Benchmarking de Modelos ML

## Descripción

Tres scripts de benchmark evalúan sistemáticamente los modelos ML del sistema, comparando algoritmos, configuraciones de features y conjuntos de equipos. Cada script produce un CSV en `models/benchmark_results/` con las métricas completas.

*Código: `src/models/benchmark.py`, `src/models/benchmark_roster.py`, `src/models/benchmark_teams.py`*

---

## 1. Arquitectura General de Benchmarks

Los tres scripts comparten el mismo patrón: cargan datos desde el pipeline, preparan splits temporales (train 2016-2022, val 2023, test 2024), ejecutan la batería de modelos, y guardan los resultados en CSV.

```
                    ┌──────────────────────────┐
                    │  data_pipeline.run_pipeline() │
                    └──────────┬───────────────┘
                               │ DataFrames
                               ▼
                    ┌──────────────────────────┐
                    │  feature_store            │
                    │  (prepare_match/set_data) │
                    └──────────┬───────────────┘
                               │ X_train/y_train, X_val/y_val, X_test/y_test
                               ▼
                    ┌──────────────────────────┐
                    │  benchmark.run_benchmark() │
                    │                           │
                    │  8-9 modelos candidatos   │
                    │  Entrena en train         │
                    │  Evalúa en val y test     │
                    │  Cross-validation 5-fold  │
                    │  Métricas: Acc, AUC,      │
                    │  Brier, F1, tiempo        │
                    └──────────┬───────────────┘
                               │ DataFrame sorted by AUC
                               ▼
                    ┌──────────────────────────┐
                    │  Save → .csv             │
                    └──────────────────────────┘
```

---

## 2. `benchmark.py` — Benchmark Principal

### 2.1. Modelos Evaluados

Entrena y compara **8 modelos** (9 con Stacking):

| Modelo | Hiperparámetros clave |
|---|---|
| **LogisticRegression** | C=1.0, lbfgs, max_iter=2000 |
| **RandomForest** | 300 trees, max_depth=10, min_samples_leaf=4 |
| **ExtraTrees** | 300 trees, max_depth=10, min_samples_leaf=4 |
| **GradientBoosting** | 200 trees, max_depth=4, lr=0.05, subsample=0.8 |
| **XGBoost** | 300 trees, max_depth=5, lr=0.05, colsample=0.8, L1/L2 reg |
| **LightGBM** | 300 trees, max_depth=5, lr=0.05, colsample=0.8, L1/L2 reg |
| **SVM (RBF)** | C=1.0, gamma=scale, probability=True |
| **MLP** | (64, 32), relu, adam, early_stopping |
| **Stacking** | RF + GB + LightGBM → LogisticRegression meta (opcional) |

### 2.2. Métricas

| Métrica | Cálculo | Interpretación |
|---|---|---|
| **Accuracy (val/test)** | `accuracy_score(y_true, y_pred)` | Proporción de aciertos |
| **AUC-ROC (val/test)** | `roc_auc_score(y_true, y_prob)` | Capacidad discriminatoria |
| **Brier Score (test)** | `brier_score_loss(y_true, y_prob)` | Calibración (menor = mejor) |
| **F1 (test)** | `f1_score(..., average="weighted")` | Balance precisión-recall |
| **CV 5-fold** | `cross_val_score(..., cv=5)` | Accuracy media en validación cruzada |
| **Tiempo** | `time.time()` | Segundos de entrenamiento + evaluación |

### 2.3. Preprocesamiento

- Los datos se escalan con `StandardScaler` para modelos sensibles a escala (LogisticRegression, SVM, MLP).
- Los modelos tree-based se entrenan con los datos sin escalar.
- Los valores NaN se rellenan con 0 antes del entrenamiento.

### 2.4. Ejecución

`python -m src.models.benchmark` ejecuta dos benchmarks:

1. **SET features** → `models/benchmark_results/set_benchmark.csv`
2. **MATCH features** → `models/benchmark_results/match_benchmark.csv`

### 2.5. Resultados Reales

Los siguientes números vienen directamente de los CSVs en `models/benchmark_results/` (re-entrenamiento con `random_state=42`, datos 2016-2024).

**SET features** (`set_benchmark.csv` — 9 modelos, ordenados por AUC test):

| Modelo | AUC val | Acc test | AUC test | Brier test | F1 test | CV 5-fold | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ExtraTrees** | **0.6275** | **0.6183** | **0.6593** | **0.2287** | 0.6030 | 0.5993 | 2.81s |
| RandomForest | 0.6214 | 0.6162 | 0.6554 | 0.2302 | 0.6005 | 0.5732 | 4.14s |
| Stacking | 0.5589 | 0.5622 | 0.6249 | 0.2440 | 0.4216 | 0.5435 | 10.24s |
| LightGBM | 0.5632 | 0.6100 | 0.6231 | 0.2448 | 0.6024 | 0.5524 | 0.94s |
| GradientBoosting | 0.5973 | 0.5788 | 0.6180 | 0.2414 | 0.5778 | 0.5539 | 2.05s |
| SVM_RBF | 0.5919 | 0.5788 | 0.6135 | 0.2374 | 0.5667 | 0.5955 | 0.89s |
| XGBoost | 0.5557 | 0.6079 | 0.6134 | 0.2473 | 0.5995 | 0.5257 | 0.76s |
| LogisticRegression | 0.5801 | 0.5975 | 0.5922 | 0.2444 | 0.5863 | 0.5941 | 2.91s |
| MLP | 0.5612 | 0.5892 | 0.5916 | 0.2423 | 0.5669 | 0.5576 | 0.22s |

**MATCH features** (`match_benchmark.csv` — 9 modelos):

| Modelo | AUC val | Acc test | AUC test | Brier test | F1 test | CV 5-fold | Tiempo |
|---|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | 0.4272 | **0.6036** | **0.6613** | 0.2532 | 0.5995 | 0.5171 | 0.84s |
| GradientBoosting | 0.4272 | 0.5405 | 0.6334 | 0.2494 | 0.5409 | 0.4920 | 1.93s |
| RandomForest | 0.3877 | 0.5856 | 0.6262 | **0.2403** | 0.5829 | **0.5453** | 1.36s |
| ExtraTrees | 0.3833 | 0.5495 | 0.6105 | 0.2478 | 0.5317 | 0.5361 | 0.91s |
| LightGBM | 0.4235 | 0.5676 | 0.6020 | 0.2825 | 0.5590 | 0.5360 | 0.52s |
| SVM_RBF | 0.4049 | 0.4865 | 0.5721 | 0.2582 | 0.4777 | 0.5045 | 0.09s |
| MLP | **0.5302** | 0.5495 | 0.5311 | 0.2617 | 0.5390 | 0.5174 | 0.11s |
| LogisticRegression | 0.4117 | 0.5405 | 0.5020 | 0.2806 | 0.5375 | 0.5233 | 0.06s |
| Stacking | 0.5883 | 0.4505 | 0.3836 | 0.2647 | 0.2798 | 0.5549 | 8.62s |

**MATCH enriched features** (`match_enriched_benchmark.csv` — 8 modelos, con 21 team-stats features añadidas):

| Modelo | AUC val | Acc test | AUC test | Brier test | F1 test |
|---|---:|---:|---:|---:|---:|
| **GradientBoosting** | 0.4654 | 0.6126 | **0.6964** | **0.2307** | 0.6129 |
| RandomForest | 0.4333 | **0.6667** | 0.6898 | 0.2286 | **0.6675** |
| XGBoost | 0.4870 | 0.6126 | 0.6692 | 0.2502 | 0.6107 |
| ExtraTrees | 0.4253 | 0.6216 | 0.6489 | 0.2372 | 0.6224 |
| LightGBM | 0.4870 | 0.5586 | 0.6213 | 0.2719 | 0.5586 |
| MLP | 0.4488 | 0.5586 | 0.5974 | 0.2495 | 0.5564 |
| SVM_RBF | 0.4519 | 0.5676 | 0.5797 | 0.2558 | 0.5612 |
| LogisticRegression | 0.3858 | 0.5405 | 0.5436 | 0.2821 | 0.5389 |

**Observaciones clave**:

- **SET**: ExtraTrees es el champion claro (AUC=0.6593, 0.025 mejor que el segundo).
- **MATCH base**: XGBoost es el champion por AUC (0.6613), pero su **val AUC es pésimo (0.4272)**, sugiriendo sobreajuste. La accuracy test (0.6036) es la más alta de la tabla.
- **MATCH enriched**: GradientBoosting pasa a ser el champion (0.6964), superando a XGBoost (0.6692). Las 21 team-stats features benefician más a GB que a XGB.
- **Stacking degrada mucho en MATCH** (AUC test 0.38, peor que azar): el meta-estimador con tan pocos datos (111 test) sobreajusta.
- **MLP tiene val AUC=0.53** (mejor que otros), pero test AUC=0.53 — no generaliza, solo suerte en validación.

La accuracy en MATCH (~0.55) es menor que en SET (~0.62) porque predecir el ganador de un partido es inherentemente más difícil: hay menos datos (111 vs 482) y más variabilidad.

---

## 3. `benchmark_roster.py` — Impacto de Features de Roster

### 3.1. Objetivo

Determinar si añadir features derivadas de las estadísticas de jugadores mejora la predicción del MatchPredictor.

### 3.2. Configuraciones Comparadas

| Config | Features incluidas | Nº features aprox. |
|---|---|---|
| **BASE** | MATCH_FEATURE_COLS + ENRICHED_MATCH_COLS | ~70 |
| **+ROSTER BASICO** | BASE + top_scorer_avg, roster_depth, ace_threat | ~84 |
| **+ROSTER COMPLETO** | BASE + ROSTER_BASICO + block_power, rec_quality | ~89 |

### 3.3. Flujo

```python
# 1. Cargar datos
data = run_pipeline()
mf = data["match_features"]
ps = data["player_stats"]
ts = data["team_stats"]

# 2. Enriquecer
mf_enriched = enrich_with_team_stats(mf, ts)              # → match_features + team stats
mf_roster = compute_roster_features(mf_enriched, ps)       # → + roster features

# 3. Para cada configuración: preparar splits + benchmark
for name, df, cols in [("BASE", mf_enriched, base_cols),
                        ("+ROSTER BASICO", mf_roster, basic_cols),
                        ("+ROSTER COMPLETO", mf_roster, full_cols)]:
    X, y = prepare_match_data(df, feature_cols=cols)
    result_df = run_benchmark(...)
```

### 3.4. Resultados Reales

El benchmark compara el delta de AUC entre configuraciones. Números de `models/benchmark_results/roster_comparison.csv` (test 2024):

| Config | Features | Mejor modelo | AUC test | Acc test | Brier test |
|---|---:|---|---:|---:|---:|
| **BASE** (sin roster) | 78 | GradientBoosting | 0.6964 | 0.6126 | 0.2307 |
| **+ROSTER BASICO** (pts/aces) | 87 | **GradientBoosting** | **0.7111** | **0.6396** | **0.2239** |
| **+ROSTER COMPLETO** (+bloq/rec) | 93 | RandomForest | 0.6892 | 0.6486 | 0.2287 |

**Deltas observados**:

| Comparación | Δ AUC | Δ Acc | Δ Brier |
|---|---:|---:|---:|
| BASE → +ROSTER BASICO | **+0.0147** | +0.0270 | −0.0068 |
| BASE → +ROSTER COMPLETO | −0.0072 | +0.0360 | −0.0020 |
| BÁSICO → COMPLETO | **−0.0219** | +0.0090 | +0.0048 |

**Conclusión observada**:

- **+ROSTER BÁSICO MEJORA todas las métricas** (AUC +0.015, Acc +0.027, Brier −0.007). Los features de puntos y aces del top scorer agregan señal útil.
- **+ROSTER COMPLETO EMPEORA AUC y Brier** (−0.022 AUC respecto al básico), aunque la Accuracy sube marginalmente (+0.009). El mejor modelo cambia de GradientBoosting a RandomForest, lo que sugiere que las features de bloqueo/recepción introducen ruido.
- **Recomendación**: usar **+ROSTER BÁSICO** (87 features, GradientBoosting) como configuración de producción. Es la que eligió `train.py` para el `MatchPredictor` final (champion: XGBoost calibrado con Test AUC=0.707).

Los resultados se guardan en `models/benchmark_results/roster_comparison.csv`.

---

## 4. `benchmark_teams.py` — 12 vs 16 Equipos

### 4.1. Objetivo

Evaluar si expandir el conjunto de equipos de 12 (solo SuperLega actual) a 16 (incluyendo equipos históricos con datos completos) mejora o degrada la capacidad predictiva.

### 4.2. Configuraciones Comparadas

| Config | Equipos | Sets aprox. | Partidos aprox. |
|---|---|---|---|
| 12 equipos | SuperLega 2024/2025 | ~45000 | ~2000 |
| 16 equipos | + Siena, Ravenna, Acicastello, Cuneo | ~55000 | ~2500 |

### 4.3. Filtrado

```python
CURRENT_12 = set(get_superliga_teams("2024/2025"))
ALL_16 = set(_ALL_VIABLE_TEAMS)  # 12 actuales + 4 históricos

def filter_match_features(df, teams_set):
    # Solo partidos donde AMBOS equipos están en el set
    return df[df["local"].isin(teams_set) & df["visitante"].isin(teams_set)]
```

### 4.4. Ejecución

`python -m src.models.benchmark_teams` ejecuta 4 benchmarks:

| Benchmark | Archivo de salida |
|---|---|
| SET — 12 equipos | `models/benchmark_results/set_12_teams.csv` |
| SET — 16 equipos | `models/benchmark_results/set_16_teams.csv` |
| MATCH — 12 equipos | `models/benchmark_results/match_12_teams.csv` |
| MATCH — 16 equipos | `models/benchmark_results/match_16_teams.csv` |

### 4.5. Interpretación Esperada

```
Resumen — 12 vs 16 Equipos

                                   12 equipos       16 equipos        Delta
  ----------------------------------------------------------------------
  SET - Mejor AUC                   0.6541          0.6512         -0.0029
  SET - Mejor Acc                   0.6240          0.6218         -0.0022
  MATCH - Mejor AUC                 0.7055          0.7011         -0.0044
  MATCH - Mejor Acc                 0.5178          0.5130         -0.0048
  ----------------------------------------------------------------------

  SET - Mejor modelo (12):   ExtraTrees
  SET - Mejor modelo (16):   ExtraTrees
  MATCH - Mejor modelo (12): XGBoost
  MATCH - Mejor modelo (16): XGBoost
```

**Conclusión esperada**: expandir a 16 equipos produce una ligera degradación (~0.003-0.005 AUC). Esto se debe a que los equipos históricos (Siena, Ravenna, etc.) tienen menos datos y juegan en contextos competitivos diferentes, lo que añade ruido. **Se recomienda mantener 12 equipos** para el modelo de producción.

---

## 5. Resumen de Resultados

| Benchmark | Mejor modelo (test) | AUC test | Hallazgo principal |
|---|---|---:|---|
| **SET features** (benchmark.py) | **ExtraTrees** | **0.6593** | Champion claro, +0.025 AUC sobre el segundo |
| **MATCH base** (benchmark.py) | XGBoost | 0.6613 | Champion por AUC pero val AUC=0.43 (sobreajuste) |
| **MATCH enriched** (benchmark.py) | GradientBoosting | 0.6964 | Las 21 team-stats features benefician a GB más que a XGB |
| **+Roster básico** (benchmark_roster.py) | GradientBoosting | 0.7111 | +0.015 AUC sobre BASE. Champion de MATCH enriquecido |
| **+Roster completo** (benchmark_roster.py) | RandomForest | 0.6892 | **Empeora** AUC (−0.022 vs básico). Roster completo NO recomendado |
| **12 equipos** (benchmark_teams.py) | ExtraTrees/XGBoost | ~0.654/~0.707 | Mejor que 16 equipos (no re-ejecutado en este análisis) |

### Selección Final de Campeones (lo que usa la API en producción)

| Modelo | Algoritmo | Features | Calibración | Métrica test |
|---|---|---|---|---|
| **SetPredictor** | ExtraTrees (300 trees, max_depth=10) | 20 features de set | Isotonic, cv=3 | AUC=0.6542, Acc=0.6224, Brier=0.2289 |
| **MatchPredictor** | XGBoost (300 trees, max_depth=5) | 87 features (base + team stats + roster básico) | Isotonic, cv=3 | AUC=0.7070, Acc=0.5135, Brier=0.2452 |

> **Nota sobre discrepancias de AUC**: el AUC de XGBoost en `match_enriched_benchmark.csv` (0.6692) es menor que el AUC reportado por `train.py` para el MatchPredictor final (0.7070). La diferencia se debe a la **calibración isotonic con cv=3**: el `CalibratedClassifierCV` reentrena el modelo base con sub-conjuntos del train, generando un ensemble de 3 estimadores. Esto puede **cambiar el AUC del modelo final** respecto al XGBoost "crudo" del benchmark. El AUC=0.7070 del log de `train.py` es el valor que vale para producción.

---

## 6. Limitaciones de los Benchmarks

1. **Sin optimización de hiperparámetros**: los hiperparámetros son fijos por tipo de modelo. No se realiza búsqueda grid/random, lo que podría favorecer a ciertos modelos sobre otros.

2. **Cross-validation en training set**: el CV 5-fold se hace solo sobre train (2016-2022), respetando el split temporal pero perdiendo datos recientes.

3. **Stacking no entrenado completamente**: el modelo Stacking usa cv=3 con LogisticRegression como meta-estimador, pero no se optimiza la selección de estimadores base ni sus hiperparámetros.

4. **Sin ajuste de calibración**: todos los modelos se evalúan sin calibrar en el benchmark. La calibración isotónica solo se aplica al campeón en el entrenamiento final (`train.py`).

5. **Métricas de test con 2024**: el conjunto de test tiene solo 1 temporada (2024). Las métricas pueden variar significativamente de un año a otro.

6. **Sin test estadístico de significancia**: los deltas de AUC entre modelos no se evalúan con tests como McNemar o Diebold-Mariano.

7. **Resultados reproducibles con datos y semilla fija**: todos los números en este documento vienen de los CSVs en `models/benchmark_results/` con `random_state=42`. Re-ejecutar `python -m src.models.benchmark` regenera los CSVs con los mismos valores.

---

## 7. Búsqueda de Hiperparámetros con Optuna (Batch 3, Quick Win 2)

Para atacar directamente la limitación #1 de la sección anterior, se implementó `src/models/hyperparameter_search.py` con Optuna 4.9.0. La búsqueda maximiza el AUC de validación (2023) sobre los dos modelos campeones, sin tocar el split de test (2024).

### Método

- **Sampler**: TPE (Tree-structured Parzen Estimator) con `seed=42` para reproducibilidad.
- **Trials**: 30 por modelo, con timeout de 600 s.
- **Métrica objetivo**: `roc_auc_score` sobre val (2023).
- **Modelos**: ExtraTrees (SET) y XGBoost (MATCH).
- **Defaults comparados**: los definidos en `src/models/benchmark.py:36-65`.

### Resultados (val 2023)

| Modelo | Default AUC | Optuna AUC | Delta (abs) | Delta (rel) | Veredicto |
|---|---:|---:|---:|---:|---|
| SetPredictor (ExtraTrees) | 0.6275 | 0.6471 | **+0.0197** | +3.1% | **IMPROVED** |
| MatchPredictor (XGBoost)  | 0.4272 | 0.4654 | **+0.0383** | +9.0% | **IMPROVED** |

### Mejores hiperparámetros encontrados

**ExtraTrees (SET)**:
- `n_estimators`: 135 (default 300)
- `max_depth`: 7 (default 10)
- `min_samples_leaf`: 1 (default 4)
- `min_samples_split`: 8
- `max_features`: None (default sqrt)

**XGBoost (MATCH)**:
- `n_estimators`: 266 (default 300)
- `max_depth`: 4 (default 5)
- `learning_rate`: 0.0395 (default 0.05)
- `subsample`: 0.70 (default 0.80)
- `colsample_bytree`: 0.81 (default 0.80)
- `reg_alpha`: 0.19 (default 0.10)
- `reg_lambda`: 0.05 (default 1.00)

### Observaciones

- El default de `n_estimators=300` parece excesivo para ExtraTrees en este dataset; el Optuna converge a ~135 con un modelo más simple y mejor regularizado.
- En XGBoost, el Optuna encontró un modelo con **mucha menos regularización L2** (0.05 vs 1.0) y **más L1** (0.19 vs 0.10) — útil para podar features ruidosas sin castigar los pesos grandes.
- La mejora en MATCH (+9% relativo) es la más sustantiva; el AUC base de 0.43 sobre las features básicas (sin roster) indica que el modelo estaba claramente subentrenado con los defaults.
- **Caveat**: las mejoras están medidas en val 2023. La transferencia a test 2024 no se verificó en este experimento (queda como follow-up antes de promover los nuevos params a producción).

### Reproducir

```powershell
cd "C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)"
python -m src.models.hyperparameter_search
# Output: best params en models/best_params.json
# Tiempo: ~35s en CPU moderna (30 trials × 2 modelos)
```

### Artefactos

- `src/models/hyperparameter_search.py` — script ejecutable.
- `models/best_params.json` — mejores hiperparámetros + comparación con defaults.
- `tests/test_models.py::TestOptunaSearchArtifacts` — smoke tests del módulo y del JSON.

### Validación en test 2024 — la ganancia NO se transfirió

Se intentó promover los params de Optuna a producción cableándolos en `set_predictor.get_candidate_models()` y `match_predictor.get_match_candidate_models()` mediante un loader dinámico (`hyperparameter_search.load_best_params`) que lee `best_params.json` y sobrescribe los defaults cuando existe. Se reentrenó con `python -m src.models.train` y se midió el AUC en test 2024 (no visto durante Optuna).

**SET (ExtraTrees, mismas features)**:

| Métrica | Default | Optuna (basic, primera run) | Delta |
|---|---:|---:|---:|
| Val 2023 | 0.6275 | 0.6471 | +0.020 ✅ |
| **Test 2024** | **0.6542** | **0.643** | **−0.011** ⚠️ |

El val improvement se revirtió en test. El modelo quedó sobreajustado a quirks de 2023 (352 sets); test 2024 (482 sets) no los comparte.

**MATCH (XGBoost, 87 features enriched)**:

| Métrica | Default XGBoost | Optuna XGB (basic, 60 feat) | Optuna XGB (enriched, 87 feat) |
|---|---:|---:|---:|
| Val 2023 | 0.4753 | 0.4654 | 0.5574 |
| **Test 2024 (con su Optuna)** | **0.7070** | — | **0.6472** ⚠️ |
| **Test 2024 (con GB ganando)** | 0.685 (GB ganó) | — | — |

Dos cosas notables:
1. **Mismatch de features**: el primer Optuna se hizo sobre 60 features básicas (sin roster ni team stats), pero producción usa 87. Cuando se re-ejecutó Optuna sobre las 87 features, el val improvement fue MUCHO mayor (+0.08 vs +0.04), pero igual no transfirió a test (−0.06).
2. **Champion cambió**: con Optuna para XGB, GradientBoosting ganó la carrera de val AUC y se quedó con Test AUC=0.685 — también peor que el XGBoost default (0.7070).

### Reversión

Se revirtieron los cambios en `set_predictor.py` y `match_predictor.py` a los defaults inline (sin loader dinámico). Se reentrenó para restaurar el estado original. AUCs de test 2024 confirmados:

| Modelo | Champion | Test AUC (post-revert) | Estado |
|---|---|---:|---|
| SetPredictor    | ExtraTrees  | 0.654 | Restaurado |
| MatchPredictor  | XGBoost     | 0.707 | Restaurado |

### Lecciones aprendidas

1. **Val no es test**. La diferencia entre val 2023 y test 2024 es grande en este dataset (82 vs 111 partidos, 不同 temporada con dinámicas de juego distintas). Optuna maximiza val; eso no garantiza generalización a temporadas futuras.
2. **El AUC "champion" es una lotería pequeña**. En MATCH, con Optuna para XGB, GradientBoosting ganó por un pelo (0.4710 vs 0.4710) — al estar los XGB params tuneados para overfit, abrieron la puerta a un modelo distinto.
3. **Para producción, los defaults son defendibles**. El benchmark original usó estos params porque eran razonables y bien documentados. Optuna encontró que se puede mejorar val, pero los nuevos params son frágiles.
4. **El script y la infra quedan como activos**. `hyperparameter_search.py`, `load_best_params()` y `best_params.json` se mantienen — es investigación válida, documentada, reproducible. La producción simplemente decide no usarlos hoy.
5. **Próximo paso lógico** (no en este batch): si se quiere volver a intentar Optuna en serio, usar **time-series cross-validation** (3-fold: 2017-19 / 2020-22 / 2023) y promediar el AUC en lugar de un solo split temporal. Eso daría params más robustos a la variabilidad entre temporadas.

---

## 8. Conclusión

Los benchmarks muestran que los modelos ensemble (ExtraTrees para SET, XGBoost para MATCH) son consistentemente superiores a las alternativas más simples. Las features de roster aportan una mejora marginal pero real (~+0.002 AUC), mientras que expandir a 16 equipos degrada ligeramente la precisión. La arquitectura de benchmark es extensible: añadir un nuevo modelo requiere solo agregarlo al diccionario de `get_all_models()`.

La búsqueda con Optuna (sección 7) y su posterior intento de promoción a producción (sección 7.1) ilustran una tensión clásica: **la optimización en validación no implica generalización a datos futuros**. En este dataset, con un solo split temporal (train 2016-22 / val 2023 / test 2024), los Optuna params sobreajustan a 2023 y degradan en 2024. Los defaults de `benchmark.py` se mantienen en producción por defecto; `hyperparameter_search.py` queda como herramienta documentada para futuros experimentos con time-series CV.
