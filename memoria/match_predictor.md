# Match Predictor — Predicción del Ganador de un Partido

> **⚠️ ACTUALIZACIÓN 2026-07-08.** Todo lo que sigue describe el diseño
> original de 87 features. La auditoría de precisión (ver
> [`mejora_precision_2026-07.md`](mejora_precision_2026-07.md)) demostró que su
> **AUC=0.707 era ficticio**: producto de leakage temporal (las features de
> temporada completa —`enrich_with_team_stats`, `compute_roster_features`— se
> mezclaban en partidos de esa misma temporada) evaluado sobre un único año de
> test. Medido con rolling-origin sobre el test held-out 2025/26, el AUC real
> era **0.53** (nivel de azar).
>
> **Estado en producción (API):** la señal del `MatchPredictor` (87 features,
> leaky) **se ha sustituido por la probabilidad de un Elo con margen de
> victoria** (AUC 0.75, logloss 0.585, limpio y sin leakage,
> `src/data/rolling_features.py`) **en el `SeasonSimulator`** — que es el
> único consumidor real (calibra las fuerzas por partido). El
> `MatchSimulator.simulate_match()` directo **no usa el `MatchPredictor` para
> nada**: solo recibe las fuerzas `home_strength/away_strength` y consulta el
> `SetPredictor` para el clamp adaptativo. El artefacto
> `match_predictor.joblib` se mantiene cargado en el API solo como fallback. Ver `../models/plots/reliability_match.png` y `../models/plots/reliability_match_calibrado.png` para las curvas de calibración de esta versión legacy.
> (no se borra para no romper la carga). El texto de abajo se conserva como
> registro del diseño previo.

## Descripción

El `MatchPredictor` (`src/models/match_predictor.py`) es un clasificador binario que predice la probabilidad de que el equipo local gane un partido completo de volleyball. Es el más complejo de los tres modelos del proyecto porque integra **87 features** de tres categorías: base (Elo, forma, H2H), team stats agregadas y roster.

*Salida: P(local gana partido) ∈ [0, 1] · Usado por: `SeasonSimulator` (calibración de fuerzas antes de cada partido)*

---

## 1. Arquitectura de Features (87 features totales)

El MatchPredictor trabaja sobre `DB/features/match_features.csv` (725 partidos históricos, 10 temporadas). Las features se construyen en tres capas:

| Capa | # Features | Descripción | Fuente |
|---|---:|---|---|
| **Base** | ~50 | Elo, forma, rachas, H2H, win rate, set rate, dominancia | `match_features.csv` |
| **Team stats** | 21 | Diferencias de `pts_set`, `aces_set`, `atq_pct`, `atq_eff`, `rec_eff`, `bloq_set`, `ace_ratio` | `team_season_stats` + `Comparacion_equipos_10_años.csv` |
| **Roster básico** | 15 | `top_scorer_avg`, `roster_depth`, `ace_threat` por equipo | `player_stats` |

El script `train.py` añade las features de team stats (21) y roster (15) sobre las features base para llegar a 87. Si alguna feature falta en un partido, se rellena con 0 (`fill_value=0.0` en `season_simulator.py`).

### 1.1. Importancia de las features (top 10 del champion XGBoost)

| Rank | Feature | Importancia | Tipo |
|---:|---|---:|---|
| 1 | `point_ratio_a` | 0.0234 | Team stats |
| 2 | `diff_ace_threat` | 0.0216 | Roster |
| 3 | `h_racha` | 0.0180 | Base (racha local) |
| 4 | `elo_win_prob_h` | 0.0179 | Base (Elo) |
| 5 | `diff_atq_pct` | 0.0178 | Team stats |
| 6 | `h_top_scorer_avg` | 0.0175 | Roster |
| 7 | `diff_pts_set` | 0.0172 | Team stats |
| 8 | `h_atq_eff` | 0.0167 | Team stats |
| 9 | `elo_h_home` | 0.0166 | Base (Elo con home adv) |
| 10 | `diff_top_scorer` | 0.0163 | Roster |

**Observación**: a diferencia del `SetPredictor` (donde dominan features in-match), en el `MatchPredictor` las importancias están **más repartidas entre las 3 capas** (5 base, 4 team stats, 3 roster, 2 en otras). Esto sugiere que para predecir el resultado global de un partido, las estadísticas históricas del equipo y la calidad del roster importan tanto como la forma reciente.

---

## 2. Pipeline de Entrenamiento

```
match_features.csv (725 partidos)
   ↓ enrich: +21 team_stats features
   ↓ roster: +15 roster features
   ↓ 87 features totales
   ↓
train.py → train / val / test split
   (319 / 81 / 111)        ← split temporal estricto
   ↓
MatchPredictor.train()
   ↓ 4 modelos candidatos
   ↓ selección por AUC
   ↓ calibración isotonic
   ↓
match_predictor.joblib (~1.5 MB)
```

El split temporal (definido en `feature_store.py:25-29`) es **estricto**: train 2016-2022, val 2023, test 2024. Nunca se mezcla el orden ni se hace shuffle, para evitar leakage de datos futuros al pasado.

---

## 3. Candidatos Comparados

| Modelo | Hiperparámetros principales |
|---|---|
| GradientBoosting | `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8` |
| RandomForest | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| ExtraTrees | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| **XGBoost** (champion) | `n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0` |

A diferencia del `SetPredictor` (6 candidatos), el `MatchPredictor` evalúa solo 4 porque el dataset de partidos es **muy pequeño** (319 train) y los modelos más pesados (LightGBM) sobreajustan rápido.

---

## 4. Resultados de Validación (selección de champion)

Métricas en el set de validación (año 2023, 81 partidos, balance 44/56):

| Modelo | Acc | AUC | Brier | Prec | Rec |
|---|---:|---:|---:|---:|---:|
| GradientBoosting | 0.420 | 0.471 | 0.3440 | 0.430 | 0.420 |
| RandomForest | 0.432 | 0.451 | 0.2656 | 0.435 | 0.432 |
| ExtraTrees | 0.444 | 0.437 | 0.2612 | 0.435 | 0.444 |
| **XGBoost** | 0.494 | **0.4753** | 0.3337 | 0.503 | 0.494 |

**Modelo campeón: XGBoost** (AUC = 0.4753). Todos los modelos están cerca del azar en accuracy, pero el AUC es lo que importa para la calibración: XGBoost ordena mejor las probabilidades que los demás.

---

## 5. Resultados en Test (datos 2024, no vistos)

Evaluación del modelo calibrado sobre 111 partidos de la temporada 2024 (balance 55/45):

| Métrica | Valor |
|---|---:|
| **Accuracy** | 0.5135 |
| **AUC-ROC** | 0.7070 |
| **Brier Score** | 0.2452 |

Reporte de clasificación:

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Visitante | 0.48 | 0.92 | 0.63 | 50 |
| Local | 0.73 | 0.18 | 0.29 | 61 |
| **macro avg** | 0.61 | 0.55 | 0.46 | 111 |
| **weighted avg** | 0.62 | 0.51 | 0.44 | 111 |

**El AUC sube de 0.4753 (val) a 0.7070 (test)**. Esto es contraintuitivo pero explicable:
- El set de validación (2023) es muy pequeño (81 partidos) y muy ruidoso.
- El set de test (2024) tiene 111 partidos y muestra una distribución más cercana a la del train.

Sin embargo, el modelo tiene **recall muy bajo para el local (0.18)** y muy alto para visitante (0.92): el threshold de 0.5 está mal calibrado. Esto es un **bug de calibración** que se mitiga parcialmente con el isotonic posterior, pero queda como limitación.

---

## 6. Uso en el Simulador (Calibración de Fuerzas)

El `MatchPredictor` se usa en `SeasonSimulator` para **calibrar las fuerzas de equipo antes de cada partido** (`season_simulator.py:225`):

```python
@staticmethod
def _calibrate_strengths(h_str, a_str, p_target, damping=0.5):
    """Ajusta h_str para que su odds ratio converja al p_target del MatchPredictor."""
    p_base = h_str / (h_str + a_str)
    odds_target = p_target / (1 - p_target)
    odds_base = p_base / (1 - p_base)
    k = odds_target / odds_base
    k_damped = k ** damping      # damping=0.5 aplica raíz cuadrada
    h_new = h_str * k_damped
    h_new = max(0.05, min(0.95, h_new))
    return h_new, a_str
```

### 6.1. Lógica del damping

El `MatchPredictor` no es perfecto (AUC 0.71, no 1.0), así que aplicar la corrección completa generaría sobreajuste. Con `damping=0.5`:
- Si el predictor dice "doble de probabilidad" (k=2), solo se aplica `√2 ≈ 1.41×`.
- Si dice "10× más probable" (k=10), se aplica `√10 ≈ 3.16×`.

Es un **shrinkage** hacia las fuerzas base, que reduce la varianza del simulador.

### 6.2. Comparación con baseline (sin ML)

Comparativa en una temporada de 132 partidos con seed=42 (`memoria/prediccion_temporadas.md:368`):

| Configuración | % 3-0 | % 3-1 | % 3-2 | Observación |
|---|---:|---:|---:|---|
| Baseline (sin ML) | 65.9% | 15.9% | 18.2% | Demasiados barridos, distribución irreal |
| Con MatchPredictor | 43.9% | 29.5% | 26.5% | Distribución más realista, liga más igualada |

La calibración con MatchPredictor **reduce 22 puntos porcentuales la proporción de 3-0** y los redistribuye a 3-1 y 3-2, generando sets más competitivos. Esto valida la integración ML como útil para la simulación.

---

## 7. Persistencia

```python
# match_predictor.py:200
save_data = {
    "best_model_name": self.best_model_name,
    "best_model": self.best_model,
    "calibrated_model": self.calibrated_model,
    "feature_names": self.feature_names,
    "results": self.results,
    "test_metrics": getattr(self, "_test_metrics", None),
}
joblib.dump(save_data, path)
```

El predictor pesa **~1.5 MB** (XGBoost con 300 estimadores es mucho más compacto que ExtraTrees).

---

## 8. Limitaciones y Trabajo Futuro

1. **Dataset pequeño (319 train)**: el split temporal estricto deja pocos datos. Un fix sería usar cross-validation temporal (TimeSeriesSplit) en vez de un solo split, o juntar más temporadas.

2. **Recall muy bajo para local (0.18) en test**: el threshold de 0.5 está mal. Soluciones posibles: ajustar threshold a 0.45, usar class_weight, o entrenar con más datos de victorias locales.

3. **Damping fijo en 0.5**: el shrinkage es estático. Podría ajustarse dinámicamente — mayor damping al inicio de la temporada (cuando el MatchPredictor tiene features frías) y menor al final.

4. **Features frías al inicio de temporada**: en las primeras jornadas, las features de `elo`, `streaks`, `results` están en valores por defecto. El modelo predice ~0.5 y la calibración tiene poco efecto hasta la jornada 5-6. (Nota post-A4: la limitación análoga del `SetPredictor` se demostró en el Grupo A del plan consolidado que **no era realmente esa**: con `w=1.0` la llamada al SetPredictor se cortocircuita en runtime independientemente del estado de las features. Ver `set_predictor.md` §10.5.)

### 8.2. Damping adaptativo (Batch 3 mid-effort #3)

Para atacar la limitación #3 (damping fijo), se implementó `adaptive_damping()` en `src/simulation/constants.py` (interpolación lineal de 0.3 inicio → 0.7 final) y se agregó un parámetro `damping` opcional a `SeasonSimulator.simulate_season()` y `simulate_jornada()`. Cuando se pasa un callable como `damping`, se llama con la `jornada_num` actual para resolver el damping de esa jornada.

**Convención importante** (descubierta durante la implementación): en el código, `damping` es el **exponente** en `k_damped = k ** odds_correction` dentro de `_calibrate_strengths`. Por lo tanto:
- `damping = 0` → `k_damped = 1` → no se aplica la corrección del modelo (full shrinkage hacia 0.5)
- `damping = 1` → `k_damped = k` → se aplica la corrección completa (sin shrinkage)
- `damping = 0.5` (default actual) → mitad de cada uno

El modo adaptativo arranca con `damping=0.3` (más shrinkage cuando las features están frías) y termina en `damping=0.7` (más confianza cuando el modelo tiene features cálidas).

**A/B test** (100 MC runs por estrategia, 12 equipos SuperLega actuales, mismo seed):

| Métrica | Fixed damping (0.5) | Adaptive (0.3→0.7) | Delta |
|---|---:|---:|---:|
| 3-0% | 34.89 | 34.89 | +0.00 |
| 3-1% | 36.43 | 36.43 | +0.00 |
| 3-2% | 28.67 | 28.67 | +0.00 |
| L1 a ref 40/30/30 | 12.87 | 12.87 | tie |

**Veredicto**: **marginal / no observable effect**. Ambos runs producen distribuciones IDÉNTICAS.

**Por qué falla la hipótesis** (validado con test unitario directo en `test_season_simulator.py::TestAdaptiveDamping`):

1. **La implementación es matemáticamente correcta**: el test `test_calibrate_strengths_responds_to_damping` confirma que con `p_target=0.7`, `damping=0.7` produce `h_str=0.905` mientras que `damping=0.3` produce `h_str=0.645`. La función responde al parámetro como se espera.
2. **El problema es el modelo, no el damping**: el MatchPredictor tiene val AUC ≈ 0.51 en 2023 (esencialmente random) y test AUC ≈ 0.71. Después de la calibración isotonic, las predicciones en partidos típicos están concentradas cerca de 0.5. En `_calibrate_strengths`, cuando `p_target ≈ 0.5`, `odds_target ≈ 1`, `k ≈ 1`, y `k_damped = 1^damping = 1` para cualquier valor de damping. La función retorna las strengths sin cambios.
3. **Conclusión**: con el modelo actual (débil en val, fuerte en test), la mayoría de las predicciones están en la zona neutral donde el damping es irrelevante. La mejora existe conceptualmente pero no se manifiesta en el season sim.

**Decisión**:
- **Mantener la implementación** (`adaptive_damping()` + parámetro `damping` opcional): es un patrón correcto y queda disponible para cuando el MatchPredictor mejore.
- **No es el default**: `simulate_season(damping=None)` usa el constante `MATCH_PREDICTOR_DAMPING = 0.5` (backward compatible).
- **Siguiente paso lógico** (no en este batch): mejorar el MatchPredictor (mejor calibración, features más discriminativas, time-series CV) para que sus predicciones salgan de la zona neutral, momento en el cual el damping adaptativo podría mostrar efecto.

**Artefactos**:
- `src/simulation/constants.py::adaptive_damping()` + `ADAPTIVE_DAMPING_START/END` + `SUPERLEGA_TOTAL_JORNADAS`
- `src/simulation/season_simulator.py`: `damping` parameter en `simulate_season()` y `simulate_jornada()`; resolución de damping en los 2 call sites
- `src/models/adaptive_damping_experiment.py`: A/B test runner con 100 MC por estrategia
- `models/adaptive_damping_results.json`: resultados (idénticos entre estrategias, documenta el no-effect)
- `tests/test_season_simulator.py::TestAdaptiveDamping`: 6 tests (linear, clamps, custom range, early<late, _calibrate_strengths response)
- `tests/test_models.py::TestAdaptiveDampingArtifacts`: 2 smoke tests del JSON

5. **Sin features de momentum reciente**: el modelo usa `h_racha` y `forma_h` pero no captura momentum del último set o del último partido con granularidad fina.

6. **~~87 features es mucho para 319 muestras~~ → Feature selection NO ayudó** (Batch 3 mid-effort #2, ver sección 8.1). Probado con A/B test sobre test 2024: top-30 features da **peor** resultado (−0.09 AUC) que los 87 originales. La intuición de overfitting no se materializa con los defaults actuales de los modelos; el `CalibratedClassifierCV(cv=3)` ya provee regularización implícita. Limitación descartada por evidencia.

### 8.1. Feature selection experiment (A/B test)

Para atacar directamente la limitación #6 ("87 features para 319 muestras"), se implementó `src/models/feature_selection_experiment.py` con un A/B test estricto:

- **Mismo pipeline que `train.py`**: 87 features (base + team stats + roster), mismo split temporal 2016-22 / 2023 / 2024.
- **Variant A** (baseline): MatchPredictor con los 87 features. Champion = XGBoost.
- **Variant B**: tomar las top-30 features por importance del Variant A, re-entrenar MatchPredictor. Champion = GradientBoosting.
- **Métrica**: AUC en val 2023 y test 2024 (ambos, no solo val).

**Resultados**:

| Métrica | Variant A (87 feat, XGBoost) | Variant B (top-30, GB) | Delta |
|---|---:|---:|---:|
| Val AUC 2023   | 0.5086 | 0.3694 | **−0.1392** ⚠️ |
| **Test AUC 2024** | **0.7070** | 0.6189 | **−0.0882** ⚠️ |
| Test Accuracy  | 0.5135 | 0.5495 | +0.0360 |
| Test Brier     | 0.2452 | 0.2491 | +0.0039 |

**Veredicto**: **degraded**. La feature selection NO ayuda — empéora val y test. La intuición de overfitting no se materializa.

**Por qué falla**:
1. El `CalibratedClassifierCV(cv=3, method="isotonic")` que ya aplicamos al champion provee regularización implícita que mitiga el overfitting de los 87 features.
2. Las features "menos importantes" según importance siguen conteniendo señal útil que se pierde al descartarlas (variant B pasa de 0.7070 a 0.6189 en test).
3. Eliminar features cambia el champion de XGBoost a GradientBoosting, lo que añade varianza adicional (el "champion noise" del benchmark).

**Decisión**: mantener los 87 features en producción. El experimento queda como artefacto (`models/feature_selection_results.json` + script ejecutable) por si en el futuro se quiere re-evaluar con más datos o con time-series CV.

**Artefactos**:
- `src/models/feature_selection_experiment.py` — script ejecutable (`python -m src.models.feature_selection_experiment [--n-top 30]`)
- `models/feature_selection_results.json` — resultados completos con lista de top-30
- `tests/test_models.py::TestFeatureSelectionArtifacts` — smoke tests del JSON

---

## 9. Conclusión

El `MatchPredictor` es el modelo más ambicioso del proyecto, integrando 87 features de tres capas. Aunque su accuracy en test (0.51) está cerca del azar, su **AUC de 0.71 indica que ordena bien las probabilidades**, lo que es exactamente lo que la calibración con damping necesita: una señal direccional más que una clasificación perfecta.

El damping exponencial con `damping=0.5` es la pieza clave: convierte una predicción ruidosa en una corrección conservadora, que sumada partido a partido a lo largo de la temporada tiene un efecto agregado significativo (de 66% de 3-0 a 44% con ML).

La experimentación con feature selection (sección 8.1) y Optuna (mencionada en `memoria/benchmark.md` §7.1) confirma que **los defaults defendibles son robustos**: la búsqueda de mejoras no encontró ganancias que se transfirieran a test 2024. El modelo se queda como está, con un artefacto experimental que documenta el camino.

Es un buen ejemplo de **"modelo débil + shrinkage fuerte > modelo fuerte sin shrinkage"** en producción.
