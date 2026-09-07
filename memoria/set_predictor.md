# Set Predictor — Predicción del Ganador de un Set

> **⚠️ ACTUALIZACIÓN 2026-07-08 (modelo) + 2026-07-08 (producción).** Medido
> con el protocolo honesto (rolling-origin, test held-out 2025/26), el
> ExtraTrees calibrado da AUC **0.65** (no infló como el match: sus features
> de set no tenían leakage de temporada completa). La auditoría de precisión
> encontró que un **LogisticRegression regularizado con pesos de recencia**
> (half-life 2 temporadas, entrenado en 2022-2024) lo mejora a **AUC 0.71 /
> acc 0.66** en el test de 2025, confirmando que en este régimen de datos
> pequeños los modelos lineales baten a los árboles profundos.
>
> ⚠️ **Importante (validación per-year, ver
> [`mejora_precision_2026-07.md` §7.2](mejora_precision_2026-07.md)):** el
> "AUC 0.71" es el test sobre 2025/26 (853 sets, la temporada más grande del
> dataset). El **CV honesto rolling-origin de 2 folds** da **0.63 ± 0.08** y
> la **media per-year 2018-2025** da **0.61** (Spearman con val_year = -0.17,
> p=0.69, sin tendencia monotónica). El legacy ExtraTrees tenía CV 0.62 ± 0.03
> sobre 4 folds, así que la mejora del v2 en el rolling-origin multi-temporada
> es +0.01 (dentro del ruido). El +0.06 del test aislado es **2025-específico**
> y queda pendiente re-validar cuando llegue 2026/27.
>
> **Estado en producción (API):** desde este commit el `set_predictor_v2.joblib`
> (LogReg+recencia) es el campeón de producción que carga `src/api/main.py`,
> y el `set_predictor.joblib` (ExtraTrees) queda como fallback. El artefacto
> se regenera con `python -m src.models.train_improved`. Detalle del proceso
> en [`mejora_precision_2026-07.md`](mejora_precision_2026-07.md).

## Descripción

El `SetPredictor` (`src/models/set_predictor.py`) es un clasificador binario que predice la probabilidad de que el equipo local gane un set individual de volleyball. Es uno de los tres modelos entrenados en el pipeline; junto con el `MatchPredictor` forma el núcleo de la calibración ML del simulador.

*Salida: P(local gana set) ∈ [0, 1] · Usado por: `MatchSimulator` (clamp adaptativo) y `RuntimeFeatureBuilder` (features in-match)*

---

## 1. Pipeline de Entrenamiento

```
match_features.csv  →  feature_store.get_set_splits()  →  train / val / test
   (3032 sets)              (split temporal)             (1345 / 352 / 482)
                                                            ↓
                                                  SetPredictor.train()
                                                            ↓
                                              6 modelos candidatos
                                              + selección por AUC
                                              + calibración isotonic
                                                            ↓
                                              set_predictor.joblib
```

El predictor sigue la estrategia estándar de la librería:

1. Carga features desde `feature_store.get_set_splits()` (split temporal: train 2016-2022, val 2023, test 2024).
2. Entrena **6 modelos candidatos** sobre el set de train.
3. Evalúa cada candidato en el set de validación y selecciona el de **mayor AUC-ROC**.
4. Calibra el modelo campeón con `CalibratedClassifierCV(cv=3, method="isotonic")` para que las probabilidades reflejen frecuencias reales.
5. Evalúa en el set de test (datos no vistos durante entrenamiento/calibración).
6. Serializa el predictor completo (scaler + best_model + calibrated_model + feature_names) en `models/set_predictor.joblib`.

---

## 2. Candidatos Comparados

| Modelo | Hiperparámetros principales |
|---|---|
| LogisticRegression | `max_iter=2000, C=1.0, solver="lbfgs"` |
| RandomForest | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| **ExtraTrees** (champion) | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| GradientBoosting | `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8` |
| XGBoost | `n_estimators=300, max_depth=5, learning_rate=0.05, reg_alpha=0.1` |
| LightGBM | `n_estimators=300, max_depth=5, learning_rate=0.05, reg_alpha=0.1` |

Todos los modelos usan `random_state=42` para reproducibilidad. La función `get_candidate_models()` (`set_predictor.py:37`) devuelve el diccionario con los seis.

---

## 3. Resultados de Validación (selección de champion)

Métricas en el set de validación (año 2023, 352 sets, balance de clases ≈ 52/48):

| Modelo | Acc | AUC | Brier | Prec | Rec |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.571 | 0.580 | 0.2511 | 0.573 | 0.571 |
| RandomForest | 0.599 | 0.621 | 0.2401 | 0.602 | 0.599 |
| **ExtraTrees** | 0.588 | **0.6275** | 0.2380 | 0.589 | 0.588 |
| GradientBoosting | 0.562 | 0.597 | 0.2531 | 0.562 | 0.562 |
| XGBoost | 0.540 | 0.556 | 0.2703 | 0.538 | 0.540 |
| LightGBM | 0.548 | 0.563 | 0.2696 | 0.547 | 0.548 |

**Modelo campeón: ExtraTrees** (AUC = 0.6275) en la versión legacy. Es elegido
por AUC porque es invariante al threshold y captura el ordenamiento de
probabilidades, que es lo que el simulador necesita para el clamp adaptativo.
La versión de **producción** (v2) usa LogisticRegression con recencia y
entrena en 2022-2024 — ver el banner al inicio del documento y
[`mejora_precision_2026-07.md` §6-§7.2](mejora_precision_2026-07.md).

---

## 4. Resultados en Test (datos 2024, no vistos)

Evaluación del modelo calibrado sobre 482 sets de la temporada 2024 (balance 56/44 a favor del local):

| Métrica | Valor |
|---|---:|
| **Accuracy** | 0.6224 |
| **AUC-ROC** | 0.6542 |
| **Brier Score** | 0.2289 |

Reporte de clasificación:

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Visitante | 0.62 | 0.39 | 0.48 | 214 |
| Local | 0.62 | 0.81 | 0.70 | 268 |
| **macro avg** | 0.62 | 0.60 | 0.59 | 482 |
| **weighted avg** | 0.62 | 0.62 | 0.60 | 482 |

El modelo tiene **alto recall para el local (0.81)** pero bajo recall para el visitante (0.39), lo que sugiere que tiende a sobre-predecir victorias del local. Esto es esperable porque (a) hay leve home advantage en los datos (56% gana_local) y (b) las features reflejan ese sesgo.

![Curva de calibración del SetPredictor legacy (ExtraTrees): sin calibrar (izquierda) y calibrado (derecha)](../models/plots/reliability_set.png)
![Curva de calibración del SetPredictor legacy calibrado](../models/plots/reliability_set_calibrado.png)

*Figuras: Curvas de calibración del modelo ExtraTrees sin calibrar (arriba) y calibrado con isotonic (abajo). Generadas por `src/models/reliability_curve.py`. La curva calibrada sigue más cerca la diagonal, indicando mejor calibración de probabilidades.*

---

## 5. Top 10 Features Más Importantes

Importancia según `ExtraTrees.feature_importances_`:

| Rank | Feature | Importancia | Interpretación |
|---:|---|---:|---|
| 1 | `diff_sets_antes` | 0.0892 | Diferencia de sets ganados antes del set actual |
| 2 | `momentum_h` | 0.0890 | Momentum reciente del local en el partido |
| 3 | `sets_a_antes` | 0.0809 | Sets ganados por el visitante antes del set actual |
| 4 | `sets_h_antes` | 0.0721 | Sets ganados por el local antes del set actual |
| 5 | `h2h_diff` | 0.0636 | Diferencia de H2H win rate |
| 6 | `set_num_norm` | 0.0636 | Número de set normalizado (1-5) |
| 7 | `diff_set_wr` | 0.0421 | Diferencia de set win rate histórico |
| 8 | `forma_a` | 0.0411 | Forma reciente del visitante |
| 9 | `forma_h` | 0.0398 | Forma reciente del local |
| 10 | `set_wr_h` | 0.0394 | Set win rate histórico del local |

**Hallazgo clave**: las 4 features más importantes son **in-match features** (momentum y score parcial de sets). Esto confirma que la dinámica del partido en curso pesa más que los features pre-partido (Elo, forma, H2H). El modelo está capturando bien la lógica "si voy 2-0 arriba, gano el 3er set con alta probabilidad".

---

## 6. Uso en el Simulador

El `SetPredictor` se usa en el simulador de temporadas (no en el simulador de partido suelto) como **señal candidata para relajar el clamp de probabilidad punto a punto** del Markov chain. El flujo en el código actual es:

```python
# src/simulation/simulator.py:251-283
# Clamp adaptativo via SetPredictor: se evalúa una vez al inicio del set
clamp_low, clamp_high = DEFAULT_CLAMP_RANGE  # [0.20, 0.80] por defecto

if set_predictor is not None and set_context_base is not None:
    # Centro del clamp: señal del Elo viva (no del SetPredictor)
    base_p_neutral = (
        point_probs["p_home_serving"] + point_probs["p_home_receiving"]
    ) / 2
    p_center = base_p_neutral

    # Cortocircuito: con w >= 1.0 ni se llama al SetPredictor
    # (sería coste puro: la salida se multiplicaría por 0)
    if SET_BLEND_WEIGHT_ELO < 1.0:
        p_set_home = self._eval_set_predictor(
            set_predictor, set_context_base, 0, 0, target_score,
            sets_home_antes, sets_away_antes,
        )
        if p_set_home is not None:
            # A2: p_set vive en escala de SET; el clamp gobierna PUNTOS.
            # Convertir antes de centrar (un favorito con P(set)=0.75
            # solo necesita P(punto)~0.55).
            p_set_punto = p_point_from_p_set(p_set_home, target_score)
            p_center = (
                SET_BLEND_WEIGHT_ELO * base_p_neutral
                + (1 - SET_BLEND_WEIGHT_ELO) * p_set_punto
            )

    clamp_low = max(POINT_PROB_CLIP_ADAPTIVE_HARD[0], p_center - CLAMP_MARGIN_POINT)
    clamp_high = min(POINT_PROB_CLIP_ADAPTIVE_HARD[1], p_center + CLAMP_MARGIN_POINT)
```

El blend es `p_center = w·base_p_neutral + (1−w)·p_set_punto`. Las constantes vigentes tras el Grupo A del plan consolidado:

| Constante | Valor | Significado |
|---|---:|---|
| `SET_BLEND_WEIGHT_ELO` | `1.0` | Peso del Elo en el centro. **1.0 ≡ ignorar al SetPredictor** (A4) |
| `CLAMP_MARGIN_POINT` | `0.10` | Margen del clamp en escala de PUNTO (A2 corrige error de escala histórico) |
| `POINT_PROB_CLIP_ADAPTIVE_HARD` | `(0.10, 0.90)` | Suelo y techo duros del clamp adaptativo |
| `CLAMP_MARGIN` | `0.20` | **LEGACY** (escala de SET). Ya no se usa en runtime; pineado en `test_team_mapper.py` hasta limpieza |

### 6.1. Estado real (post-A4)

Con `SET_BLEND_WEIGHT_ELO = 1.0` (adoptado tras A4), el método `_eval_set_predictor` **nunca se invoca** en runtime: la guarda `if SET_BLEND_WEIGHT_ELO < 1.0` corta antes de la llamada. El centro del clamp es siempre `base_p_neutral` (la media de `p_home_serving` y `p_home_receiving`, derivada de las fuerzas calibradas por Elo), y el margen es `CLAMP_MARGIN_POINT = 0.10`.

**Qué sobrevive del SetPredictor en el sistema** (cuesta poco, queda por compatibilidad y trazabilidad):

- El artefacto `set_predictor_v2.joblib` se regenera con `python -m src.models.train_improved` (2 KB).
- El API lo carga al arrancar (`src/api/main.py:65`) y lo expone en `GET /api/modelo/info`.
- El contexto `set_context_base` se construye en cada set vía `_build_set_context_base` (línea 162-169 de `simulator.py`) **si el caller pasa `set_predictor` y `team_features`**. Es un dict, coste despreciable.
- Los tests pinean el contrato A3 (determinismo, pureza, no-live-score, discriminación) en `tests/test_set_contract.py`.
- `use_set_calibration` (flag del API) hoy controla solo si se construye el contexto; con `w=1.0` la salida se descartaría aunque se evaluase.

**Por qué se mantiene el cableado**: el plan consolidado preveía como Guardrail 9 que el clamp pudiese no aportar; la opción de borrar el cableado no se ejecutó porque (a) deja el sistema "vivo" para una futura re-evaluación si llegan datos más informativos, (b) el coste es despreciable, (c) argumentar el resultado negativo en la defensa requiere mostrar el cableado y el tuneo. Detalle completo del desenlace en `mejora_precision_2026-07.md` §7 y en `simulator.md` §4.3.

---

## 7. Persistencia y Carga

### Guardado (`save`)

```python
# set_predictor.py:252
save_data = {
    "scaler": self.scaler,
    "best_model_name": self.best_model_name,
    "best_model": self.best_model,
    "calibrated_model": self.calibrated_model,
    "feature_names": self.feature_names,
    "results": self.results,
}
joblib.dump(save_data, path)
```

El predictor completo pesa **~19 MB** (el `ExtraTrees` con 300 estimadores es el componente dominante).

### Carga (`try_load_v2`)

```python
# src/api/main.py:65-68
set_predictor, sp_source = LogRegSetPredictor.try_load_v2(
    MODELS_DIR / "set_predictor_v2.joblib",
    MODELS_DIR / "set_predictor.joblib",
)
# sp_source: "v2_logreg_recency" o "legacy_extratrees"
```

El adaptador v2 (`LogRegSetPredictor`, `src/models/set_predictor_v2.py`) implementa una carga en cascada: primero intenta cargar `set_predictor_v2.joblib` (LogReg+recencia en producción); si no existe o falla, cae al `set_predictor.joblib` (ExtraTrees legacy, entrenado por `SetPredictor.train()`). Si ambos fallan, el `set_predictor` queda en `None` y el simulador usa el clamp por defecto `[0.20, 0.80]`.

El retorno `sp_source` es un string que indica qué camino se cargó (`"v2_logreg_recency"` o `"legacy_extratrees"`), para trazabilidad en los logs del API y en el endpoint `/api/modelo/info`.

---

## 8. Limitaciones y Trabajo Futuro

1. **Recall bajo para visitante (0.39)**: el modelo sub-predice victorias visitantes. Podría corregirse con class_weight, oversampling o features que capten mejor el "away upset".
2. **No aporta señal útil al clamp (resultado negativo A4, Guardrail 9)**: con el reescalado a espacio de punto de A2 y el blend en `w ∈ {0.5, 0.7, 0.9, 1.0}`, los tuneos de A4 fueron indistinguibles del baseline `OFF` a partir de `w = 0.9`. El desenlace del plan consolidado fue adoptar `w = 1.0` (no consultar al SetPredictor en el clamp) y documentar el resultado negativo. La causa no fue la hipótesis previa de "features frías en las primeras jornadas": A4 midió con estado limpio y el resultado fue el mismo con o sin features calientes. La conclusión es que, **con los datos actuales**, el SetPredictor no añade señal al clamp del Markov chain que la del Elo con margen ya no provea. La sub-dispersión detectada en A6 (Spearman = −1.0 con cuatro equipos a `std = 0.00`) **fue corregida por B3** (PointProbabilityModel con regresión continua, 2026-07-22): su origen era el modelo de punto (binarización + mapping sesgado), no el clamp. La puerta queda abierta para un SetPredictor de otro tipo (p. ej. con features de jugador o de momentum entre sets) si en el futuro se demuestra que aporta señal. Detalle en `mejora_precision_2026-07.md` §7/§7.3 y en `simulator.md` §4.3/§10.1-10.2.
3. **Sin features de momentum entre sets**: el modelo usa `momentum_h` intra-set pero no captura momentum entre sets (racha de sets ganados/perdidos). Útil si en el futuro se reabre el debate sobre un SetPredictor más rico.
4. **Calibración isotonic puede sobre-ajustar** (solo afecta al legacy ExtraTrees, no al v2 en producción): con `cv=3` y 1345 sets de train, hay riesgo de sobrecalibración. Una alternativa es Platt scaling o `cv=5`. El v2 LogReg+recencia no usa calibración post-hoc porque el LogReg ya produce probabilidades calibradas por construcción.

---

## 9. Conclusión

El `SetPredictor` implementa un patrón estándar de **comparar 6 modelos, seleccionar el mejor por AUC, y calibrar con isotonic** (legacy ExtraTrees) o un esquema más simple con **LogisticRegression regularizado y pesos de recencia** (v2 en producción, 2026-07). El champion actual (v2 LogReg+recencia) alcanza un AUC test 2025 de **0.697** con CV rolling-origin de **0.679 ± 0.017** (n=1193), por encima del azar pero lejos de un predictor fuerte — esperable porque la varianza intra-partido es alta y el volleyball tiene mucho ruido.

La integración con el simulador (clamp adaptativo) le da un rol **candidato, no activo**: el plan consolidado demostró experimentalmente (Grupo A, items A2-A4) que con los datos actuales la señal del SetPredictor no mejora el clamp del Markov chain más allá de lo que ya aporta el Elo con margen. El desenlace honesto (Guardrail 9) fue `SET_BLEND_WEIGHT_ELO = 1.0`, equivalente a no consultar al SetPredictor. El modelo se queda cableado y validado por tests (contrato A3) por dos razones: (a) deja el sistema listo para una futura re-evaluación si llegan features más informativas, y (b) el resultado negativo es defendible en la memoria del TFG mostrando el tuneo completo.

---

## 10. Adaptador LogRegSetPredictor (v2)

El adaptador `LogRegSetPredictor` (`src/models/set_predictor_v2.py`, ~121 líneas) envuelve el modelo de producción **LogisticRegression con recencia** que sustituyó al ExtraTrees calibrado como campeón de la API en julio de 2026.

### 10.1. Motivación

El modelo legacy (ExtraTrees + calibración isotonic) tenía tres problemas en producción:

1. **Tamaño**: ~19 MB frente a ~5 KB del LogReg, por los 300 estimadores del ExtraTrees.
2. **AUC estancado**: AUC 0.65 en test, con validación cruzada 4 folds de 0.62 ± 0.03, sin tendencia de mejora al añadir más datos.
3. **Calibración isotonic frágil**: con solo 1345 sets de entrenamiento, `CalibratedClassifierCV(cv=3, method="isotonic")` corría riesgo de sobre-ajuste.

El LogisticRegression con recencia (half-life 2 temporadas, C=0.5) resuelve los tres: es órdenes de magnitud más pequeño, alcanza AUC 0.71 en test 2025 (CV 0.63 ± 0.08), y al no necesitar calibración post-hoc elimina una fuente de varianza.

### 10.2. Contrato Duck-Typed

El adaptador expone la misma interfaz que el `SetPredictor` legacy para que el API y el simulador puedan usar cualquiera de los dos sin cambios:

```python
# Contrato mínimo:
#   .feature_names: list[str]     — 21 features de set
#   .predict_proba(df) → ndarray  — forma [n, 2], columna 1 = P(local gana set)
```

El `try_load_v2` devuelve `(LogRegSetPredictor, "v2_logreg_recency")` si carga el artefacto v2, o `(SetPredictor, "legacy_extratrees")` si cae al fallback. Esto permite que `src/api/main.py` maneje la selección en una sola línea.

### 10.3. Entrenamiento

El modelo v2 se entrena con `python -m src.models.train_improved`, que:

- Usa datos de 2022-2024 (recencia: half-life=2 temporadas, ponderando más los sets recientes).
- Entrena un `LogisticRegression(C=0.5, max_iter=2000, random_state=42)` sobre 21 features de set.
- NO aplica scaler ni calibración post-hoc (el LogReg ya produce probabilidades calibradas por construcción).
- Serializa el artefacto como `models/set_predictor_v2.joblib`.

### 10.4. Validación

| Métrica | Valor |
|---|---:|
| AUC test 2025 (1193 sets) | **0.697** |
| CV rolling-origin 2 folds | 0.679 ± 0.017 |
| Accuracy test 2025 | 0.650 |
| Brier Score test 2025 | 0.216 |

Cifras tras corrección B0b (2026-07-15): `set_features.csv` regenerado sin
colisión (n 853→1193). Datos pre-B0b en
[`registro_historico_b0.md`](registro_historico_b0.md) §B.2.

El desglose completo de la validación (incluyendo el análisis per-year y la discusión de por qué el "AUC 0.71" es 2025-específico) está en [`mejora_precision_2026-07.md` §6-§7.2](mejora_precision_2026-07.md) y [`prediccion_temporadas.md` §7](prediccion_temporadas.md).

### 10.5. Estado en Runtime (post-A4)

El adaptador `LogRegSetPredictor` se carga al arrancar el API y se mantiene cableado al `MatchSimulator` (se pasa como argumento `set_predictor=sp`), pero **en la configuración actual su predicción no entra al clamp**. El motivo es el resultado negativo del Grupo A del plan consolidado:

- **A2** corrigió un error de escala histórico: el clamp se construía alrededor de `p_set` (escala de SET) pero la cadena de Markov va en `p_punto`. Ahora el clamp se centra con `p_point_from_p_set(p_set, target_score)` y un margen de `0.10` en espacio de punto (constante `CLAMP_MARGIN_POINT`).
- **A4** probó un blend `p_center = w·base_p_neutral + (1−w)·p_set_punto` con `w ∈ {0.5, 0.7, 0.9, 1.0}`. Resultado: `w=0.9` y `w=1.0` son **idénticos** entre sí e idénticos al baseline `OFF` (sin SetPredictor). El plan preveía esto como Guardrail 9 y el desenlace fue `w=1.0`: la llamada a `_eval_set_predictor` queda **cortocircuitada** por la guarda `if SET_BLEND_WEIGHT_ELO < 1.0` en `src/simulation/simulator.py:266`. Lo que sí aporta valor es el reescalado de A2.

**Tabla resumen de qué hace y qué no hace el SetPredictor hoy**:

| Capa | Estado | Coste |
|---|---|---|
| Entrenar (`train_improved.py`) | Se regenera el `.joblib` | bajo (manual) |
| Cargar (`main.py:65`) | Sí, con fallback legacy | despreciable |
| Exponer en `GET /api/modelo/info` | Sí (nombre + features) | despreciable |
| Construir `set_context` (`_build_set_context_base`, `simulator.py:165`) | Sí, si el caller pasa `set_predictor` y `team_features` | bajo (un dict por set) |
| Evaluar el modelo (`_eval_set_predictor`, `simulator.py:390`) | **No** (cortocircuito `w=1.0`) | 0 |
| Aplicar salida al clamp | **No** (descartado) | 0 |
| Tests pinned (contrato A3) | 188 tests verdes | 0 en runtime |

**Por qué no se borra el cableado**: (a) el plan lo previó como resultado defendible, no como limpieza; (b) el coste de mantenerlo es despreciable; (c) deja el sistema listo para reactivar la mezcla si en el futuro se demuestra que un SetPredictor con features de jugador o momentum entre-sets sí aporta. La sub-dispersión que B3 corrigió vino del modelo de punto, no del clamp, así que este cableado sigue siendo defendible como experimento futuro. Ver `mejora_precision_2026-07.md` §7 (cierre del Grupo A) y §7.3 (B3), y `simulator.md` §4.3 (mecanismo final A2+A4) para el detalle completo.
