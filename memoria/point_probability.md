# Point Probability Model — Probabilidad de Ganar un Punto

> **⚠️ ACTUALIZACIÓN 2026-07-22 — B3 del plan consolidado (regresión continua).**
> El modelo dejó de binarizar el target y de mapear la salida a `[0.45, 0.55]`. Ahora
> entrena una `Ridge(alpha=1.0)` sobre el ratio de puntos **continuo** `point_ratio_h`,
> con features rolling sin leakage, y la salida es `clip(pred, POINT_RATIO_CLIP)` con
> `POINT_RATIO_CLIP = (0.40, 0.60)` solo de salvavidas. La motivación completa, las
> cifras del backtest sin leakage (Brier 0.273 → 0.182, ECE 0.242 → 0.057, % 3-0
> 53% → 37.6% vs 38.7% real) y las dos desviaciones conscientes de la spec están en
> [`mejora_precision_2026-07.md` §7.3](mejora_precision_2026-07.md) y
> [`simulator.md` §10.1-10.2](simulator.md). Este documento refleja el estado post-B3.

## Descripción

El `PointProbabilityModel` (`src/models/point_probability.py`) es el modelo más simple del proyecto: convierte las fuerzas relativas de dos equipos en **4 probabilidades de ganar un punto**, según quién esté sacando. Es el bloque fundamental que alimenta el `MatchSimulator` (Markov chain punto a punto).

*Salida: dict con `p_home_serving`, `p_home_receiving`, `p_away_serving`, `p_away_receiving` · Usado por: `MatchSimulator` en cada rally*

---

## 1. Las 4 Probabilidades

En volleyball, la probabilidad de ganar un punto depende de **quién saca**:

| Probabilidad | Significado | Típica |
|---|---|---:|
| `p_home_serving` | P(local gana el punto \| local saca) | ~0.45 |
| `p_home_receiving` | P(local gana el punto \| visitante saca) | ~0.65 |
| `p_away_serving` | P(visitante gana el punto \| visitante saca) | ~0.35 |
| `p_away_receiving` | P(visitante gana el punto \| local saca) | ~0.55 |

**Observación clave**: `p_home_receiving` > `p_home_serving` porque cuando el visitante saca, el local está en **recepción** y en volleyball profesional el equipo que recibe gana ~62% de los rallies (puede organizar un ataque combinado). La diferencia `p_home_receiving - p_home_serving` ≈ 2 × `sideout_rate - 0.5` ≈ 0.24, que es lo que se observa.

---

## 2. Fórmula de Cálculo

> **⚠️ Actualización (Batch 3).** Antes de Batch 3, un único `DEFAULT_SIDEOUT_RATE = 0.62` global se usaba en todos los partidos (ver `src/simulation/constants.py:15`). Desde Batch 3, el proxy per-equipo (calculado desde `DB/sets_partidos.csv` vía `src/data/team_sideout.py`) es el default. La implementación y validación del per-team sideout se detalla en la [§5.1 (Per-team sideout)](#51-per-team-sideout--implementación-y-validación).

```python
# point_probability.py:216-281
def get_point_probabilities(
    self,
    match_features: Optional[dict] = None,
    home_strength: float = 0.5,
    away_strength: float = 0.5,
    home_sideout: float = DEFAULT_SIDEOUT_RATE,
    away_sideout: float = DEFAULT_SIDEOUT_RATE,
) -> dict:
    # 1. Probabilidad base de punto del local
    if match_features and self.is_fitted:
        # B3: Ridge predice DIRECTAMENTE el ratio de puntos del local.
        # El clip es solo un salvavidas (ver POINT_RATIO_CLIP); ya no hay
        # mapping `0.45 + 0.10 * p_dominante`.
        X = pd.DataFrame([match_features])[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        pred = float(self.model.predict(X_scaled)[0])
        p_home_point = float(
            np.clip(pred, POINT_RATIO_CLIP[0], POINT_RATIO_CLIP[1])
        )
    else:
        # Fallback sin modelo: usar strength directamente
        total = home_strength + away_strength
        p_home_point = home_strength / total if total > 0 else 0.5

    p_away_point = 1.0 - p_home_point

    # 2. Ajuste por sideout PER-TEAM (Batch 3)
    # Cuando LOCAL saca: la probabilidad depende del AWAY sideout
    # Cuando VISITANTE saca: depende del HOME sideout
    p_home_serving = p_home_point * (1 - away_sideout) / (
        p_home_point * (1 - away_sideout) + p_away_point * away_sideout
    )
    p_home_receiving = p_home_point * home_sideout / (
        p_home_point * home_sideout + p_away_point * (1 - home_sideout)
    )

    # 3. Clamp final a POINT_PROB_CLIP (0.25, 0.75) para evitar extremos
    p_home_serving = np.clip(p_home_serving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])
    p_home_receiving = np.clip(p_home_receiving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])

    return {
        "p_home_serving": p_home_serving,
        "p_home_receiving": p_home_receiving,
        "p_away_serving": 1.0 - p_home_receiving,
        "p_away_receiving": 1.0 - p_home_serving,
    }
```

### 2.1. Desglose de la fórmula (post-B3)

**Paso 1**: `p_home_point` se calcula de dos formas:
- **Con modelo entrenado** (`is_fitted=True`): una `Ridge(alpha=1.0)` predice **directamente** el ratio de puntos del local `point_ratio_h = pts_h / (pts_h + pts_a)`, en función de 6 features rolling pre-partido. La salida pasa por `clip(pred, POINT_RATIO_CLIP)` con `POINT_RATIO_CLIP = (0.40, 0.60)` solo de salvavidas: los ratios reales de la SuperLega caen en `[0.38, 0.65]` con media 0.5081, así que el clip apenas muerde en partidos normales y solo evita explosiones en features extremas. **No hay ya un mapping `0.45 + 0.10 * p_dominante`** que sesgara la salida hacia el local: ese sesgo era el origen de la sobreconfianza que corregía B3.
- **Sin modelo** (fallback): `p_home_point = home_strength / (home + away)`. Es la probabilidad "naive" basada solo en win rates.

**Paso 2**: Ajuste por sideout rate **per-team** (`home_sideout`, `away_sideout`). En volleyball, el equipo que recibe el saque tiene ventaja (~60-65% de los rallies). La clave asimétrica es:
- Cuando el **local saca**, su probabilidad se reduce según el **away** sideout (qué tan bueno es el visitante recibiendo): `p_home_serving = p_home × (1 - away_sideout) / [p_home × (1 - away_sideout) + p_away × away_sideout]`.
- Cuando el **visitante saca**, la probabilidad del local se multiplica por el **home** sideout: `p_home_receiving = p_home × home_sideout / [p_home × home_sideout + p_away × (1 - home_sideout)]`.

**Paso 3**: Clamp final a `POINT_PROB_CLIP = (0.25, 0.75)` para evitar eventos deterministas en el simulador (un equipo nunca tiene <25% o >75% de probabilidad de ganar un punto individual, sin importar las strengths).

---

## 3. Modelo Subyacente (Ridge con regresión continua, post-B3)

> **⚠️ Cambio B3 (2026-07-22).** El modelo pasó de `LogisticRegression` con
> target binarizado a `Ridge(alpha=1.0)` con target **continuo** `point_ratio_h`.
> La binarización tiraba la información de magnitud (un partido 0.51 y otro
> 0.58 eran la misma clase) y obligaba al mapping `0.45 + 0.10 * p_dominante`,
> que sesgaba la salida hacia el local. Detalle en
> [`mejora_precision_2026-07.md` §7.3](mejora_precision_2026-07.md).

El modelo entrenado es una `Ridge(alpha=1.0, random_state=42)` (de `sklearn.linear_model`). Sus features son las 6 rolling pre-partido que ya consume `_FEATURE_KEYS`:

| Feature | Mapeo desde runtime | Fuente |
|---|---|---|
| `elo_diff` | igual | `rolling_features.elo_diff` |
| `diff_win_rate_global` | igual | `rolling_features.diff_win_rate` |
| `diff_set_win_rate` | → `diff_set_ratio` | en runtime, `set_win_rate = set_ratio` algebraicamente |
| `diff_dominancia` | → `diff_set_ratio` | en runtime, `dominancia = set_win_rate − 0.5` (la diferencia cancela el −0.5) |
| `diff_set_ratio` | igual | `rolling_features.diff_set_ratio` |
| `diff_forma_efectiva` | → `diff_form_ewma` | recencia EWMA equivalente |

> **Desviación consciente de la spec de B3** (Guardrail 3): el plan mapeaba
> `diff_dominancia → diff_set_diff_exp`. Claude detectó que en runtime
> `diff_dominancia`, `diff_set_win_rate` y `diff_set_ratio` son **algebraicamente
> idénticas** (porque `dominancia = set_win_rate − 0.5` y la diferencia cancela el
> `−0.5`). Se mapean las tres a `diff_set_ratio` para reproducir esa identidad; la
> L2 de Ridge absorbe la colinealidad. Es preferible a entrenar con una señal que
> en producción nunca se sirve. `_FEATURE_KEYS` no cambia, así que los 3
> productores del dict siguen válidos.

**Target continuo**: `y = point_ratio_h = pts_h / (pts_h + pts_a)`. La función
`build_point_training_data(max_season=None)` une las features rolling pre-partido
con el ratio real del partido (outcome, válido como target) por la clave natural
`(temporada_inicio, jornada_num, local, visitante)`. `max_season` permite entrenar
sin ver la temporada de test (necesario para la medida sin leakage de B3).

### 3.1. Métricas del último re-entrenamiento (post-B3)

```
[PointProbability] Base home point prob: 0.5081
[PointProbability] Base away point prob: 0.4919
[PointProbability] Default sideout rate: 0.62
```

Es decir, el modelo base estima que el local gana **50.81%** de los puntos en promedio — el ratio real medio de la SuperLega — frente al 53% sesgado del modelo binarizado anterior. El intercept aprendido del Ridge es 0.5081, no 0.5387. Esta diferencia pequeña por punto se amplifica ~7× a lo largo del Markov chain y era el origen de la sobreconfianza en favoritos que midió B1 (ECE 0.242, 53% de 3-0 simulados vs 39% reales).

### 3.2. Tamaño del artefacto

`models/point_probability.joblib` pesa **~1.8 KB** porque una `Ridge` con 6 features es incluso más compacta que la LogReg anterior. Esto es normal y NO indica bug — el archivo contiene:
- `model`: `Ridge(alpha=1.0, random_state=42)` con 6 coeficientes + intercept
- `scaler`: `StandardScaler` con media y std de 6 features
- `feature_cols`: lista de 6 strings
- `base_home_point_prob`, `base_away_point_prob`, `is_fitted`: floats/booleans

El formato de persistencia es un `dict` con esas 6 keys, **idéntico** al del modelo binarizado anterior, restaurado en `PointProbabilityModel.load()`. Adicionalmente, para el backtest sin leakage, `models/point_probability_lt2024.joblib` contiene el mismo modelo reentrenado solo con historia < 2024 (786 partidos).

### 3.3. Validación (backtest B1 sobre 2024, sin leakage, 222 partidos, n=500)

| Métrica | Antes (binarizado) | **B3 (Ridge continuo)** | Elo (ref) |
|---|---:|---:|---:|
| Brier | 0.2731 | **0.1815** | 0.1941 |
| LogLoss | 0.8241 | **0.5365** | 0.5690 |
| Accuracy | 0.6486 | **0.7207** | 0.6892 |
| ECE | 0.2419 | **0.0565** | 0.0454 |
| 3-0 simulado | 53.0 % | **37.6 %** | real 38.7 % |
| L1 márgenes | 0.2858 | **0.0315** | — |

El simulador pasa de **degradar** la señal Elo a **superarla** en Brier, logloss y accuracy, y la distribución de 3-0/3-1/3-2 calca a la real (L1 = 0.031, antes 0.286). El control de leakage (mismo backtest con el modelo de producción, que sí incluye 2024) da Brier 0.1822 y ECE 0.0824 — la mejora es estructural, no artefacto. Detalle completo en `mejora_precision_2026-07.md` §7.3 y `simulator.md` §10.1-10.2.

---

## 4. Uso en el Simulador

El `MatchSimulator` recibe el `PointProbabilityModel` en su constructor (`src/api/main.py:93`) y lo usa en cada rally:

```python
# src/simulation/simulator.py:392 (aprox, _default_point_probs fallback)
# Cuando el point_model está disponible:
probs = self.point_model.get_point_probabilities(
    home_strength=home_strength,
    away_strength=away_strength,
)
p_home_serving = probs["p_home_serving"]
p_home_receiving = probs["p_home_receiving"]

# En el loop punto a punto:
if server == "home":
    p_home_wins = p_home_serving
else:
    p_home_wins = p_home_receiving
```

El modelo aporta el **componente "fuerza"** del Markov chain. La dinámica intra-rally (momentum, sideout) se suma por separado en `MatchSimulator`.

### 4.1. Fallback si el modelo no carga

Si `point_probability.joblib` falta o está corrupto, la API pone `point_model = None` y el `MatchSimulator` usa `_default_point_probs(home_strength, away_strength)` (`simulator.py:392`), que aplica la misma fórmula con `sideout=0.62` hardcodeado pero sin pasar por el LogisticRegression. Es decir, **el simulador sigue funcionando pero sin ajuste por features de partido**.

---

## 5. Limitaciones

1. **~~Sideout rate constante (0.62)~~ → Per-team sideout proxy** (Batch 3 mid-effort): el modelo ahora acepta `home_sideout` y `away_sideout` como parámetros. El `MatchSimulator` los resuelve desde `src/data/team_sideout.py`, que computa por equipo el ratio de puntos ganados (proxy de sideout) desde `DB/sets_partidos.csv` y cachea el resultado. Equipos sin datos suficientes caen al fallback `DEFAULT_SIDEOUT_RATE = 0.62`. **Limitación residual**: no hay point-level data, así que el "sideout" real (P(ganar recibiendo)) no se puede medir directo; usamos el point-ratio agregado como aproximación. La mejora es real pero no es el sideout canónico de la literatura.

2. **Probabilidad de punto fija en el set**: el modelo devuelve un set de 4 probabilidades al inicio del partido y no las actualiza durante el set. No modela fatiga, cambios tácticos, ni el efecto del marcador parcial (estar 24-20 vs. 0-0 se trata igual).

3. **~~Mapping conservador [0.45, 0.55]~~ → Regresión continua con clip (0.40, 0.60)** (B3, 2026-07-22): la `LogisticRegression` con target binarizado se reemplazó por una `Ridge` con target continuo `point_ratio_h`. La salida es `clip(pred, POINT_RATIO_CLIP)` con `POINT_RATIO_CLIP = (0.40, 0.60)` solo de salvavidas. El sesgo de "p = 0.5387 con features neutras" desapareció: el intercept aprendido es 0.5081 (el ratio real medio).

4. **~~Target binarizado~~ → Target continuo** (B3): `fit()` ahora entrena sobre `y = point_ratio_h` continuo, no sobre `y_binary = (point_ratio_h > 0.5)`. Esto preserva la información de magnitud (un 0.51 y un 0.58 ya no son la misma clase).

5. **Features limitadas (6)**: el modelo solo usa diferencias de stats agregadas, no features in-match (momentum, score parcial, cansancio). Para el partido suelto, esto es suficiente. Para la temporada, podría complementarse con features de `RuntimeFeatureBuilder`.

6. **Sobreconfianza residual**: el ECE (0.057) sigue por encima del Elo puro (0.045). Queda algo de sobreconfianza residual en el modelo, especialmente en las colas (`elo_diff = ±400` da P = 0.95 / 0.085), que conviene vigilar si en el futuro se amplía el dataset (B6) con más partidos desiguales.

7. **Sin reentrenamiento periódico**: el modelo se entrena una vez con todos los datos históricos. Si la liga cambia (nuevo equipo, regla nueva), el modelo se desactualiza.

### 5.1. Per-team sideout — implementación y validación

**Motivación** (de la sección 7 original, ahora atacada): la literatura reporta que equipos como Perugia/Trento tienen sideout ~65% y los débiles ~58%. Con el 0.62 global se borraba la señal de habilidad por recepción.

**Implementación** (Batch 3 mid-effort):
- `src/data/team_sideout.py`: lee `sets_partidos.csv`, agrega puntos ganados / jugados por equipo (como local Y visitante), normaliza nombres con `team_mapper.normalize_team_name`, filtra equipos con <50 sets históricos, cachea el resultado en memoria.
- `PointProbabilityModel.get_point_probabilities()` ahora acepta `home_sideout` y `away_sideout` (defaults al global). Las fórmulas de `p_home_serving` y `p_home_receiving` usan el sideout del equipo que RECIBE en cada rally (no del que saca), respetando la simetría de Markov: `p_home_serving + p_away_receiving = 1` y `p_home_receiving + p_away_serving = 1`.
- `MatchSimulator.simulate_match` llama a `get_sideout_rates(home_team, away_team)` y pasa los valores al `point_model` o al fallback `_default_point_probs`.
- Equipos sin datos suficientes (ej: nuevos en la liga) caen al `DEFAULT_SIDEOUT_RATE = 0.62`.

**Datos observados** (34 equipos con datos suficientes):
- Top: Perugia 0.530, Lube 0.520, Trento 0.519
- Bottom: Grottazzolina 0.471, Cuneo 0.474, Cantù 0.475
- Rango ~0.06 (menor al reportado por la literatura 0.07, pero consistente con la dirección: tops sideoutean más, débiles menos).

**Validación cualitativa** (500 simulaciones MC por matchup con `home_strength=0.55, away_strength=0.45`):

| Matchup | Sideout (h/a) | 3-0% | 3-1% | 3-2% |
|---|---|---:|---:|---:|
| Perugia vs Grottazzolina (mismatch grande) | 0.530 / 0.471 | 70.6 | 23.2 | 6.2 |
| Trento vs Padova (mismatch moderado)    | 0.519 / 0.476 | 66.8 | 23.8 | 9.4 |
| Perugia vs Trento (top vs top)           | 0.530 / 0.519 | 55.4 | 30.6 | 14.0 |
| Modena vs Lube (mid vs mid)              | 0.504 / 0.520 | 46.6 | 32.6 | 20.8 |

El patrón es el esperado: mismatch grande → muchos 3-0, matchup equilibrado → distribución más uniforme con 3-2. Antes (con 0.62 global y sin distinguir equipos) la varianza entre matchups era menor.

**Limitación residual**: el proxy point-ratio no separa "sideout" (ganar recibiendo) de "winning when serving". El verdadero sideout de la literatura solo se puede medir con point-level data, que no tenemos. La mejora es una aproximación, no la solución completa.

---

## 6. Conclusión

El `PointProbabilityModel` es el componente más "artesanal" del proyecto: una fórmula con 3 pasos (probabilidad base → ajuste por sideout → clamp) y una `Ridge` con regresión continua como pieza opcional (post-B3). Su valor está en **convertir dos scalars (home_strength, away_strength) en 4 probabilidades físicamente plausibles** que el Markov chain puede consumir.

Es el modelo con menos AUC explícito (no es un clasificador puro, es un regresor de ratio) pero **el más estable**: sus predicciones son siempre sensatas gracias al clamp y al sideout per-team. Tras B3, la `Ridge` aporta una corrección continua y calibrada que la LogReg binarizada + mapping aplastante no podía dar (el sesgo de 0.5387 con features neutras desapareció, ahora el intercept es 0.5081).

**Resultado clave (post-B3, backtest B1 sobre 2024 sin leakage)**: el simulador pasa de **degradar** la señal Elo a **superarla** en Brier (0.182 vs 0.194), logloss (0.537 vs 0.569) y accuracy (0.721 vs 0.689), y la distribución de 3-0/3-1/3-2 calca a la real (L1 = 0.031 vs 0.286). El ECE mejora 4,3× (0.242 → 0.057), quedando ya muy cerca del Elo puro (0.045). Detalle completo en `mejora_precision_2026-07.md` §7.3 y `simulator.md` §10.1-10.2.

Para el TFG, este modelo es interesante porque muestra que **no siempre se necesita un modelo complejo**: a veces un modelo paramétrico simple con buenos priors (sideout per-team + clip de salvavidas) supera a un modelo "data-driven" sin esas restricciones. La lección metodológica de B3 (la binarización tira información, los mappings sesgan la salida, y la cadena amplifica los sesgos) es aplicable a otros problemas similares.

---

## 7. Carencias Conocidas y Roadmap de Mejora

El `PointProbabilityModel` actual tiene estas limitaciones que lo dejan sub-utilizado:

1. **Solo 6 features**: el modelo usa `elo_diff` y diferencias de win rate, pero ignora features in-match (momentum, score parcial, cansancio), features de roster (top scorer) y features de superficie/condiciones.

2. **~~Mapping conservador [0.45, 0.55]~~ → Regresión continua con clip (0.40, 0.60)** (B3, 2026-07-22, cerrado): la salida ya no es un rango estrecho artificial; es el ratio continuo predicho por Ridge, recortado a un salvavidas razonable.

3. **~~Sideout rate constante (0.62)~~ → Per-team sideout proxy** (Batch 3 mid-effort, ver sección 5.1). Implementado como proxy point-ratio, no como feature aprendido de verdad. Limitación residual: no separa "sideout" de "win when serving" sin point-level data.

4. **~~Target binarizado~~ → Target continuo** (B3, 2026-07-22, cerrado): el modelo entrena sobre `y = point_ratio_h` continuo, no sobre `y_binary = (point_ratio_h > 0.5)`.

5. **No captura el efecto de "estar 24-20"**: la probabilidad de ganar el ultimo punto de un set es MUY distinta segun el marcador. El modelo no tiene features in-set.

6. **Sobreconfianza residual en las colas** (B3 follow-up): el modelo con `elo_diff = ±400` da P = 0.95 / 0.085, que podría ser extremo si B6 (más datos) añade partidos muy desiguales al histórico. Vigilar en re-entrenamientos futuros.

7. **Sin reentrenamiento periódico**: el modelo se entrena una vez con todos los datos históricos. Si la liga cambia (nuevo equipo, regla nueva), el modelo se desactualiza.

### Roadmap de mejora

**Corto plazo (1-2h)**:
- [x] Pasar `match_features` a `simulate_match` en los 3 call sites para que el modelo se use de verdad. (commit a1b3701)
- [x] **Per-team sideout proxy** (Batch 3 mid-effort): implementado vía `src/data/team_sideout.py`, validado cualitativamente (3-0 distribution).
- [x] **B3 — Regresión continua con clip (0.40, 0.60)** (2026-07-22): Ridge(alpha=1.0) sobre target continuo, sin mapping sesgado. Brier 0.273 → 0.182, ECE 0.242 → 0.057.
- [ ] Ampliar features a las 15 que el `SetPredictor` consume.

**Mediano plazo (4-6h)**:
- [ ] Sideout rate **aprendido** (no proxy): feature `sideout_rate_h` calculado de los ultimos N sets como `P(ganar_punto | recibiendo)`, idealmente desde point-level data.
- [ ] Validacion cruzada temporal (no un solo split) con el protocolo honesto de B1.
- [ ] Modelo jerárquico: P(ganar_punto) condicionado a estado del set y del partido.

**Largo plazo (TFG siguiente)**:
- [ ] Features in-set: score parcial, momentum, jugador al saque.
- [ ] Modelo jerarquico: P(ganar_set) y P(ganar_punto | estado_set) acoplados.
- [ ] Comparar con un modelo transformer / LSTM que consuma secuencias de puntos.
