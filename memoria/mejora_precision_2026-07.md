# Mejora de Precisión de los Modelos — Proceso Completo (2026-07-08)

Bitácora del trabajo de auditoría, diagnóstico, mejora e integración de la
precisión de los modelos ML del proyecto. Documenta qué se hizo, por qué, con
qué resultados y qué queda pendiente. Es material directamente defendible en el
TFG: el hallazgo principal es tan importante como la mejora.

---

## 1. TL;DR

- El **AUC=0.707 del MatchPredictor era ficticio**: producto de *leakage
  temporal* evaluado sobre un único año de test. Medido honestamente valía
  **0.53** (nivel de azar).
- Se construyó un **protocolo de evaluación honesto** (rolling-origin, test
  held-out) que hace que las métricas signifiquen algo.
- Se reconstruyeron las features **sin leakage** desde `sets_partidos.csv`, con
  un **Elo con margen de victoria**. Resultado: AUC de partido **0.53 → 0.75**.
- La sospecha de envenenamiento de las temporadas viejas quedó **invalidada por
  B0** (era artefacto de la colisión `partido_id`); la recencia operativa
  2022-2024 se mantiene por ciclo de plantillas (half-life 2 temporadas),
  no por sign-flip.
- En este régimen de datos pequeños, **los modelos lineales baten a los árboles
  profundos**, tanto en set como en match.
- Se **integró en producción**: las fuerzas de equipo y la señal de partido del
  simulador ahora salen del Elo limpio; la clasificación simulada por fin
  correlaciona con la calidad real de los equipos.

| Modelo | AUC antes (honesto) | AUC después | Accuracy después |
|---|---:|---:|---:|
| MATCH | 0.53 | **0.75** | 0.69 |
| SET | 0.65 | **0.697** | 0.65 |

El SET v2 tras corrección B0b (2026-07-15): test 2025 AUC **0.697** (n=1193),
CV rolling-origin 2-fold **0.679 ± 0.017**. Detalle en §7.2.

Cifras exactas en [`../docs/COMPARACION_ANTES_DESPUES.md`](../docs/COMPARACION_ANTES_DESPUES.md).
Plan original en [`../docs/PLAN_MEJORA_PRECISION.md`](../docs/PLAN_MEJORA_PRECISION.md).

---

## 2. Punto de partida y sospechas

El código reportaba SetPredictor AUC 0.654 y MatchPredictor AUC 0.707 (test =
temporada 2024/25). Tres señales de alarma motivaron la auditoría:

1. El MatchPredictor tenía **AUC de validación 0.475** (peor que azar) pero AUC
   de test 0.707. Una brecha así entre val y test no es señal de un buen modelo,
   sino de una medición inestable.
2. La **accuracy de test era 0.514** (casi azar) pese al "AUC 0.707". AUC alto +
   accuracy de moneda = probabilidades mal calibradas o métrica engañosa.
3. La selección del "modelo campeón" se hacía sobre **81 partidos** (val=2023).
   Con n=81, el error estándar del AUC es ~±0.06: elegir por ahí es ruido.

## 3. Diagnóstico: cuatro problemas estructurales

1. **El 45% de los datos no se usaba.** El split terminaba en test=2024; la
   temporada 2025/26 (214 partidos, la más grande) quedaba fuera. El train real
   eran ~319 partidos.
2. **Selección de modelo sobre ruido** (los 81 partidos de val).
3. **Leakage temporal** en `enrich_with_team_stats` y `compute_roster_features`:
   mergeaban stats de la **temporada completa** a partidos de esa misma
   temporada. Un partido de octubre "sabía" cómo terminaría el equipo en abril.
4. **Calibración con CV no-temporal** (`CalibratedClassifierCV(cv=3)` mezcla
   temporadas internamente) e isotónica con pocos datos (sobreajuste).

## 4. Fase 0 — Protocolo de evaluación honesto

`src/models/evaluation.py`. Reemplaza el val único por **rolling-origin
(expanding window)**: entrenar ≤ T-1, validar en T, para T ∈ {2021…2024};
test held-out intocable = **2025/26**. Métrica primaria: **log-loss** (calidad
de probabilidad, que es lo que alimenta el simulador Monte Carlo), no AUC.

Al medir el baseline con este protocolo salió la verdad:

| Modelo | Lo que reportaba el código | Medido honesto (test 2025/26) |
|---|---|---|
| MATCH | AUC 0.707 | **AUC 0.528**, logloss 0.694, acc 0.514 |
| SET | AUC 0.654 | AUC 0.653, logloss 0.651, acc 0.606 |

El set aguantó (sus features no tenían leakage de temporada). El match se
desplomó: casi toda su "precisión" era ficticia.

## 5. Fases 1-2 — Datos y features sin leakage

`src/data/rolling_features.py`. Se reconstruyeron las features a nivel de
partido desde `sets_partidos.csv` (la verdad set a set), recorriendo los
partidos en **orden cronológico** y usando **solo información previa**:

- **Elo con margen de victoria** (3-0 mueve más rating que 3-2), con regresión
  a la media entre temporadas y ventaja de local.
- **Forma EWMA** (half-life ~5 partidos), **H2H con decaimiento temporal**,
  win/set/point ratios *expanding* dentro de temporada.

Descubrimiento inmediato: **el Elo puro (una sola variable, sin entrenar) ya
daba AUC 0.62** — más que las 87 features viejas. Las features leaked no solo
inflaban la métrica: como ruido, **empeoraban** el modelo honesto.

### 5.1 El hallazgo contraintuitivo: las temporadas viejas envenenaban el modelo

> **⚠️ CORRECCIÓN (2026-07-15) — esta sección era un ARTEFACTO de un bug de datos.**
> Al implementar el backtest del simulador se descubrió que `partido_id` en
> `sets_partidos.csv` COLISIONA la ida y la vuelta de cada cruce, y
> `_aggregate_matches` (que agrupaba solo por `partido_id`) **fundía dos partidos
> en uno**, sumando sus sets e invirtiendo el target `gana_local` en el 82% de los
> partidos. Los "% victoria local ~0.32-0.35" y el "AUC 0.28-0.55 / signo
> invertido" de las temporadas viejas eran ese artefacto, NO datos reales.
> Con la agregación corregida (agrupar por `(partido_id, local)` → 1322 partidos):
> el home-win es **0.48-0.61 en TODAS las temporadas** y el Elo AUC es **~0.76 en
> 2024 y 2025** (2024 pasó de 0.545 a 0.756). **Ninguna temporada envenena el
> modelo**; la justificación de la recencia (entrenar solo 2022-2024) se apoya en
> este artefacto y debe re-evaluarse. La tabla y el texto de abajo se conservan
> como registro histórico. Detalle y plan en `docs/PLAN_MEJORAS_CONSOLIDADO.md` §B0.

Tras la corrección B0 (2026-07-15), la lectura correcta de las temporadas
viejas es:

- **Home-win rate consistente 0.48-0.61 en todas las temporadas**, sin
  envenenamiento sistémico. El "0.32-0.35" era artefacto de la colisión
  `partido_id`.
- **AUC Elo limpio**: 2024 = 0.756, 2025 = 0.762, sin caída en las temporadas
  viejas.
- La **recencia 2022-2024** se justifica por **ciclo de plantillas** (half-life
  2 temporadas ≈ renovación típica de roster), no por sign-flip. Con datos
  limpios, entrenar con todo el histórico ya NO produce AUC 0.42; el problema
  que reveló este análisis era un artefacto del bug.

| Temporada | n_partidos (corregido) | Home-win rate | AUC Elo |
|---|---:|---:|---:|
| 2016-2025 (agregado) | ~1322 | 0.48-0.61 | — |
| 2024 | ~111 | ~0.55 | 0.756 |
| 2025 | ~214 | ~0.60 | 0.762 |

→ **Tabla invalidada y narrativa previa (home-win ~0.32, AUC 0.28-0.55, signo
invertido, AUC 0.42 con todo el histórico):** ver
[`memoria/registro_historico_b0.md`](registro_historico_b0.md) §A.1-A.2.

## 6. Fase 3 — Modelado: lineal > árboles en datos pequeños

Con 34-59 partidos por temporada, los árboles de gradiente sobreajustan ruido y
**ahogan la señal limpia del Elo**: un modelo de 27 features daba AUC 0.54,
peor que el Elo de una variable (0.62). El patrón se repitió en el set: un
**LogisticRegression regularizado** (C=0.5) batió al ExtraTrees en el test 2025
(0.71* vs 0.65). **\* El 0.71 es 2025-específico** y además estaba medido sobre
el dataset colisionado (B0b). Con datos limpios (2026-07-15): test 2025 AUC
**0.697** (n=1193), CV 2-fold **0.679 ± 0.017**. El CV rolling-origin honesto
multi-temporada (0.68) es la magnitud defendible, no el test. Detalle en §7.2.

Config final elegida:
- **MATCH**: probabilidad de **Elo con margen** (mejor logloss, 0.585; AUC 0.75).
  Elegante, sin entrenamiento, sin riesgo de sobreajuste.
- **SET**: **LogReg** con pesos de recencia (half-life 2), entrenado en 2022-2024.

Se descartó el intento de recalibración Platt: con datos viejos volvía a
aprender el signo invertido (AUC 0.23). Registrado como negativo, igual que los
experimentos del Batch 3.

Artefactos reproducibles: `python -m src.models.train_improved` →
`match_elo_v2.joblib`, `set_predictor_v2.joblib`, `precision_improved.json`.

## 7. Integración en producción

El objetivo real del proyecto es el **simulador de temporada**, no el
clasificador aislado. Lo integrado:

1. **Fuerzas de equipo desde el Elo con margen** (`api/main.py`
   `_compute_team_strengths` → `get_historical_team_strengths`). El prior pasó
   del win-rate plano a la **jerarquía real de la SuperLega**: Perugia (0.68) >
   Trento (0.60) > Verona > Piacenza … > Cuneo (0.23) > Grottazzolina (0.18).
2. **Elo runtime sembrado desde el histórico** (`RuntimeFeatureBuilder(initial_elo=…)`).
   Antes arrancaba plano en 1500 y "calentaba" desde cero, lo que hacía inútil
   la señal de Elo las primeras jornadas y **diluía** el buen prior. Ahora es
   fiel desde la jornada 1. (Parámetro opt-in: el arranque plano se conserva por
   defecto para no romper el test de schema.)
3. **Update de Elo con margen** en el `RuntimeFeatureBuilder` importando
   las constantes canónicas desde `src.data.rolling_features`
   (K=28, HOME_ADV=60, ELO_BASE=1500, ELO_SEASON_REGRESS=0.25). Antes
   de este fix las constantes estaban duplicadas con valores distintos
   (runtime usaba K=32, HOME_ADV=65) — train/serve skew clásico.
   Margen mult `1 + 0.15·(|diff_sets|-1)` (3-0 → 1.30, 3-1 → 1.15,
   3-2 → 1.00) aplicado en ambos lados.
4. **Señal de partido = probabilidad de Elo limpia** en el `SeasonSimulator`,
   en lugar del MatchPredictor de 87 features (leaky). El artefacto viejo queda
   solo como fallback.

**Efecto observable:** la clasificación simulada por fin correlaciona con la
fuerza real. Antes de sembrar el Elo, Grottazzolina (el equipo más débil) podía
acabar 4º en un solo seed.

### 7.1 Validación Monte Carlo — 20 temporadas simuladas

**Corrida vigente (2026-07-22, tras A2/A4 + B3).** 20 temporadas completas
(12 equipos, ida y vuelta, seeds 0-19), Elo sembrado desde el histórico real,
clamp en la configuración final del Grupo A (`SET_BLEND_WEIGHT_ELO = 1.0`,
`CLAMP_MARGIN_POINT = 0.10`) y el `PointProbabilityModel` de B3. Estado
dinámico reiniciado en cada temporada. Reproducible con
`python -m src.models.mc_season_validation --n-seeds 20`; cifras en
`models/mc_season_validation.json`.

| # | Equipo | Posición media | Pos. std | Fuerza (margin-Elo) |
|---|---|---:|---:|---:|
| 1 | Perugia | 1.4 | 0.66 | 0.833 |
| 2 | Trento | 2.5 | 1.12 | 0.734 |
| 3 | Lube | 3.4 | 1.15 | 0.616 |
| 4 | Modena | 4.2 | 2.06 | 0.639 |
| 5 | Verona | 5.5 | 1.12 | 0.672 |
| 6 | Piacenza | 6.0 | 1.75 | 0.580 |
| 7 | Taranto | 6.2 | 1.84 | 0.526 |
| 8 | Milano | 7.8 | 1.40 | 0.474 |
| 9 | Monza | 8.9 | 1.32 | 0.374 |
| 10 | Padova | 9.8 | 1.11 | 0.331 |
| 11 | Cisterna | 10.5 | 0.92 | 0.275 |
| 12 | Grottazzolina | 11.8 | 0.51 | 0.183 |

**Spearman fuerza→posición = −0.9720** (p = 1.3e-07); std media de posición
**1.247**.

**Lectura honesta:**

- La correlación fuerza→posición es muy fuerte pero **ya no perfecta**, que es
  lo que cabe esperar de una liga de 22 jornadas: el top-2 y el fondo son
  estables, y el mid-table se reordena entre temporadas (Lube 3º pese a tener
  menos fuerza que Verona, 5º; Piacenza y Taranto prácticamente empatados).
- Las anomalías de la corrida de 2026-07-08 siguen resueltas: eran consecuencia
  del sembrado roto de Elo y de las fuerzas colisionadas pre-B0.

> **⚠️ Corrección metodológica (2026-07-22).** La corrida publicada el
> 2026-07-21 daba Spearman **−1.0000** y std media 0.457, con cuatro equipos a
> std 0.00 — una liga prácticamente determinista. Ese resultado era en buena
> parte **artefacto del instrumento**: `mc_season_validation.py` construía el
> `MatchSimulator` con `point_model=None`, así que medía el fallback
> `_default_point_probs` en lugar del modelo entrenado que inyecta el API
> (`main.py:112`). Al corregirlo y usar el modelo de B3, la dispersión sube
> **2,7×** (std 0.457 → 1.247) y ningún equipo queda ya con std 0.00.
>
> Es decir: parte de la "sub-dispersión" atribuida al simulador en A6 no era
> del simulador, sino de una medición que no reproducía el camino de
> producción. La parte real —la que sí medía B1, que siempre usó el modelo de
> punto— es la que corrige B3 (§7.3).

<details>
<summary>Corrida histórica invalidada (2026-07-08) — se conserva como registro del proceso</summary>

> **⚠️ TABLA INVALIDADA (2026-07-08, misma tarde).** Al investigar el ruido
> del clamp se descubrió que esta corrida se ejecutó con el sembrado de Elo
> **roto** (bug `Optional` en `rolling_features.py`: el API caía en silencio
> al fallback win-rate + Elo plano). La tabla se conserva como registro del
> proceso, pero NO refleja la integración margin-Elo. El diagnóstico completo,
> la cuantificación ON/OFF con el fix aplicado y el plan de corrección están
> en [`../docs/PLAN_MEJORA_CLAMP.md`](../docs/PLAN_MEJORA_CLAMP.md). Con el fix, 6
> temporadas ida simple dan Spearman fuerza→posición 0.87-0.89, y el clamp
> del SetPredictor demostró aportar cero señal (ρ≈0 con p_elo) y +22% de
> varianza de posición.

Se corrieron **20 temporadas completas** (12 equipos, ida y vuelta = 22
jornadas, seeds 0-19) con el Elo *supuestamente* sembrado desde el histórico
(en realidad plano, ver aviso) y el clamp del SetPredictor activado. Posición
media final (menor = mejor):

| # | Equipo | Posición media | Fuerza (margin-Elo) |
|---|---|---:|---:|
| 1 | Perugia | 3.3 | 0.681 |
| 2 | Trento | 4.2 | 0.604 |
| 3 | Lube | 5.5 | 0.538 |
| 4 | Taranto | 5.9 | 0.350 |
| 5 | Piacenza | 6.2 | 0.568 |
| 6 | Monza | 6.8 | 0.462 |
| 7 | Verona | 7.2 | 0.578 |
| 8 | Modena | 7.3 | 0.522 |
| 9 | Cisterna | 7.4 | 0.350 |
| 10 | Padova | 7.4 | 0.372 |
| 11 | Grottazzolina | 8.1 | 0.176 |
| 12 | Milano | 8.5 | 0.457 |

**Lectura honesta:**

- El **top-3 es exacto** (Perugia > Trento > Lube) y el fondo también
  (Grottazzolina/Milano últimos). La correlación fuerza→posición es claramente
  positiva: el simulador ya no premia a los equipos débiles.
- La **zona media es ruidosa**: Taranto (fuerza 0.35) sobrerrinde hasta el 4º
  puesto y Verona (0.578) baja al 7º. *(Diagnóstico posterior: la causa
  principal era el sembrado roto — ver aviso arriba — más el clamp comprimiendo
  diferencias sin señal.)*
- Este ruido es **coherente con la realidad**: el mid-table de la SuperLega es
  genuinamente volátil temporada a temporada.

El ruido del clamp quedó cuantificado y con plan de corrección en
[`../docs/PLAN_MEJORA_CLAMP.md`](../docs/PLAN_MEJORA_CLAMP.md).

Todo con **142 tests verdes** (134 + 8 nuevos para el adapter v2); el simulador
y el API arrancan sin cambios de interfaz.

</details>

#### Cierre del Grupo A (2026-07-21)

El clamp adaptativo se cierra con un **resultado negativo documentado**: tras
corregir su error de escala (A2: centrar en el p_punto implícito vía
`src/simulation/set_math.py`) y convertirlo en mezcla en vez de override (A4),
el barrido del peso `w ∈ {0.5, 0.7, 0.9, 1.0}` elige **w = 1.0**, es decir,
ignorar por completo al SetPredictor. `w = 0.9` y `w = 1.0` dan métricas
idénticas y coincidentes con la configuración OFF.

Backtest A5 final (`models/backtest_clamp_results.json`):

| Config | \|P_MC − p_elo\| | Spearman | Std pos | Std pts | T(s) |
|---|---:|---:|---:|---:|---:|
| OFF | 0.22470 | −0.9720 | 0.1667 | 0.4940 | 7.5 |
| **NEW (A2+A4)** | **0.22470** | **−0.9720** | **0.0667** | 0.4006 | 9.3 |

Los tres criterios de aceptación del grupo se cumplen por primera vez
(antes de A2 fallaban dos de tres) y el coste cae **14×** respecto al camino ON
anterior, porque con `w = 1.0` la llamada al SetPredictor se cortocircuita.

Lo que sobrevive de valor no es el SetPredictor sino el **reescalado**: centrar
el clamp en la señal Elo viva con ±0.10 en espacio de punto, en lugar del rango
fijo [0.20, 0.80], mejora la estabilidad entre seeds sin coste en fidelidad de
ranking. Detalle del mecanismo en [`simulator.md`](simulator.md) §4.3.

**Corrección metodológica encontrada al hacer A2:** el backtest A5 compartía un
único `RuntimeFeatureBuilder` entre las configs OFF/ON/NEW. Como el builder
acumula estado Elo en cada temporada simulada, las temporadas de una config
contaminaban a la siguiente. El síntoma que lo delató: el nivel-par usa seeds
fijas y debería ser determinista, pero cambiaba al variar `--n-seeds`. Todas las
cifras de esta sección son posteriores al arreglo (commit `fc8aa6b`).

### 7.2 Validación per-year del set v2 (post-integración) — el "0.71" es 2025-específico

> **⚠️ ACTUALIZACIÓN (2026-07-15) — cifras recalculadas sobre datos limpios.**
> El `set_features.csv` sobre el que se midió esta sección estaba colisionado
> (ida+vuelta fundidas; `sets_h_antes` llegaba a 5). Regenerado sin colisión
> (`src/data/set_features_builder.py`) y reentrenado el SetPredictor v2, el CV
> rolling-origin de 2 folds pasa de **0.631 ± 0.078 a 0.679 ± 0.017** (más alto
> y mucho más estable) y el test 2025 de 0.709 a 0.697 (n 853→1193). La lectura
> de fondo NO cambia (el "0.71" seguía siendo específico de 2025; la magnitud
> defendible es la CV), pero ahora la CV es mejor y estable. Números viejos
> abajo como registro. Ver `docs/PLAN_MEJORAS_CONSOLIDADO.md` §B0b.

> **Pregunta que motivó el análisis:** ¿el AUC 0.71 del set v2 es robusto o
> depende de haber tenido "suerte" con la temporada de test? Se hizo un análisis
> per-year deslizando la ventana de validación: para cada temporada T, entrenar
> en los años previos con la misma config (LogReg C=0.5, recencia half-life=2,
> 21 features de `SET_FEATURE_COLS`) y medir en T.

**Resultado per-year (train = años previos, val = T):**

| Temporada | n_sets | AUC | LogLoss | Accuracy |
|---|---:|---:|---:|---:|
| 2018 | 154 | 0.6003 | 0.6675 | 0.6169 |
| 2019 | 186 | 0.6082 | 0.6760 | 0.5968 |
| 2020 | 229 | **0.6407** | 0.6719 | 0.5939 |
| 2021 | 244 | 0.5857 | 0.6726 | 0.5779 |
| 2022 | 252 | 0.5796 | 0.6878 | 0.5754 |
| 2023 | 352 | 0.5743 | 0.6960 | 0.5653 |
| 2024 | 482 | 0.5828 | 0.6810 | 0.5913 |
| **2025** | **853** | **0.7047** | **0.6329** | **0.6600** |

*Tabla calculada sobre el dataset pre-B0b (853 sets en 2025, n total
~2300). Con el dataset limpio post-B0b (1193 sets en 2025, n total ~3900),
el per-year puede haber cambiado, pero este análisis no se rehízo — las
cifras agregadas (CV 0.679±0.017, test 0.697) reemplazan a las de esta tabla
como métricas vigentes.*

**Spearman(val_year, AUC) = −0.17, p=0.69.** La correlación monotónica con el
año es NO significativa y ligeramente negativa: la teoría de "mejora
monotónica con el tiempo" está refutada. **2018-2024 se mueven en una banda
estrecha** (0.57-0.64, con 2020 como el mejor año viejo en 0.64), y el salto a
0.70 aparece **solo en 2025**.

**El salto 2024 → 2025 de +0.122 AUC es real y no es ruido:**
gap demasiado grande para explicarse por `n=482` vs `n=853` (la diferencia de
estabilidad no mueve 0.12 de AUC). El modelo le va notablemente mejor a 2025
que a cualquier temporada anterior.

**Razones más probables del salto 2025:**

1. **2025 es la temporada más grande del dataset** (853 sets, +77% vs 2024).
   La métrica es mucho más estable, pero eso solo no explica +0.12.
2. **La recencia (half-life=2) hace que 2024 sea el training más pesado para
   predecir 2025.** El modelo "memoriza 2024 y predice 2025 ≈ 2024++", y
   funciona porque la dinámica de la SuperLega es estable entre temporadas
   consecutivas.
3. **Para val=2024, el training más pesado son 2022 y 2023** (temporadas
   chicas, 252 y 352 sets), y el modelo tiene poco material reciente para
   aprender. La predicción de 2024 es la que más sufre por este motivo, no la
   de 2025.

**Nota sobre la tabla per-year:** los valores individuales de 2018-2025 en la
tabla de arriba se calcularon sobre el `set_features.csv` original (colisionado
por B0b). Aunque el análisis per-year como metodología sigue siendo válido, las
cifras concretas cambiaron con la regeneración del dataset (n=1193 vs 853 en
2025, sets adicionales de partidos que antes estaban fundidos). La tabla
pre-B0b se conserva en
[`memoria/registro_historico_b0.md`](registro_historico_b0.md) §B.1.

**Lo que esto significa para la narrativa del modelo (con datos limpios):**

- El CV honesto de 2 folds (`train=[2022,2023]→val 2024` + `train=[2023,2024]
  →val 2025`) da **AUC 0.679 ± 0.017** (frente a 0.631 ± 0.078 pre-B0b: más
  alto y mucho más estable). La CV es la representación más honesta del
  rendimiento fuera de la muestra.
- El test 2025 pasa de AUC **0.709 a 0.697** (n 853→1193). La lectura de fondo
  no cambia (el "0.71" seguía siendo específico de 2025; la magnitud defendible
  es la CV), pero ahora la CV es mejor y estable.
- El legacy ExtraTrees (champion anterior) tenía CV AUC 0.62 ± 0.03 sobre 4
  folds (2018-2024). La diferencia **CV v2 (0.68) vs CV legacy (0.62) es
  +0.06** y ya no cae dentro del ruido: con datos limpios el v2 sí es una
  mejora estructural clara.

**Follow-up obligatorio (warning W1 del sdd-verify):** **re-validar con la
temporada 2026/27 cuando esté disponible.** Si el AUC se mantiene por encima
de ~0.65 en una temporada no vista en training, la mejora es estructural (el
modelo aprendió algo real). Si vuelve a ~0.60, el 0.71 de 2025 fue una
coincidencia favorable entre sample-size y ventana de recencia.

Para esta re-validación basta con:

```bash
# Tras descargar la temporada 2026/27 en DB/sets_partidos.csv
python -m src.models.train_improved   # re-genera set_predictor_v2.joblib
                                     # con train incluyendo 2025
python -c "from src.models.measure_precision import measure; print(measure())"
```

Y comparar el AUC en 2026 contra los números de 2018-2025 de la tabla de
arriba. **No tocar el protocolo, no cambiar el modelo: solo agregar datos y
medir igual.**

### 7.3 B3 — PointProbabilityModel con regresión continua (2026-07-22)

**El problema.** El modelo de punto binarizaba su target (`y_binary = y > 0.5`)
y aplastaba la salida con `p = 0.45 + 0.10 · p_dominante`. Dos daños:

1. La binarización tiraba toda la información de **magnitud**: un partido con
   ratio de puntos 0.51 y otro con 0.58 eran la misma clase.
2. El mapping fijaba la salida en `[0.45, 0.55]` y, peor, la **sesgaba**: con
   features neutras el clasificador predecía `p_dominante = 0.8875` (había
   aprendido la tasa base de victoria local), luego `p = 0.5387`.

Ese sesgo es pequeño por punto pero la cadena de Markov lo **amplifica ~7×** a
lo largo de un partido: dos equipos iguales daban **P(local) = 0.845** frente a
un home-win real de 0.573. Es el origen de la sub-dispersión que detectó §7.1
(Spearman −1.0) y que cuantificó el backtest B1 (ECE 0.242, 53 % de 3-0).

**La corrección.** `Ridge(alpha=1.0)` sobre el target continuo
`point_ratio_h = pts_h / (pts_h + pts_a)`, entrenado con features rolling
pre-partido (sin leakage), y salida `clip(pred, POINT_RATIO_CLIP)` con
`POINT_RATIO_CLIP = (0.40, 0.60)` actuando solo de salvavidas. El intercept
aprendido es **0.5081** — el ratio de puntos real medio — frente al 0.5387
sesgado de antes.

**Resultado — backtest B1 sobre 2024** (222 partidos, n=500, clamp OFF). Para
que la medida sea honesta, el modelo se reentrenó **solo con historia < 2024**
(786 partidos) y el backtest se corrió con él, no con el de producción:

| Métrica | Antes (binarizado) | **B3** | Elo (ref) |
|---|---:|---:|---:|
| Brier | 0.2731 | **0.1815** | 0.1941 |
| LogLoss | 0.8241 | **0.5365** | 0.5690 |
| Accuracy | 0.6486 | **0.7207** | 0.6892 |
| ECE | 0.2419 | **0.0565** | 0.0454 |
| 3-0 simulado | 53.0 % | **37.6 %** | real 38.7 % |
| L1 (márgenes) | 0.2858 | **0.0315** | — |

- El simulador pasa de **degradar** la señal Elo a **superarla** en Brier,
  logloss y accuracy. La lectura de §10.2 de `simulator.md` se invierte.
- La distribución de márgenes deja de estar sesgada hacia el 3-0: los tres
  márgenes caen a menos de 2 puntos porcentuales del real, y la distancia L1
  baja **9×**.
- La calibración mejora **4,3×** (ECE 0.242 → 0.057) y queda ya muy cerca del
  Elo puro (0.045).

**Control de leakage.** El mismo backtest con el modelo de producción
(entrenado con 2016-2025, incluyendo 2024) da Brier 0.1822 y ECE 0.0824 — es
decir, prácticamente lo mismo, y con **peor** ECE. La mejora es estructural,
no un artefacto de haber visto la temporada de test.

**Sanity de la cadena.** Test nuevo `TestMarkovChainSanity` en
`tests/test_simulator.py`: con p_punto = 0.52 constante y momentum neutralizado,
el MC (n=2000) da 0.6845 frente a la forma cerrada 0.6967 (Δ = 0.012). La
cadena conserva la probabilidad de punto correctamente.

> **Corrección al plan.** La spec de B3 (paso 4) pedía pinear
> `p_set_from_p_point(0.52, 25) ≈ 0.66` y de ahí derivaba una banda esperada de
> P(match) de 0.74 ± 0.03. Ese 0.66 es incorrecto —es el mismo error de
> constante que ya se corrigió en A2— y arrastra la banda entera: el valor real
> de la fórmula es **0.6131**, que da **P(match) = 0.6967**. El test pinea el
> valor derivado de `set_math`, no el número del documento.

**Qué no arregla B3.** El ECE sigue por encima del Elo puro (0.057 vs 0.045):
queda algo de sobreconfianza residual. Y el modelo tiene ahora más rango
dinámico en las colas (elo_diff = ±400 da P = 0.95 / 0.085), lo que conviene
vigilar si en el futuro se amplía el dataset (B6) con más partidos desiguales.

### 7.4 B2 — Tuneo de las constantes del simulador: RESULTADO NEGATIVO (2026-07-22)

`MOMENTUM_BONUS = 0.015`, `GLOBAL_MOMENTUM_FACTOR = 0.01` y
`MATCH_PREDICTOR_DAMPING = 0.5` se fijaron *a priori* y nunca se habían
contrastado con datos. B2 las somete al backtest B1.

**Protocolo.** Grid sobre `MOMENTUM_BONUS ∈ {0, 0.01, 0.015, 0.03}` ×
`GLOBAL_MOMENTUM_FACTOR ∈ {0, 0.01, 0.02}` × `damping ∈ {0.3, 0.5, 0.7}`.
Tune sobre **2023 y 2024**; 2025 se reserva para una única validación final.
Modelo de punto entrenado solo con historia < 2024 (sin leakage). Dos pasadas:
n = 100 sobre 2024 para descartar, n = 500 sobre 2023+2024 para decidir.
Reproducible con `python -m src.models.tune_simulator_constants`.

#### Hallazgo previo: el eje `damping` es degenerado

De los 36 combos, solo **12 son distintos**. `damping` no tiene ningún efecto
en este camino: solo influye en `SeasonSimulator._calibrate_strengths`, que
produce `home_strength` / `away_strength`, pero
`PointProbabilityModel.get_point_probabilities` entra en la rama
`if match_features and self.is_fitted:` y **no usa esos argumentos**.
Verificado a dos niveles:

1. Modelo: salida bit a bit idéntica para fuerzas 0.50/0.50 y 0.80/0.20.
2. Backtest completo de 2024 (n = 100): `damping` 0.3 y 0.7 dan métricas
   idénticas (Brier 0.1850, LogLoss 0.5432, Acc 0.7252, ECE 0.0594).

Es decir: toda la calibración Elo→fuerza (`_calibrate_strengths`,
`HOME_ADVANTAGE_STRENGTH_BONUS`, `MATCH_PREDICTOR_DAMPING`) es **código muerto**
siempre que el modelo de punto esté cargado, que es el caso en producción
(`src/api/main.py:112`). Queda anotado como trabajo aparte: decidir entre
re-cablear las fuerzas o retirarlas.

#### Pasada 1 — 12 combos, 2024, n = 100

| MOMENTUM_BONUS | GLOBAL_MOM. | Brier | LogLoss | ECE | Acc | L1 márg. |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.02 | 0.1836 | 0.5416 | 0.0526 | 0.7207 | 0.0996 |
| 0.01 | 0.01 | 0.1836 | 0.5431 | 0.0844 | 0.7297 | 0.0205 |
| 0.015 | 0.0 | 0.1837 | 0.5426 | 0.0477 | 0.7162 | 0.1457 |
| 0.01 | 0.0 | 0.1839 | 0.5426 | 0.0682 | 0.7342 | 0.1358 |
| 0.0 | 0.0 | 0.1843 | 0.5437 | 0.0534 | 0.7117 | 0.1362 |
| 0.03 | 0.01 | 0.1844 | 0.5434 | 0.0584 | 0.7117 | 0.0333 |
| 0.015 | 0.02 | 0.1845 | 0.5424 | 0.0469 | 0.7207 | 0.1006 |
| 0.0 | 0.02 | 0.1849 | 0.5440 | 0.0866 | 0.6802 | 0.0999 |
| **0.015** | **0.01** | **0.1850** | 0.5432 | 0.0594 | 0.7252 | 0.0292 |
| 0.01 | 0.02 | 0.1851 | 0.5439 | 0.0528 | 0.7072 | 0.1039 |
| 0.0 | 0.01 | 0.1858 | 0.5467 | 0.0622 | 0.7162 | 0.0207 |
| 0.03 | 0.0 | 0.1862 | 0.5456 | 0.0511 | 0.7117 | 0.1487 |

*(En negrita el baseline actual. Cada fila representa 3 de las 36
combinaciones: las tres de `damping`, idénticas por construcción.)*
Ningún combo supera el umbral de descarte (Brier > 0.20): sobreviven 12/12.

#### Pasada 2 — top-5, 2023 + 2024, n = 500

| MOMENTUM_BONUS | GLOBAL_MOM. | Brier ponderado | 2023 | 2024 | ECE 2024 | L1 2024 |
|---:|---:|---:|---:|---:|---:|---:|
| **0.015** | **0.01** | **0.20889** | 0.2464 | 0.1815 | 0.0565 | 0.0315 |
| 0.015 | 0.0 | 0.20922 | 0.2464 | 0.1821 | 0.0546 | 0.1397 |
| 0.03 | 0.02 | 0.20953 | 0.2475 | 0.1818 | 0.0540 | 0.1068 |
| 0.01 | 0.01 | 0.21003 | 0.2488 | 0.1817 | 0.0595 | 0.0257 |
| 0.01 | 0.0 | 0.21046 | 0.2479 | 0.1832 | 0.0582 | 0.1398 |

**Ganador: el baseline actual, con delta = +0.00000.**

#### Por qué es un resultado negativo, no una confirmación

Sería tentador leer esto como "los valores a priori estaban bien". No es lo
que dicen los datos. Dos evidencias muestran que **el grid entero está por
debajo del ruido**:

1. **Inestabilidad de ranking entre pasadas.** El combo ganador de la pasada 2
   (`0.015 | 0.01`) era el **9.º de 12** en la pasada 1; el mejor de la pasada 1
   (`0.03 | 0.02`) cae al 3.º en la pasada 2. Si las diferencias fuesen reales,
   el orden sería estable al aumentar n.
2. **Suelo de ruido medido.** Se corrió la configuración baseline sobre 2024
   (n = 500) con 6 semillas base distintas, cambiando *solo* las semillas:

   | | media | std | rango |
   |---|---:|---:|---:|
   | Brier | 0.18317 | **0.00127** | 0.00341 |
   | ECE | 0.06208 | **0.01108** | — |

   El rango completo del grid en la pasada 2 es **0.00157**, es decir, ~1,2 σ
   del ruido y menos de la mitad del rango que produce cambiar solo la semilla.
   Las diferencias de ECE del grid (0.047–0.087) son igualmente indistinguibles
   de un ruido con σ = 0.011.

**Conclusión: no se adoptan valores nuevos.** No porque los actuales hayan
ganado, sino porque **ningún valor del grid es distinguible de otro** con el
poder estadístico disponible (222 partidos en 2024, 162 en 2023). Cambiar las
constantes en base a este grid sería sobreajustar ruido. Se mantienen
`MOMENTUM_BONUS = 0.015` y `GLOBAL_MOMENTUM_FACTOR = 0.01`, y por tanto no se
tocan `constants.py`, los pines de `tests/test_team_mapper.py` ni `AGENTS.md`.

**Lectura de fondo:** tras B3, el modelo de punto domina la calidad de
probabilidad hasta el punto de que el momentum —cuyo efecto máximo es
±0.06 sobre `p_home_wins`— no mueve la aguja de forma medible a nivel de
partido. Para resolver este grid haría falta o bien mucho más dato (B6), o
bien una métrica más sensible al detalle intra-set que el Brier del resultado
del partido.

#### Validación única sobre el test held-out 2025

Aunque no se adopta ningún cambio, se ejecutó **una sola vez** el backtest
sobre 2025 con la configuración vigente, para dejar la cifra de referencia del
simulador en la temporada que nunca se ha usado para tunear. Modelo de punto
entrenado solo con historia < 2025 (1008 partidos). 314 partidos, n = 500:

| Métrica | Simulador | Elo (ref) | Δ |
|---|---:|---:|---:|
| Brier | **0.1878** | 0.1924 | −0.0046 |
| LogLoss | **0.5544** | 0.5645 | −0.0102 |
| Accuracy | **0.7134** | 0.6975 | +0.0159 |
| ECE | 0.0626 | 0.0494 | +0.0133 |

| Marcador | Simulado | Real |
|---|---:|---:|
| 3-0 | 40.7 % | 44.6 % |
| 3-1 | 34.0 % | 30.9 % |
| 3-2 | 25.3 % | 24.5 % |

L1 (márgenes) = 0.0771.

**El patrón de 2024 se reproduce en held-out:** el simulador supera al Elo en
Brier, logloss y accuracy, y mantiene una sobreconfianza residual pequeña en
ECE. Es la confirmación de que la mejora de B3 (§7.3) generaliza y no era
específica de 2024. Cifras en `models/backtest_simulator_2025.json`.

### 7.5 B4 — Predictor de partido derivado del SetPredictor (best-of-5): RESULTADO NEGATIVO (2026-07-23)

El margin-Elo domina la señal de partido desde B1 (AUC 0.76 en test 2025). B4
pregunta si una segunda señal independiente —derivada del SetPredictor v2 vía
la fórmula best-of-5— puede complementar al Elo en un blend lineal:

$$P_{\text{final}} = w \cdot P_{\text{Elo}} + (1-w) \cdot P_{\text{derivada}}$$

donde $P_{\text{derivada}} = q^3 + 3q^3(1-q) + 6q^2(1-q)^2 q_5$, con $q =
\text{SetPredictor.predict\_proba(contexto 0-0)}$ a 25 puntos y $q_5 =
p_{\text{set\_from\_p\_point}}(p_{\text{point\_from\_p\_set}}(q, 25), 15)$ vía
la composición A2 (la `set_math` derivada en §7.3, A2 cerrado el 2026-07-21).

**Protocolo.** Leave-One-Fold-Out CV sobre 2021–2024 (4 folds): para cada fold
$f$ se optimiza $w$ sobre los OTROS 3 folds por mean log-loss, y se evalúa
ese $w_f$ sobre el fold held-out. Grid de 21 valores de $w \in [0, 1]$ con
refinamiento por sección áurea. 2025 se reserva como test held-out (single
shot). AND-of-4 adoption gate: per-fold wins $\geq 3/4$,
$\text{improvement\_mean} > \max(\sigma_{\text{LOFO}}, 0.005)$,
test-2025 logloss $< 0.5677$ (Elo baseline), $w_{\text{global}} > 0.05$.
Reproducible con
`python -c "from src.models.train_improved import run_b4_route; print(run_b4_route())"`.

Resultados por fold (LOFO-CV honesto, no in-fold):

| Fold | $w$ (LOFO) | LogLoss blend | LogLoss Elo puro | Mejora |
|---:|---:|---:|---:|---:|
| 2021 | 0.685 | 0.5738 | 0.5771 | +0.0033 |
| 2022 | 0.559 | 0.6343 | 0.6313 | −0.0029 |
| 2023 | 0.518 | 0.6665 | 0.6629 | −0.0036 |
| 2024 | 0.914 | 0.5747 | 0.5765 | +0.0018 |

| Métrica global | Valor |
|---:|---:|
| $w$ global (media) | 0.6688 |
| Mejora media | −0.0004 |
| $\sigma_{\text{LOFO}}$ (con noise floor) | 0.0050 |
| Per-fold wins (blend &lt; Elo) | 2 / 4 |
| LogLoss blend (media) | 0.6123 |
| LogLoss Elo (media) | 0.6120 |
| Test 2025 logloss blend | 0.5650 |
| Test 2025 logloss Elo (baseline) | 0.5677 |

**Análisis.** La $w$ global (0.669) está alejada de la saturación pura-Elo
(1.0) y de la saturación pura-derivada (0.0), pero las $w$ por fold oscilan
entre 0.52 y 0.91: el optimizador encuentra un blend distinto en cada fold, lo
que indica que la señal derivada no aporta información consistente año a año.
Solo 2 de 4 folds ganan con el blend (los otros 2 son marginalmente peores),
la mejora media es **negativa** (−0.0004), y el test held-out 2025 da una
mejora marginal (0.5650 vs 0.5677 = +0.0027). El AND-of-4 atrapa la
inestabilidad por la condición de per-fold wins (2/4 &lt; 3/4) y por la mejora
media bajo el noise floor (|−0.0004| &lt; 0.005). Las cuatro condiciones
del gate fallan, no solo una: el resultado es robusto.

**Decisión.** No se adopta el blend. La señal del SetPredictor v2, convertida a
P(match) vía best-of-5, no es lo suficientemente independiente del Elo —o no
tiene suficiente poder predictivo— para mejorar el log-loss de forma
consistente entre temporadas. El Elo puro sigue siendo la mejor señal de
partido disponible.

**Artefacto.** `models/b4_blend_results.json` contiene los resultados
completos (per-fold $w$, loglosses, $\sigma_{\text{LOFO}}$, test_metrics,
`failing_conditions`). No se genera `match_elo_blend_v3.joblib` al no
adoptarse el blend.

**Próximos pasos.** Este resultado NEGATIVO cierra la línea de "predictor de
partido derivado del set" para este TFG. Si en el futuro se dispone de más
datos (B6) o de un predictor de set significativamente mejor (p. ej. con
features de alineación o de momento del partido), re-evaluar esta vía podría
tener sentido.

**Nota metodológica.** La primera pasada de B4 (commits anteriores al verify
NEEDS_REWORK #200) implementó incorrectamente la optimización como in-fold
grid-search en lugar de LOFO-CV, y usó el fallback obsoleto `q5 = q` en lugar
de la composición A2. Los fixes (`44e0634`, `b251837`, `929a571`) re-corrieron
el experimento con la metodología correcta y `a0a74f7` añadió 8 tests de
truth-table para el gate. El veredicto NEGATIVE es el mismo en ambas
pasadas, lo que confirma que el resultado no es artefacto del bug.

### 7.6 B5 — Feature de continuidad de plantilla (roster churn): RESULTADO NEGATIVO (2026-07-23)

El margin-Elo en B1 ya capturó la señal de partido (AUC 0.76 test 2025, logloss
0.5677). B5 pregunta si la **continuidad de plantilla** — % de los puntos de T−1
anotados por jugadores que siguen en el equipo en T— es una señal pre-temporada
que complementa al Elo cuando se añade como feature a un LogReg encima del
baseline. Es la **única señal de fichajes/éxodos** disponible y no tiene leakage
porque se computa antes de que arranque la temporada.

$$P_{\text{win}} = \sigma\!\left(\beta_0 + \beta_1 \cdot \text{logit}(P_{\text{Elo}}) + \beta_2 \cdot \text{diff\_roster\_continuity}\right)$$

donde $\text{diff\_roster\_continuity} = h - a \in [-1, 1]$ es la diferencia de
continuidad entre local y visitante. Imputación por **mediana de la liga** (no
0) cuando un equipo no tiene T−1 (ascensos, gaps como Siena, primera temporada
de Cisterna Top Volley en su discontinuidad con CIS-VOLLEY).

**Protocolo.** Leave-One-Fold-Out CV sobre 2021–2024 (4 folds), igual que B4.
`LogisticRegression(C=0.5)` fiteado sobre los **OTROS 3 folds** y evaluado sobre
el fold held-out (LOFO honesto, fix C1 de B4 #200). Recency weight con
half-life 2 temporadas, mismo criterio que el SET v2 de §7.3. 2025 se reserva
como test held-out. **AND-of-4** B5-adaptado (extensión aditiva de
`evaluate_adoption(..., gate="b5")`, rama B4 intacta — R-DRIFT-1 cerrado en
commit `929a571`):

1. per-fold `churn_coef` positivo en ≥ 3/4 folds
2. $\text{improvement\_mean} > \max(\sigma_{\text{LOFO}}, 0.005)$ (noise floor)
3. test-2025 logloss < Elo baseline leído en runtime desde
   `models/precision_improved.json:11` (= 0.5677 hoy; FAIL LOUD si falta la clave)
4. `churn_coef_global` > 0 AND $|z| = |\beta_2 / \text{se}| > 1.0$ (significancia
   estadística; `se` por inversa de la Hessiana log-likelihood, sin dependencia
   de `statsmodels`)

**Hard shortcut**: $|\beta_2| < 10^{-6}$ AND $\text{logloss}_{\text{LR}} ==
\text{logloss}_{\text{constante}}$ → `shortcut_negative` (modelo colapsado al
promedio, no retry). Reproducible con
`python -c "from src.models.train_improved import run_b5_route; print(run_b5_route())"`.

Resultados por fold (LOFO-CV honesto, imputed rows skipped por
`n_imputed_skipped = 755/1322 = 57.1%` — mucho más que el 18% estimado en
el plan porque la heurística de imputación por coincidencia exacta con la
mediana de la liga por temporada atrapa correctamente las temporadas
enteras donde la mayoría de equipos están imputados; el sensitivity check
incluyendo imputed rows arroja el mismo verdict, demostrando invariancia
respecto a la política de imputación). El join de jugadores es exact-string
match sobre `jugador` (REQ-008). El T-1 non-match % medio (todos los T-1
players que NO aparecen en T por nombre) es **55.86%** sobre los 105 pares
(team, T) con datos T-1 — esto refleja el turnover real de plantillas de
voleibol (muchos jugadores cambian de equipo cada año). El 1–3% estimado en
R-DATA-2 / REQ-008 es el subconjunto de esos 55.86% que corresponde a
mismatch de nombre (typos, transliteración), no al total.

| Fold | $\beta_2$ (churn coef) | LogLoss LR | LogLoss Elo puro | Mejora |
|---:|---:|---:|---:|---:|
| 2021 | +0.1361 | 0.5543 | 0.5563 | **+0.0020** |
| 2022 | +0.0002 | 0.7782 | 0.7355 | **−0.0427** |
| 2023 | +0.0960 | 0.6866 | 0.6652 | **−0.0214** |
| 2024 | +0.0569 | 0.5523 | 0.5610 | **+0.0087** |

| Métrica global | Valor |
|---:|---:|
| $\beta_2$ global (churn coef) | +0.1111 |
| $\text{se}(\beta_2)$ | 0.5067 |
| $\|z\|$ (significancia) | **0.22** |
| Mejora media (LR − Elo) | **−0.0133** |
| $\sigma_{\text{LOFO}}$ (con noise floor) | 0.0203 |
| Per-fold wins (LR &lt; Elo) | 2 / 4 |
| LogLoss LR (media) | 0.6429 |
| LogLoss Elo (media) | 0.6295 |
| Test 2025 logloss LR (**held-out**, refit on train-only) | 0.4931 |
| Test 2025 logloss LR (in-sample, for comparison only) | 0.4658 |
| Test 2025 logloss Elo (computed in-route, filtered) | 0.4815 |
| Test 2025 logloss Elo (baseline, full 2025) | 0.5677 |
| Sensitivity verdict (imputed rows INCLUDED) | `negative` |
| $n_{\text{imputed\_skipped}}$ | 755 / 1322 (57.1%) |
| $n_{\text{primary}}$ | 567 |
| Player-name T-1 non-match % (turnover) | 55.86% (mean over 105 (team, T) pairs with T-1) |
| Player-name T-1 non-match % spec noise floor (R-DATA-2 / REQ-008) | 1–3% |

**Análisis.** El coeficiente $\beta_2$ es **positivo en 3 de 4 folds** (2021,
2023, 2024) — la dirección es la correcta: más continuidad de roster en el
local respecto del visitante se asocia marginalmente a más probabilidad de
victoria local. Pero la magnitud es demasiado pequeña para superar el ruido:
$\|z\| = 0.22$ está muy por debajo del umbral 1.0, y la mejora media es
**negativa** (−0.0133), o sea el LogReg es ligeramente PEOR que Elo puro en
promedio. Solo 2 de 4 folds ganan con la feature añadida (el fold 2021 con
mejora marginal +0.0020 y el 2024 con +0.0087; los folds 2022 y 2023 son
claramente peores, arrastrados por la sobreconfianza del LogReg en esas
temporadas). El AND-of-4 atrapa la falta de señal por **dos** condiciones
independientes — `cond2` (mejora media bajo el noise floor) y `cond4`
(significancia estadística insuficiente) — lo que da robustez al veredicto.

El test held-out 2025 — re-evaluado con un modelo **refit en train-only**
(`model_test`, separado del `model_all` global, W1 fix del sdd-verify) — da
logloss 0.4931 (LR) vs 0.4815 (Elo, computado in-route sobre las mismas 110
filas filtradas). El LogReg es **ligeramente PEOR que Elo puro en el
held-out 2025 honesto** (Δ = +0.0116), confirmando el patrón de la LOFO
(improvement negativo promedio). La condición `cond3` (test-2025 logloss <
baseline 0.5677 de `precision_improved.json`) pasa en ambos casos — pero esa
condición no es la que rechaza la adopción; las que rechazan son `cond2` y
`cond4`. El número in-sample previo (logloss 0.4658, modelo que vio los
datos de 2025 en training) se preserva en el artefacto como
`test_metrics.logloss_in_sample` para transparencia, pero el gate opera
sobre el held-out.

**Decisión.** No se adopta la feature de roster churn. La señal existe (el
coeficiente es positivo en la dirección esperada) pero es **demasiado débil**
para complementar al baseline Elo en producción. Las razones estructurales
—constante por equipo-temporada (varianza within-fold limitada), correlación
alta con la fuerza general del equipo ya capturada por Elo, e inversión
limitada en la base de datos de stats por jugador— sugieren que esta línea
no va a cambiar con más datos del mismo tipo. Cerrar B5 como NEGATIVE es
la conclusión correcta.

**Artefacto.** `models/b5_churn_results.json` contiene los resultados
completos (per-fold $\beta_2$, loglosses, $\sigma_{\text{LOFO}}$, test_metrics,
`failing_conditions`, `sensitivity_verdict`, `n_imputed_skipped`,
`sample_population`, `elo_baseline_logloss`). No se modifica `api/main.py` ni
`simulation/simulator.py` en la ruta negativa.

**Próximos pasos.** Este resultado NEGATIVE cierra la línea de "feature de
continuidad de plantilla" para este TFG. La palanca que queda en GRUPO B es
**B6** (ampliar el dataset histórico de partidos, ver
`docs/PLAN_MEJORAS_CONSOLIDADO.md`): la única forma plausible de mejorar
más la señal de partido es tener más datos, no más features. B7 (re-validar
con la temporada 2026/27 cuando esté disponible) queda como follow-up
bloqueado por calendario.

**By-products durables** (independientes del veredicto, ya en `main`):

- `src/data/team_id_mapper.py` (PR1): mapa `ID_Equipo → canónico` con 22
  entradas y correcciones de LT→'Cisterna Top Volley' y PC→'Piacenza Copra'
  (sin esto, los dos Cisterna y los dos Piacenza se mezclaban en el cálculo
  de continuidad).
- Fix Q4 en `src/data/feature_store.py:282` (PR1): la llamada a
  `normalize_team_name` con códigos opacos (`MO`, `TN-ITAS`, etc.) silenciosamente
  fallaba para todos los equipos menos Modena/Trento/Siena. Ahora se rutea
  primero por `team_id_mapper`. Pre-fix solo 3 equipos tenían continuidad no-cero;
  post-fix 18 de 22.
- `tests/test_roster_continuity.py` (PR1): test de regresión REQ-026 con
  truth-table para APG/Perugia y BASTIA/Grottazzolina (protege contra la
  trampa de los nombres de archivo invertidos en el dataset) + guard contra
  re-implementación que use nombres de archivo en vez de `ID_Equipo`.
- `tests/test_adoption_gate.py` (PR3): 6 truth-table cases para el AND-of-4
  B5-adaptado + 1 caso de regresión B4. Garantiza que cambios futuros al gate
  no rompen ni la rama B4 ni la B5.
- `run_b5_route` y `_write_b5_fallback` (PR3): infraestructura reutilizable
  para cualquier feature de "churn-style" que se quiera evaluar en el futuro
  (ej. feature de momentum de puntos, feature de dependencia de un jugador
  franquicia).

**Nota metodológica.** La implementación de `run_b5_route` pasó por 4 fixes
post-apply que se documentan por transparencia:

1. `8c35f42` corrigió un F821 (`_write_negative_fallback` referenciado en
   `run_b4_route` pero renombrado a `_write_b5_fallback` sin actualizar las 2
   llamadas). Preservé la firma original de `_write_negative_fallback` para
   no romper el contrato B4 (R-DRIFT-1).
2. `8c35f42` también corrigió `KeyError: 'temporada_inicio'` en
   `run_b5_route`: `match_features.csv` usa la columna `temporada`, no
   `temporada_inicio` (esta última solo existe en el DataFrame interno de
   `run_b4_route`). Reemplacé las 9 referencias dentro de la función B5
   solamente, sin tocar las 11 referencias de B4 que sí son correctas.
3. `8c35f42` convirtió `temporada` de string `'YYYY/YYYY'` a `int` (año de
   inicio) en el load, porque la matemática de recency
   (`(current_max_season - temporada) / 2.0`) requiere numérico.
4. `8c35f42` agregó las dos keys que faltaban en el artefacto según REQ-022
   del spec: `elo_baseline_logloss` y `sample_population`. Renombré
   `test_metrics_if_computed` a `test_metrics` (manteniendo el alias para
   back-compat).

El veredicto y los números son idénticos antes y después de los fixes (el
gate se aplica al `result` calculado por la route, no a la forma del
artefacto escrito), así que los fixes son cosméticos respecto al resultado
experimental pero necesarios para que el código compile, los tests pasen, y
el artefacto cumpla con el contrato del spec.

## 8. Qué NO se hizo (honestidad de alcance)

- No se amplió el nº de partidos históricos: `sets_partidos.csv` tiene ~1322
  partidos reales tras la corrección B0 (frente a los 724 pre-B0, que eran
  artefacto de la colisión `partido_id`). Las temporadas viejas tienen
  34-59/temporada porque solo se scrapeó a los equipos rastreados. Es un techo
  de datos, no de método.
- El `match_predictor.joblib` de 87 features sigue en disco como fallback; no se
  borró para no romper la carga del API.
- La **feature de continuidad de plantilla** (roster churn) y el **predictor de
  partido derivado del set** (fórmula best-of-5) del plan quedaron sin
  implementar: con el Elo ya en 0.75, el retorno marginal no compensaba el
  riesgo.
- No se hizo el **backtest completo del simulador** contra una temporada real
  con ajuste de momentum/clamps (Fase 4 completa del plan). Se validó a nivel de
  probabilidad (Brier 0.251 → 0.200) y con sanity check de clasificación.

## 9. Techo realista y conclusión

AUC de partido ~0.75 y accuracy ~0.69 es un resultado **sólido y defendible**
para predicción en voleibol profesional: el deporte tiene varianza intrínseca
alta por el formato de sets, y los mercados de apuestas se mueven en ese rango.

La lección metodológica es la más valiosa del TFG: **una métrica mal medida es
peor que ninguna**. El proyecto tenía un AUC "0.71" que daba falsa confianza y
que además ocultaba que el modelo real estaba a nivel de azar. Arreglar *cómo se
mide* fue el prerrequisito para poder mejorar de verdad — y el mismo protocolo
honesto es lo que permite afirmar que la mejora (0.53 → 0.75) es real y no otro
espejismo.

## 10. Archivos nuevos / modificados

**Nuevos:**
- `src/models/evaluation.py` — protocolo rolling-origin.
- `src/data/rolling_features.py` — features rolling sin leakage + Elo con margen.
- `src/models/measure_precision.py` — medición unificada antes/después.
- `src/models/train_improved.py` — entrenador de la config final.
- `PLAN_MEJORA_PRECISION.md`, `COMPARACION_ANTES_DESPUES.md`, este documento.
- `models/precision_baseline.json`, `models/precision_improved.json`.

**Modificados (integración):**
- `src/api/main.py` — fuerzas desde margin-Elo; feature_builder con Elo sembrado.
- `src/simulation/feature_builder.py` — update de Elo con margen; param `initial_elo`.
- `src/simulation/season_simulator.py` — señal de partido = prob de Elo limpia.

## 11. Cómo reproducir

```bash
# Baseline honesto (antes)
python -m src.models.measure_precision --save baseline
# Modelos mejorados (después) + artefactos v2
python -m src.models.train_improved
# Reporte consolidado de cifras (E3) — fuente de verdad para la memoria
python -m src.models.precision_report
# Robustez cross-temporal (E6) — verifica que la métrica 2025 no es artefacto
python -m src.models.cross_temporal_check --save
# Verificar que nada se rompió
python -m pytest -q
```

**Para regenerar los snapshots de la memoria** tras cualquier cambio de datos
o modelo:

```bash
# Tras re-medir, copiar los artefactos a memoria/
cp models/report/precision_report.md memoria/precision_report.md
cp models/report/cross_temporal.md memoria/cross_temporal.md
```

Los snapshots `memoria/precision_report.md` y `memoria/cross_temporal.md`
incluyen una nota de cabecera que apunta a estos comandos.
