# Marco Teórico — Predictor(2)

> **Propósito de este documento.** Aunar en un único lugar los conceptos
> matemáticos y de ML que sustentan el TFG. Es la fuente única de teoría para
> el **Cap 2 (Marco teórico)** de la memoria LaTeX — evita tener que extraer
> definiciones dispersas de `mejora_precision_2026-07.md`, `match_predictor.md`,
> `INDICE.md`, etc.
>
> Estructura: 7 secciones (modelos de Markov, sistemas Elo, ML supervisado,
> validación temporal, métricas, calibración, Monte Carlo). Cada sección
> incluye las definiciones formales, la justificación de su uso en el proyecto
> y las referencias cruzadas a los `.md` de `memoria/` donde se aplica.

---

## 1. Modelado de eventos discretos

### 1.1. Cadenas de Markov de tiempo discreto

Una **cadena de Markov** es un proceso estocástico $\{X_t\}_{t \in \mathbb{N}}$ con espacio de estados finito $\mathcal{S}$ que satisface la **propiedad de Markov** (memoria de orden 1):

$$P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \ldots, X_0 = i_0) = P(X_{t+1} = j \mid X_t = i) =: p_{ij}$$

La dinámica queda definida por la **matriz de transición** $P = (p_{ij})_{i,j \in \mathcal{S}}$, con $p_{ij} \geq 0$ y $\sum_j p_{ij} = 1$. Una trayectoria de longitud $n$ tiene probabilidad $\prod_{t=0}^{n-1} p_{X_t, X_{t+1}}$.

En el límite, las cadenas irreducibles y aperiódicas convergen a una **distribución estacionaria** $\pi$ que satisface $\pi P = \pi$, independientemente de la distribución inicial.

### 1.2. Aplicación a la simulación deportiva

En `Predictor(2)`, cada **rally** (punto) es un ensayo de Bernoulli donde el estado relevante es `(saca, racha, marcador_parcial)`. La propiedad de Markov se asume en su versión *condicional al estado*: $P(\text{punto local} \mid \text{marcador}, \text{saca}, \text{rachas})$ no depende de la historia más allá de estos tres factores. Esto es estándar en simulación deportiva de tenis, voleibol, badminton y se remonta a [Simonian et al. (1998)](https://example.com) y [Sackmann (2015)](https://example.com).

**Limitación reconocida**: la asunción ignora fatiga, cambios tácticos mid-set, momentum emocional. La propia sección de conclusiones del TFG enumera esto como limitación atacable (ver `memoria/simulator.md` §10).

### 1.3. Cálculo de probabilidad de set cerrado (forma analítica)

Dada $p = P(\text{local gana un punto})$, la probabilidad de ganar un set a 25 puntos (con勝 por 2) tiene forma cerrada. En `src/simulation/set_math.py` (introducido en A2 del plan consolidado) la función `p_set_from_p_point(p, target=25)` la implementa por convolución de binomiales con la regla de勝 por 2. Para el tie-break a 15 puntos se usa `target=15`.

Esta forma cerrada es el **gold standard** que valida la cadena de Markov del simulador: el test `TestMarkovChainSanity` (`tests/test_simulator.py`, B3) fija `p = 0.52`, simula n=2000 veces y compara con la forma cerrada (Δ ≈ 0.012 sobre 0.6967).

---

## 2. Sistemas de rating deportivo

### 2.1. Sistema Elo clásico

El **sistema Elo** ([Elo, 1978](https://example.com)), originalmente para ajedrez, asigna a cada jugador/equipo un rating $R \in \mathbb{R}$ que se actualiza tras cada partido según el resultado observado vs. el esperado:

$$R' = R + K \cdot (s - E[s])$$

donde $s \in \{0, 1\}$ es el resultado observado (0 = pierde, 1 = gana), $K > 0$ es el factor de actualización (típicamente 20-32 en ligas profesionales) y $E[s]$ es la probabilidad esperada de victoria según la fórmula logística:

$$E[s] = \frac{1}{1 + 10^{(R_{\text{rival}} - R_{\text{local}})/400}}$$

El rating se interpreta como una diferencia de fuerza: si $R_{\text{local}} - R_{\text{rival}} = 200$, entonces $E[s] \approx 0.76$, el favorito gana 3 de cada 4 partidos en el largo plazo.

### 2.2. Elo con margen de victoria (FiveThirtyEight)

El Elo plano ignora *cómo* se ganó. La variante propuesta por [Silver (2015, FiveThirtyEight NFL)](https://example.com) para deportes de puntos corridos incorpora el **margen de victoria** $M = \text{puntos}_{\text{ganador}} - \text{puntos}_{\text{perdedor}}$ mediante un multiplicador logarítmico:

$$R' = R + K \cdot \ln(1 + M) \cdot (s - E[s])$$

El $\ln(1 + M)$ crece con el margen pero con **rendimientos decrecientes** (ganar por 20 actualiza el rating solo ~1.3× más que ganar por 4). Esto refleja que en fútbol/voleibol/baloncesto una victoria amplia es más informativa, pero no 5× más.

En `Predictor(2)`, la implementación en `src/data/rolling_features.py` usa:
- $K = 28$ (constante canónica del proyecto, pineada en `AGENTS.md`)
- $\text{HOME\_ADV} = 60$ puntos Elo sumados al local antes de calcular $E[s]$
- Margen: $\text{margin} = |\text{pts}_h - \text{pts}_a|$ actualizado tras cada partido real en el backtest, o simulado en el SeasonSimulator

Evaluado con protocolo rolling-origin sobre 2025/26: **AUC 0.762** (n=314 partidos), Brier 0.193, log-loss 0.568 (ver `memoria/prediccion_temporadas.md` §6 y `mejora_precision_2026-07.md` §5).

### 2.3. Probabilidad de victoria esperada vs. probabilidad calibrada

La $E[s]$ del Elo es una **probabilidad calibrada** en el sentido frequentista: para todos los partidos con $E[s] \approx 0.7$, el 70% los gana el local a largo plazo. Esto la diferencia de las salidas brutas de los modelos de ML (LogReg, XGBoost) que, sin calibración post-hoc, dan probabilidades sesgadas o mal calibradas (ver §6).

---

## 3. Aprendizaje supervisado para predicción deportiva

### 3.1. Regresión logística con regularización L2

La **regresión logística** modela $P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ con $\sigma(z) = (1 + e^{-z})^{-1}$. La regularización L2 (ridge) penaliza $\|\mathbf{w}\|_2^2$, dando el problema de optimización:

$$\min_{\mathbf{w}, b} \; -\sum_i \log \sigma\bigl(y_i (\mathbf{w}^\top \mathbf{x}_i + b)\bigr) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

**Por qué se eligió para `SetPredictor` v2**: en el régimen de **datos pequeños** (~1200 sets tras B0b), los modelos lineales regularizados ganan a los ensembles de árboles (ver `mejora_precision_2026-07.md` §6: "logloss CV 0.654 → 0.634" al pasar de ExtraTrees a LogReg+recencia). El `SetPredictor` v2 en producción es una `LogisticRegression(C=0.5)` (inversa de $\lambda$) con **pesos de recencia** half-life=2 temporadas.

### 3.2. Árboles de decisión y ensembles

Un **árbol de decisión** particiona recursivamente el espacio de features $\mathcal{X}$ por ejes paralelos, eligiendo en cada nodo la división que maximiza la reducción de impureza (Gini o entropía). Es interpretable pero muy inestable: pequeños cambios en los datos producen árboles muy diferentes.

Los **ensembles** mitigan esto promediando muchos árboles:
- **RandomForest** ([Breiman, 2001](https://example.com)): bagging de árboles con subconjuntos aleatorios de features en cada split.
- **ExtraTrees** ([Geurts et al., 2006](https://example.com)): igual pero con umbrales aleatorios, no最优. Más rápido, a veces más robusto en datasets pequeños.

### 3.3. Gradient boosting

El **gradient boosting** ([Friedman, 2001](https://example.com)) construye el ensemble secuencialmente: cada árbol nuevo ajusta los **residuos** del ensemble actual según el gradiente de la loss. Implementaciones eficientes:
- **XGBoost** ([Chen & Guestrin, 2016](https://example.com)): regularización L1+L2, shrinkage, column subsampling.
- **LightGBM** ([Ke et al., 2017](https://example.com)): histograma + leaf-wise growth, mucho más rápido en datasets grandes.

`Predictor(2)` usa ambos como candidatos en el benchmark y en la calibración de MatchPredictor v1 (ver `memoria/benchmark.md`). En el `SetPredictor` v2 en producción **no se usan** porque el régimen de datos no los favorece.

### 3.4. Pesos de recencia temporal

En series temporales deportivas, los partidos de la temporada actual son **más informativos** que los de hace 5 años. Una **función de peso exponencial** half-life=2 temporadas asigna a la temporada $t$ un peso:

$$w(t_{\text{current}} - t) = 2^{-(t_{\text{current}} - t) / 2}$$

de modo que la temporada 2023 pesa 0.5 cuando estamos en 2025, y la temporada 2018 pesa ~0.03. El `SetPredictor` v2 entrena con sample weights así sobre `train 2022-2024` y obtiene un CV AUC 0.679 ± 0.017 más alto y estable que el de 4-fold sin pesos (0.631 ± 0.078).

---

## 4. Validación temporal y prevención de leakage

### 4.1. El problema del leakage temporal

El **leakage temporal** ocurre cuando el modelo ve, durante el entrenamiento, información que en producción no estaría disponible. Ejemplos típicos:
- Usar el **ranking final de la temporada** como feature para predecir partidos de esa temporada.
- Calcular `win_rate_last5` sobre los últimos 5 partidos *de la temporada completa* (incluyendo partidos futuros desde el punto de vista del partido a predecir).
- Entrenar con datos hasta 2025 y validar con datos hasta 2024 (el modelo "vio el futuro").

En el TFG se documenta un caso real (`mejora_precision_2026-07.md` §1): el `MatchPredictor` v1 reportaba **AUC 0.707** sobre test 2024, pero el valor honesto con rolling-origin era **0.53**. La diferencia provenía de features que incluían estadísticas de temporada completa. Reconstruir las features sin leakage (Elo rolling, EWMA pre-partido, H2H histórico) bajó el AUC reportado y subió el AUC honesto a 0.762 — el modelo real era mucho mejor de lo que la métrica inflada sugería.

### 4.2. Protocolo rolling-origin

El **protocolo rolling-origin** ([Tashman, 2000](https://example.com); [Hyndman & Athanasopoulos, 2018](https://example.com)) es el estándar para evaluar forecasts en series temporales:

1. Fijar un horizonte de test (en `Predictor(2)`, la temporada 2024/25).
2. Para cada fold $k$:
   - Entrenar con todas las temporadas $\leq t_k$.
   - Predecir la temporada $t_k + 1$ completa.
3. Reportar la métrica agregada sobre todos los folds.

En la práctica del TFG se usa un **rolling-origin de 2 folds** sobre sets:
- Fold 1: `train 2016-2022 → val 2023`
- Fold 2: `train 2016-2023 → val 2024`
- (Test held-out: 2025, **no se itera sobre él**).

La varianza entre folds es la **estimación honesta de incertidumbre** sobre el rendimiento fuera de la muestra.

### 4.3. Features rolling pre-partido

Para evitar el leakage, todas las features del pipeline de producción son **rolling pre-partido**: para un partido de la jornada $j$ de la temporada $t$, se usan solo partidos de la temporada $t$ con jornada $< j$ (o de temporadas $< t$ si $j$ es muy temprana). El módulo `src/data/rolling_features.py` implementa:

- `elo_diff`: rating Elo recalculado partido a partido desde el inicio del histórico
- `diff_win_rate`: win rate acumulado hasta la jornada $j-1$
- `diff_set_ratio`: igual para sets
- `diff_form_ewma`: forma con **exponential weighted moving average** con $\alpha$ controlado

Esto es lo que distingue las features de producción de las del `MatchPredictor` v1 (con leakage): las primeras se computan como si el partido no existiera todavía.

---

## 5. Métricas de evaluación

### 5.1. Brier score

El **Brier score** ([Brier, 1950](https://example.com)) mide el error cuadrático medio entre la probabilidad predicha y el resultado binario observado:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - s_i)^2$$

donde $p_i \in [0, 1]$ es la probabilidad predicha de "local gana" y $s_i \in \{0, 1\}$ es el resultado. **Rango**: $[0, 1]$, **menor es mejor**. Un clasificador perfecto tiene Brier 0; uno constante en 0.5 tiene Brier 0.25.

El Brier es **estrictamente proper**: se minimiza prediciendo la verdadera probabilidad. Es la métrica preferida del TFG para evaluar la calidad de probabilidad del simulador, por encima de accuracy (que es invariante a la calibración).

### 5.2. Log-loss (cross-entropy binaria)

El **log-loss** mide la sorpresa logarítmica media:

$$\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \bigl[ s_i \log p_i + (1 - s_i) \log (1 - p_i) \bigr]$$

**Rango**: $[0, +\infty)$, **menor es mejor**. Más sensible que el Brier a predicciones muy seguras pero incorrectas: un clasificador que da $p = 0.99$ y falla paga $-log(0.01) \approx 4.6$, mientras que con Brier paga solo $0.98^2 = 0.96$.

### 5.3. Expected Calibration Error (ECE)

El **ECE** ([Naeini et al., 2015](https://example.com)) mide la **calibración** independientemente de la discriminación. Se discretiza el rango $[0, 1]$ en $K$ bins, se calcula la confianza media $\bar{p}_k$ y la frecuencia observada $\bar{s}_k$ en cada bin:

$$\text{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{n} \bigl| \bar{p}_k - \bar{s}_k \bigr|$$

**Rango**: $[0, 1]$, **menor es mejor**. Un modelo perfectamente calibrado tiene ECE 0. En el TFG, el simulador pre-B3 tenía ECE 0.242 (sobreconfiado) y post-B3 bajó a 0.057, muy cerca del Elo puro (0.045).

### 5.4. AUC-ROC

El **AUC** ([Fawcett, 2006](https://example.com)) es el área bajo la curva ROC (Receiver Operating Characteristic), que grafica TPR vs. FPR al variar el umbral de decisión. Equivale a la **probabilidad de que el modelo asigne mayor score a un positivo aleatorio que a un negativo aleatorio**. **Rango**: $[0.5, 1]$, **mayor es mejor**, con 0.5 = azar y 1.0 = perfecto.

AUC es **invariante a la calibración** (solo mide ranking) y **robusto a clases desbalanceadas**. En el TFG: MATCH 0.762, SET 0.697, ambos con test held-out 2025/26.

### 5.5. Accuracy

La **accuracy** es la fracción de predicciones correctas con un umbral (típicamente 0.5):

$$\text{Acc} = \frac{1}{n} \sum_i \mathbf{1}[s_i = \mathbb{1}[p_i \geq 0.5]]$$

**Rango**: $[0, 1]$, **mayor es mejor**. Fácil de interpretar pero **engañosa en desbalance de clases** y **sensible al umbral**. En el TFG se reporta siempre junto con Brier y ECE, no sola.

### 5.6. Distancia L1 entre distribuciones de marcador

Para la distribución de marcadores (3-0 / 3-1 / 3-2) entre lo simulado y lo real, la métrica natural es la **distancia L1** (variación total):

$$L_1 = \frac{1}{2} \sum_{m \in \mathcal{M}} \bigl| P_{\text{sim}}(m) - P_{\text{real}}(m) \bigr|$$

El factor 1/2 garantiza $L_1 \in [0, 1]$. En el TFG, pre-B3 $L_1 = 0.286$ (53%/30%/17% simulado vs 39%/35%/26% real) y post-B3 $L_1 = 0.031$ (37.6%/34.7%/27.7% vs 38.7%/35.1%/26.1%).

---

## 6. Calibración de probabilidades

### 6.1. Platt scaling

El **Platt scaling** ([Platt, 1999](https://example.com)) aprende una regresión logística de 1 parámetro sobre los scores del modelo:

$$P(y = 1 \mid s) = \sigma(a \cdot s + b)$$

donde $s$ es el score bruto (puede ser un margen o un logit) y $(a, b)$ se ajustan sobre un conjunto de validación. Es barato, estable, y suele mejorar la calibración sin perder mucha discriminación. Asume que la distorsión es **sigmoide**.

### 6.2. Isotonic regression

La **isotonic regression** ([Zadrozny & Elkan, 2002](https://example.com)) aprende una función monótona no paramétrica sobre los scores. Es más flexible que Platt (no asume forma) pero necesita más datos (≥1000 muestras) y puede sobreajustar con datasets pequeños.

En `Predictor(2)`, el `SetPredictor` legacy ExtraTrees aplicaba `CalibratedClassifierCV(cv=3, method="isotonic")` tras seleccionar el campeón por AUC. En el v2 (LogReg) **no se aplica** porque la regresión logística ya produce probabilidades calibradas por construcción (ver §6.3).

### 6.3. Por qué LogReg no necesita calibración post-hoc

La salida de LogReg es $\sigma(\mathbf{w}^\top \mathbf{x} + b)$, que **minimiza la log-loss** sobre los datos de entrenamiento. Si el modelo está bien especificado (forma funcional correcta), esto converge a la verdadera $P(y = 1 \mid \mathbf{x})$. En la práctica, las predicciones LogReg suelen estar bien calibradas "out of the box", lo cual es una ventaja sobre los ensembles de árboles (cuyas salidas son votes o promedios y necesitan calibración post-hoc).

En el TFG, esto se manifiesta en que el `SetPredictor` v2 LogReg+recencia se entrena sin `CalibratedClassifierCV` y obtiene ECE 0.057 en el backtest — comparable al del Elo puro (0.045), que es la referencia de calibración.

---

## 7. Simulación Monte Carlo

### 7.1. Estimación de incertidumbre por repetición

La **simulación Monte Carlo** ([Metropolis & Ulam, 1949](https://example.com)) estima expectativas de funcionales de procesos estocásticos repitiendo el experimento $N$ veces con semillas distintas y promediando:

$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(\omega_i), \quad \omega_i \sim P$$

En `Predictor(2)`, el `MatchSimulator.monte_carlo_simulate()` corre $N$ simulaciones independientes de un partido y devuelve la **distribución empírica** de marcadores y la probabilidad de cada ganador. El **error estándar** del estimador es $\sigma / \sqrt{N}$, donde $\sigma$ es la desviación estándar de $f(\omega)$ entre corridas.

### 7.2. Distribución de marcadores vs. resultado puntual

Una sola simulación da un **marcador puntual** (ej. 3-1). $N$ simulaciones dan una **distribución** (ej. 3-0 en 38%, 3-1 en 35%, 3-2 en 27%). La distribución es la información **útil** para predicción (probabilidades de cada resultado) y para toma de decisiones (qué apuesta ofrece valor).

En la UI, el endpoint `POST /api/simular/partido` con `n_simulaciones_mc > 0` devuelve la distribución agregada; con `n_simulaciones_mc = 0` devuelve un solo partido. La elección de $N$ es un trade-off **tiempo/precisión**: $N = 2000$ basta para tener errores estándar < 1pp en las probabilidades de marcador (test empírico en `simulator.md` §6).

---

## Referencias cruzadas a `memoria/`

| Sección de este md | Aplicación en el proyecto | Fuente original |
|---|---|---|
| §1 (Markov) | `MatchSimulator._simulate_set` en `src/simulation/simulator.py` | `simulator.md` §4 |
| §1.3 (forma cerrada) | `p_set_from_p_point` en `src/simulation/set_math.py` (A2) | `mejora_precision_2026-07.md` §7 |
| §2 (Elo) | `get_historical_team_elo` y `_compute_margin_elo` en `src/data/rolling_features.py` | `match_predictor.md` y `prediccion_temporadas.md` §6 |
| §3 (ML) | `SetPredictor` y `MatchPredictor` en `src/models/` | `set_predictor.md`, `match_predictor.md`, `benchmark.md` |
| §4 (validación) | `evaluate_model_rolling` en `src/models/evaluation.py` (Fase 0) | `mejora_precision_2026-07.md` §1, §5 |
| §5 (métricas) | `src/models/measure_precision.py`, `backtest_simulator.py` | `INDICE.md` "Métricas clave" |
| §6 (calibración) | `SetPredictor` legacy con isotonic; v2 sin calibración | `set_predictor.md` §3 |
| §7 (Monte Carlo) | `MatchSimulator.monte_carlo_simulate` en `src/simulation/simulator.py` | `simulator.md` §6, `prediccion_partidos.md` §6 |

---

**Cómo usar este documento**: el Cap 2 de la memoria LaTeX se redacta
directamente desde aquí, ajustando el tono y la extensión a la rúbrica UPV
(~25-35 páginas de cuerpo). Las fórmulas y definiciones ya están en su forma
canónica; la prosa que las rodea se condensa o expande según necesidad.
