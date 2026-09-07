# Cross-Temporal Robustness Check

> **⚠️ SNAPSHOT (regenerado el 2026-07-23).** Este documento es una **foto
> generada por el script** `src/models/cross_temporal_check.py` (commit
> `b4b4498`). La **única fuente de verdad** de la tabla es el script. Tras
> cualquier cambio de datos o modelo, regenerar con
> `python -m src.models.cross_temporal_check --save` (escribe en
> `models/report/cross_temporal.md`). Para re-fotocopiar este `.md` desde la
> fuente:
>
> ```bash
> python -m src.models.cross_temporal_check --save
> cp models/report/cross_temporal.md memoria/cross_temporal.md
> ```
>
> El script es de **solo evaluación** (no reentrena nada): lee la señal de Elo
> pre-computada (`elo_win_prob_h`) de `DB/features/match_features.csv` y mide las
> métricas estándar de clasificación binaria por año. La fila 2025 se reemplaza
> por el número canónico de `models/precision_improved.json` para comparación
> directa. Ver [`mejora_precision_2026-07.md` §6](mejora_precision_2026-07.md)
> (margin-Elo rolling sin leakage) para el contexto del modelo evaluado.

Production margin-Elo evaluated on each held-out year. The 2025 row is the
canonical number from `models/precision_improved.json`; the other rows are
computed by `python -m src.models.cross_temporal_check` on the same production
Elo signal from `DB/features/match_features.csv`.

| Year | n | logloss | AUC | Brier | accuracy | source |
|---:|---:|---:|---:|---:|---:|---|
| 2022 | 118 | 0.6313 | 0.6694 | 0.2217 | 0.6356 | cross_temporal_check |
| 2023 | 162 | 0.6629 | 0.6246 | 0.2345 | 0.5802 | cross_temporal_check |
| 2024 | 222 | 0.5765 | 0.7561 | 0.1967 | 0.6937 | cross_temporal_check |
| 2025 | 314 | 0.5677 | 0.7624 | 0.1930 | 0.7038 | precision_improved.json |
