# SuperLega Simulator

Sistema de predicción y simulación de partidos y temporadas de la **SuperLega**
italiana de voleibol masculino, construido sobre un histórico de 1 322 partidos
correspondientes a diez temporadas (2016/2017 – 2025/2026).

El sistema combina tres componentes:

1. **Rating Elo con margen** — resume la fuerza relativa de cada equipo y se
   actualiza secuencialmente tras cada encuentro, sin fase de entrenamiento.
   Es la señal de partido en producción.
2. **Regresión Ridge de seis variables** — traduce la diferencia de fuerza en
   la probabilidad de ganar un rally individual.
3. **Simulador de cadena de Markov** — mediante repetición de Monte Carlo,
   reproduce el partido punto a punto y devuelve tanto la probabilidad de
   victoria como la distribución de marcadores posibles.

Todas las variables se construyen exclusivamente con información anterior al
encuentro que se predice, y la evaluación sigue un esquema de **origen móvil**
(*rolling origin*) que impide utilizar datos del futuro.

## Resultados

Sobre la temporada 2025/2026, reservada como conjunto de prueba:

| Objetivo | n     | Log-loss | AUC    | Brier  | Acierto |
| -------- | ----- | -------- | ------ | ------ | ------- |
| Partido  | 314   | 0,5677   | 0,7624 | 0,1930 | 0,7038  |
| Set      | 1 193 | 0,6218   | 0,6970 | 0,2164 | 0,6505  |

El simulador completo reproduce además la distribución real de marcadores con
una distancia L1 de 0,0315 en el backtest de 2024/2025.

Las cifras se regeneran con `python -m src.models.precision_report` y viven en
`models/precision_improved.json`, que es la fuente que consume la API.

## Requisitos

- Python ≥ 3.10
- Node.js ≥ 18 (para la interfaz web)

## Puesta en marcha

```bash
pip install -e ".[test]"
```

Los modelos entrenados ya están versionados en `models/`, así que la API
arranca sin necesidad de reentrenar:

```bash
python -m src.api.main
```

La API queda en `http://localhost:8000`. Para la interfaz web, en otra terminal:

```bash
cd src/web
npm install
npm run dev
```

Vite sirve en `http://localhost:5173` y hace proxy de `/api` hacia el backend.

Para desplegar como una sola aplicación, `npm run build` genera `src/web/dist/`,
que la API monta automáticamente en `/` al arrancar.

## Reentrenamiento

El pipeline completo, partiendo de los CSV de `DB/`:

```bash
python -m src.data.data_pipeline      # carga y normalización
python -m src.data.feature_store      # construcción de variables
python -m src.models.train            # entrenamiento y persistencia
```

## Tests

```bash
pytest -q -m "not slow"    # rápidos
pytest -q                  # incluye los que requieren los modelos entrenados
ruff check src/ tests/ && black --check src/ tests/
```

## Estructura

```
DB/          Datos originales en CSV/XLSX (resultados, sets, estadísticas)
models/      Artefactos entrenados y métricas de evaluación
memoria/     Documentación técnica de cada componente
src/data/    Carga, normalización y construcción de variables
src/models/  Entrenamiento, evaluación y experimentos
src/simulation/  Cadena de Markov, simulador de partido y de temporada
src/api/     Servicio REST en FastAPI
src/web/     Interfaz en React + Vite
tests/       Suite de pruebas
```

## API

| Método | Ruta                              | Descripción                              |
| ------ | --------------------------------- | ---------------------------------------- |
| GET    | `/api/equipos`                    | Equipos disponibles con su fuerza         |
| GET    | `/api/equipos/{nombre}`           | Detalle y plantilla de un equipo          |
| POST   | `/api/simular/partido`            | Partido individual o Monte Carlo          |
| POST   | `/api/simular/temporada`          | Temporada completa                        |
| POST   | `/api/simular/temporada/iniciar`  | Inicializa una temporada jornada a jornada |
| POST   | `/api/simular/temporada/jornada`  | Simula una jornada concreta               |
| GET    | `/api/modelo/info`                | Componentes del sistema y sus métricas    |

## Fuente de los datos

Portal oficial de la Lega Pallavolo Serie A. Los ficheros de `DB/` recogen
calendarios, resultados, marcadores parciales de cada set y estadísticas
agregadas por equipo y por jugador, normalizados a 22 nombres canónicos de club.
