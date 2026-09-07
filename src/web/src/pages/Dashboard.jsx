import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Play, Trophy } from 'lucide-react'
import { getEquipos, getModeloInfo } from '../api.js'
import { fmtInt, fmtMetric, fmtPct, fmtRatio, teamMarkColor } from '../lib/format.js'

const FILTROS = [
  { id: 'actual', label: 'Actuales' },
  { id: 'historico', label: 'Históricos' },
  { id: 'todos', label: 'Todos' },
]

/** Color de la barra de fuerza: escala lima única, nunca el color del club. */
function strengthColor(fuerza) {
  if (fuerza >= 0.6) return 'var(--lime)'
  if (fuerza >= 0.45) return 'var(--lime-dim)'
  return 'var(--lime-dimmer)'
}

function Componente({ n, titulo, detalle }) {
  return (
    <div style={{ display: 'flex', gap: 'var(--s3)', alignItems: 'flex-start' }}>
      <span className="mono" style={{ fontSize: 11, color: 'var(--lime)', paddingTop: 4 }}>
        {n}
      </span>
      <div>
        <div
          className="cond"
          style={{
            fontSize: 24,
            fontWeight: 800,
            lineHeight: 1.05,
            color: 'var(--text-strong)',
          }}
        >
          {titulo}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 2 }}>{detalle}</div>
      </div>
    </div>
  )
}

function Metrica({ valor, etiqueta, destacado }) {
  return (
    <div>
      <div
        className="mono"
        style={{ fontSize: 20, fontWeight: 600, color: destacado ? 'var(--lime)' : 'var(--text)' }}
      >
        {valor}
      </div>
      <div
        style={{
          fontSize: 10,
          color: 'var(--text-faint)',
          letterSpacing: '0.07em',
          textTransform: 'uppercase',
          marginTop: 2,
        }}
      >
        {etiqueta}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [equipos, setEquipos] = useState([])
  const [modelo, setModelo] = useState(null)
  const [filtro, setFiltro] = useState('actual')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let vivo = true
    Promise.all([getEquipos().catch(() => null), getModeloInfo().catch(() => null)])
      .then(([eqData, modeloData]) => {
        if (!vivo) return
        setEquipos(eqData?.equipos || [])
        setModelo(modeloData)
      })
      .finally(() => vivo && setLoading(false))
    return () => {
      vivo = false
    }
  }, [])

  const actuales = equipos.filter((e) => e.categoria === 'actual')
  const historicos = equipos.filter((e) => e.categoria === 'historico')
  const visibles = (filtro === 'todos' ? equipos : filtro === 'actual' ? actuales : historicos)
    .slice()
    .sort((a, b) => (b.fuerza ?? 0) - (a.fuerza ?? 0))

  const senal = modelo?.senal_partido
  const test = senal?.test
  const datos = modelo?.datos

  return (
    <>
      {/* ── Hero ── */}
      <section className="page-bleed" style={{ borderBottom: '1px solid var(--line)' }}>
        <div className="hero">
          <div className="fade-in">
            <div className="kicker" style={{ color: 'var(--lime)', marginBottom: 'var(--s3)' }}>
              Simulador Monte Carlo
            </div>
            <h1 className="hero-title">
              SuperLega
              <br />
              <span style={{ color: 'var(--lime)' }}>Simulator</span>
            </h1>

            <div className="btn-group" style={{ marginTop: 'var(--s6)', marginBottom: 'var(--s6)' }}>
              <Link to="/simular-partido" className="btn btn-primary btn-lg">
                <Play size={17} fill="currentColor" aria-hidden="true" />
                Simular partido
              </Link>
              <Link to="/simular-temporada" className="btn btn-lg">
                <Trophy size={17} aria-hidden="true" />
                Temporada completa
              </Link>
            </div>

            {datos?.partidos && (
              <div className="hero-stats">
                <div className="kicker">
                  Histórico analizado{datos.rango ? ` · ${datos.rango}` : ''}
                </div>
                <div className="hero-stats-row">
                  {[
                    { v: fmtInt(datos.partidos), l: 'Partidos' },
                    { v: fmtInt(datos.sets), l: 'Sets' },
                    { v: datos.equipos_canonicos, l: 'Equipos' },
                    { v: datos.temporadas, l: 'Temporadas' },
                  ].map((s) => (
                    <div key={s.l}>
                      <div className="hero-stat-value">{s.v}</div>
                      <div className="kicker" style={{ marginTop: 4 }}>
                        {s.l}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Tarjeta del motor: los tres componentes reales del sistema. */}
          <aside className="panel panel-accent panel-pad fade-in" aria-label="Motor del simulador">
            <div className="kicker" style={{ marginBottom: 'var(--s4)' }}>
              El motor
            </div>

            {senal ? (
              <>
                <div className="stack" style={{ gap: 'var(--s4)' }}>
                  <Componente
                    n="01"
                    titulo={senal.modelo}
                    detalle={`Señal de partido · ${senal.n_variables} variable${
                      senal.n_variables === 1 ? '' : 's'
                    }, sin entrenamiento`}
                  />
                  <Componente
                    n="02"
                    titulo={modelo?.modelo_punto?.modelo?.split(' ')[0] || 'Ridge'}
                    detalle="Probabilidad de ganar un rally"
                  />
                  <Componente
                    n="03"
                    titulo="Markov + Monte Carlo"
                    detalle="Simulación punto a punto del encuentro"
                  />
                </div>

                {test && (
                  <>
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
                        gap: 'var(--s3)',
                        marginTop: 'var(--s5)',
                        paddingTop: 'var(--s4)',
                        borderTop: '1px solid #232329',
                      }}
                    >
                      <Metrica valor={fmtMetric(test.auc)} etiqueta="AUC" destacado />
                      <Metrica valor={fmtMetric(test.brier)} etiqueta="Brier" />
                      <Metrica valor={fmtPct(test.acierto)} etiqueta="Acierto" />
                    </div>
                    <div
                      className="mono"
                      style={{
                        fontSize: 10.5,
                        color: 'var(--text-ghost)',
                        marginTop: 'var(--s3)',
                        lineHeight: 1.5,
                      }}
                    >
                      Test reservado {test.temporada_test} · {test.n} partidos
                    </div>
                  </>
                )}
              </>
            ) : (
              <div style={{ color: 'var(--text-faint)', fontSize: 13 }}>
                {loading ? 'Cargando información del modelo…' : 'Información del modelo no disponible'}
              </div>
            )}
          </aside>
        </div>
      </section>

      {/* ── Equipos ── */}
      <section className="page" style={{ paddingTop: 'var(--s8)' }}>
        <div className="section-head">
          <h2 className="section-title">Los equipos</h2>
          <div className="segmented" role="group" aria-label="Filtrar equipos">
            {FILTROS.map((f) => {
              const n = f.id === 'actual' ? actuales.length : f.id === 'historico' ? historicos.length : equipos.length
              return (
                <button
                  key={f.id}
                  type="button"
                  aria-pressed={filtro === f.id}
                  onClick={() => setFiltro(f.id)}
                >
                  {f.label} {n > 0 ? n : ''}
                </button>
              )
            })}
          </div>
        </div>

        {loading ? (
          <div className="loading">
            <span className="spinner" aria-hidden="true" />
            Cargando equipos…
          </div>
        ) : visibles.length === 0 ? (
          <div className="empty">No hay equipos en esta categoría</div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
              gap: 'var(--s2) var(--s5)',
            }}
          >
            {visibles.map((eq, i) => (
              <Link
                key={eq.nombre}
                to={`/equipo/${encodeURIComponent(eq.nombre)}`}
                className="team-band fade-in"
                style={{
                  borderLeftColor: teamMarkColor(eq.colores),
                  animationDelay: `${Math.min(i, 12) * 0.02}s`,
                }}
              >
                <span className="team-band-rank" aria-hidden="true">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <span
                  className="cond"
                  style={{
                    fontSize: 22,
                    fontWeight: 700,
                    letterSpacing: '0.02em',
                    textTransform: 'uppercase',
                    color: 'var(--text-strong)',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {eq.nombre}
                </span>
                <span className="strength-track" aria-hidden="true">
                  <span
                    className="strength-fill"
                    style={{
                      width: `${Math.max((eq.fuerza ?? 0) * 100, 2)}%`,
                      background: strengthColor(eq.fuerza ?? 0),
                    }}
                  />
                </span>
                <span className="mono num" style={{ fontSize: 13 }}>
                  {fmtRatio(eq.fuerza)}
                  <span className="sr-only"> de fuerza</span>
                </span>
              </Link>
            ))}
          </div>
        )}

        <p className="note mt-6">
          La barra de color a la izquierda identifica al club; la fuerza se lee siempre en la misma
          escala lima, así que ningún color de equipo compite con el dato. La fuerza es el rating
          Elo con margen al cierre de la temporada, reescalado a [0,1].
        </p>
      </section>
    </>
  )
}
