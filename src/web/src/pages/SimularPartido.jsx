import { useState, useEffect, useRef, useMemo } from 'react'
import { Play, Loader2 } from 'lucide-react'
import { getEquipos, getModeloInfo, simularPartido } from '../api.js'
import {
  ensureDistinct,
  fmtInt,
  fmtPct,
  readableOn,
  teamColors,
  teamMarkColor,
} from '../lib/format.js'
import PlayerTable from '../components/PlayerTable.jsx'

// El backend recorta la petición a MAX_MC_ITERATIONS, así que las opciones se
// derivan de ese tope: ofrecer 10 000 cuando el servidor sólo ejecuta 5 000
// hacía que el rótulo del control y el de los resultados no coincidieran.
const MC_BASE = 2000
const MC_TOPE_POR_DEFECTO = 5000

function opcionesMC(tope) {
  const max = Number.isFinite(tope) && tope > 0 ? tope : MC_TOPE_POR_DEFECTO
  return [...new Set([Math.min(MC_BASE, max), max])].sort((a, b) => a - b)
}

/** Acumula las stats por jugador a lo largo de todos los sets. */
function acumular(sets, lado) {
  const porJugador = {}
  for (const s of sets || []) {
    const stats = lado === 'local' ? s.stats_local : s.stats_visitante
    for (const p of stats || []) {
      const nombre = p.jugador || 'Desconocido'
      if (!porJugador[nombre]) {
        porJugador[nombre] = {
          jugador: nombre,
          puntos: 0,
          aces: 0,
          ataques_ganados: 0,
          bloqueos: 0,
          recepciones_exc: 0,
        }
      }
      const acc = porJugador[nombre]
      acc.puntos += p.puntos || 0
      acc.aces += p.aces || 0
      acc.ataques_ganados += p.ataques_ganados || 0
      acc.bloqueos += p.bloqueos || 0
      acc.recepciones_exc += p.recepciones_exc || 0
    }
  }
  return Object.values(porJugador).sort((a, b) => b.puntos - a.puntos)
}

function SelectorEquipo({ id, label, valor, onChange, equipos }) {
  const color = teamMarkColor(teamColors(equipos, valor))
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <div className="team-select" style={{ borderLeftColor: color }}>
        <select id={id} value={valor} onChange={(e) => onChange(e.target.value)}>
          {equipos.map((eq) => (
            <option key={eq.nombre} value={eq.nombre}>
              {eq.nombre}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

export default function SimularPartido() {
  const [equipos, setEquipos] = useState([])
  const [local, setLocal] = useState('')
  const [visitante, setVisitante] = useState('')
  const [topeMC, setTopeMC] = useState(MC_TOPE_POR_DEFECTO)
  const [nSims, setNSims] = useState(MC_BASE)
  const [resultado, setResultado] = useState(null)
  const [probabilidad, setProbabilidad] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const resultRef = useRef(null)

  useEffect(() => {
    getEquipos()
      .then((data) => {
        const eqs = data.equipos || []
        setEquipos(eqs)
        if (eqs.length >= 2) {
          setLocal(eqs[0].nombre)
          setVisitante(eqs[1].nombre)
        }
      })
      .catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    getModeloInfo()
      .then((info) => {
        const tope = info?.simulador?.max_simulaciones
        if (Number.isFinite(tope) && tope > 0) {
          setTopeMC(tope)
          setNSims((actual) => Math.min(actual, tope))
        }
      })
      .catch(() => {
        /* si no se puede leer el tope, se mantiene el valor por defecto */
      })
  }, [])

  const opciones = useMemo(() => opcionesMC(topeMC), [topeMC])
  const mismoEquipo = Boolean(local) && local === visitante

  const simular = async () => {
    if (!local || !visitante || mismoEquipo) return
    setLoading(true)
    setError(null)
    setResultado(null)
    setProbabilidad(null)

    try {
      // Las dos llamadas van juntas: el marcador sin la probabilidad previa
      // no dice nada, y la probabilidad sin un partido concreto se queda en
      // abstracto. Antes eran dos modos excluyentes.
      const [mc, partido] = await Promise.all([
        simularPartido({ local, visitante, n_simulaciones_mc: nSims, generar_puntos: false }),
        simularPartido({
          local,
          visitante,
          generar_puntos: true,
          generar_stats_jugadores: true,
        }),
      ])
      setProbabilidad(mc)
      setResultado(partido)
      window.setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        60,
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const colorLocal = teamMarkColor(teamColors(equipos, resultado?.local || local))
  // Si el visitante lleva un color confundible con el del local, se oscurece:
  // los dos campos del marcador tienen que leerse como equipos distintos.
  const colorVisitante = ensureDistinct(
    colorLocal,
    teamMarkColor(teamColors(equipos, resultado?.visitante || visitante)),
  )

  const acumLocal = useMemo(() => acumular(resultado?.sets, 'local'), [resultado])
  const acumVisitante = useMemo(() => acumular(resultado?.sets, 'visitante'), [resultado])

  const anotadores = useMemo(() => {
    const todos = [
      ...acumLocal.map((p) => ({ ...p, equipo: resultado?.local, barColor: colorLocal })),
      ...acumVisitante.map((p) => ({ ...p, equipo: resultado?.visitante, barColor: colorVisitante })),
    ]
    return todos.sort((a, b) => b.puntos - a.puntos)
  }, [acumLocal, acumVisitante, resultado, colorLocal, colorVisitante])

  const distribucion = useMemo(() => {
    const dist = probabilidad?.distribucion || {}
    const orden = ['3-0', '3-1', '3-2', '2-3', '1-3', '0-3']
    const max = Math.max(...Object.values(dist), 0.0001)
    return orden
      .filter((k) => k in dist)
      .map((k) => ({
        marcador: k,
        pct: dist[k],
        ancho: (dist[k] / max) * 100,
        localGana: Number(k[0]) > Number(k[2]),
      }))
  }, [probabilidad])

  const localGanador = resultado && resultado.ganador === resultado.local

  return (
    <>
      <div className="page-bleed">
        <div className="page-head" style={{ padding: 'var(--s7) var(--gutter) var(--s5)' }}>
          <h1 className="page-title">Simular partido</h1>
          <p className="page-sub">
            El sistema estima primero la probabilidad de victoria por Monte Carlo y después juega un
            encuentro concreto, punto a punto.
          </p>
        </div>
      </div>

      {/* ── Configuración ── */}
      <div
        className="page-bleed"
        style={{ background: 'var(--surface)', borderBottom: '1px solid var(--line)' }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 'var(--s3)',
            alignItems: 'end',
            padding: 'var(--s5) var(--gutter)',
          }}
        >
          <SelectorEquipo
            id="select-local"
            label="Local"
            valor={local}
            onChange={setLocal}
            equipos={equipos}
          />
          <SelectorEquipo
            id="select-visitante"
            label="Visitante"
            valor={visitante}
            onChange={setVisitante}
            equipos={equipos}
          />
          <div>
            <span className="field-label" id="lbl-sims">
              Simulaciones
            </span>
            <div className="segmented" role="group" aria-labelledby="lbl-sims">
              {opciones.map((n) => (
                <button
                  key={n}
                  type="button"
                  aria-pressed={nSims === n}
                  onClick={() => setNSims(n)}
                  className="mono"
                >
                  {fmtInt(n)}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            style={{ minHeight: 46 }}
            onClick={simular}
            disabled={loading || !local || !visitante || mismoEquipo}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="spin-icon" aria-hidden="true" /> Simulando…
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" aria-hidden="true" /> Simular
              </>
            )}
          </button>
        </div>

        {(mismoEquipo || error) && (
          <div style={{ padding: '0 var(--gutter) var(--s5)' }}>
            <p className={`notice ${error ? 'notice-error' : ''}`} role="alert">
              {error || 'Los equipos local y visitante deben ser distintos.'}
            </p>
          </div>
        )}
      </div>

      <div ref={resultRef} aria-live="polite">
        {resultado && probabilidad && (
          <>
            {/* ── Probabilidad previa ── */}
            <div className="page-bleed">
              <div style={{ padding: 'var(--s6) var(--gutter) var(--s4)' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    marginBottom: 'var(--s3)',
                    gap: 'var(--s4)',
                    flexWrap: 'wrap',
                  }}
                >
                  <span className="kicker">
                    Probabilidad previa · {fmtInt(probabilidad.n_simulaciones)}{' '}
                    simulaciones
                  </span>
                </div>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
                    alignItems: 'center',
                    gap: 'var(--s4)',
                  }}
                >
                  <span
                    className="mono"
                    style={{
                      fontSize: 22,
                      fontWeight: 600,
                      color:
                        probabilidad.prob_local >= 0.5 ? 'var(--text-strong)' : 'var(--text-dim)',
                    }}
                  >
                    {fmtPct(probabilidad.prob_local)}
                  </span>
                  <span style={{ display: 'flex', height: 10, overflow: 'hidden' }}>
                    <span
                      style={{ width: `${probabilidad.prob_local * 100}%`, background: 'var(--lime)' }}
                    />
                    <span style={{ width: 1, background: 'var(--bg)' }} />
                    <span
                      style={{
                        width: `${probabilidad.prob_visitante * 100}%`,
                        background: 'var(--text-ghost)',
                      }}
                    />
                  </span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 22,
                      fontWeight: 600,
                      color:
                        probabilidad.prob_visitante >= 0.5
                          ? 'var(--text-strong)'
                          : 'var(--text-dim)',
                    }}
                  >
                    {fmtPct(probabilidad.prob_visitante)}
                  </span>
                </div>
              </div>
            </div>

            {/* ── Marcador ── */}
            <div className="page-bleed">
              <div className="scoreboard fade-in">
                <div
                  className="scoreboard-side home"
                  style={{ background: colorLocal, color: readableOn(colorLocal) }}
                >
                  <div>
                    <div
                      className="kicker"
                      style={{ color: 'currentColor', opacity: 0.7, marginBottom: 6 }}
                    >
                      Local · {fmtPct(probabilidad.prob_local)} previo
                    </div>
                    <div className="scoreboard-team">{resultado.local}</div>
                  </div>
                </div>

                <div className="scoreboard-center">
                  <div className="scoreboard-sets">
                    <span className={localGanador ? 'score-win' : 'score-lose'}>
                      {resultado.sets_local}
                    </span>
                    <span className="score-sep">–</span>
                    <span className={!localGanador ? 'score-win' : 'score-lose'}>
                      {resultado.sets_visitante}
                    </span>
                  </div>
                  <div className="kicker" style={{ position: 'relative' }}>
                    Final
                  </div>
                  <p className="sr-only">
                    {resultado.ganador} gana por {resultado.sets_local} a{' '}
                    {resultado.sets_visitante}
                  </p>
                </div>

                <div
                  className="scoreboard-side away"
                  style={{ background: colorVisitante, color: readableOn(colorVisitante) }}
                >
                  <div>
                    <div
                      className="kicker"
                      style={{ color: 'currentColor', opacity: 0.7, marginBottom: 6 }}
                    >
                      Visitante · {fmtPct(probabilidad.prob_visitante)} previo
                    </div>
                    <div className="scoreboard-team">{resultado.visitante}</div>
                  </div>
                </div>
              </div>

              {/* ── Sets ── */}
              <div className="set-strip">
                {resultado.sets.map((s) => {
                  const gano = s.ganador === resultado.local
                  return (
                    <div key={s.numero} className={`set-cell ${gano ? 'won' : 'lost'}`}>
                      <div className="kicker">Set {s.numero}</div>
                      <div className="set-score">
                        <span style={{ color: gano ? 'var(--lime)' : 'var(--text-faint)' }}>
                          {s.puntos_local}
                        </span>
                        <span style={{ color: 'var(--line-strong)' }}> – </span>
                        <span style={{ color: !gano ? 'var(--lime)' : 'var(--text-faint)' }}>
                          {s.puntos_visitante}
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* ── Distribución + anotadores ── */}
            <section className="page" style={{ paddingTop: 'var(--s7)' }}>
              <div className="split">
                <div>
                  <h2 className="section-title">Cómo acaba el partido</h2>
                  <p className="page-sub" style={{ marginBottom: 'var(--s5)' }}>
                    Frecuencia de cada marcador en{' '}
                    {fmtInt(probabilidad.n_simulaciones)} simulaciones.
                  </p>

                  <div className="stack" style={{ gap: 'var(--s2)' }}>
                    {distribucion.map((d) => (
                      <div key={d.marcador} className="bar-row">
                        <span
                          className="cond"
                          style={{
                            fontSize: 20,
                            fontWeight: 700,
                            color: d.localGana ? 'var(--lime)' : 'var(--text-dim)',
                          }}
                        >
                          {d.marcador.replace('-', '–')}
                        </span>
                        <span className="bar-track">
                          <span
                            className="bar-fill"
                            style={{
                              width: `${d.ancho}%`,
                              background: d.localGana ? 'var(--lime-dim)' : 'var(--text-ghost)',
                            }}
                          />
                        </span>
                        <span
                          className="mono num"
                          style={{ fontSize: 12, color: 'var(--text-dim)' }}
                        >
                          {fmtPct(d.pct)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h2 className="section-title">Anotadores</h2>
                  <p className="page-sub" style={{ marginBottom: 'var(--s5)' }}>
                    Acumulado del partido, ambos equipos.
                  </p>
                  <PlayerTable
                    players={anotadores}
                    showBar
                    limit={12}
                    caption="Estadísticas acumuladas de los jugadores del partido"
                    emptyText="Este partido se simuló sin estadísticas de jugador"
                  />
                </div>
              </div>

              {/* ── Detalle por set ── */}
              {resultado.sets.some((s) => s.stats_local?.length) && (
                <section className="mt-7">
                  <h2 className="section-title" style={{ marginBottom: 'var(--s4)' }}>
                    Detalle por set
                  </h2>
                  <div className="stack" style={{ gap: 'var(--s5)' }}>
                    {resultado.sets.map((s) => (
                      <div key={s.numero} className="panel panel-pad">
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'baseline',
                            gap: 'var(--s4)',
                            marginBottom: 'var(--s4)',
                            flexWrap: 'wrap',
                          }}
                        >
                          <span className="kicker">Set {s.numero}</span>
                          <span
                            className="cond"
                            style={{ fontSize: 26, fontWeight: 800, color: 'var(--text-strong)' }}
                          >
                            {s.puntos_local} – {s.puntos_visitante}
                          </span>
                          <span
                            className="cond"
                            style={{
                              fontSize: 18,
                              fontWeight: 700,
                              textTransform: 'uppercase',
                              letterSpacing: '0.06em',
                              color: 'var(--lime)',
                            }}
                          >
                            {s.ganador}
                          </span>
                        </div>
                        <div className="grid-2">
                          {[
                            { equipo: resultado.local, stats: s.stats_local, color: colorLocal },
                            {
                              equipo: resultado.visitante,
                              stats: s.stats_visitante,
                              color: colorVisitante,
                            },
                          ].map(({ equipo, stats, color }) => (
                            <div key={equipo}>
                              <div
                                className="kicker"
                                style={{
                                  marginBottom: 'var(--s2)',
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: 'var(--s2)',
                                }}
                              >
                                <span
                                  aria-hidden="true"
                                  style={{ width: 10, height: 10, background: color }}
                                />
                                {equipo}
                              </div>
                              <PlayerTable
                                players={(stats || [])
                                  .slice()
                                  .sort((a, b) => (b.puntos || 0) - (a.puntos || 0))}
                                limit={7}
                                caption={`Estadísticas de ${equipo} en el set ${s.numero}`}
                                emptyText="Sin datos"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </section>
          </>
        )}
      </div>
    </>
  )
}
