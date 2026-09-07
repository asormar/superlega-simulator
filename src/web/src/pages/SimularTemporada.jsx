import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  Play,
  Pause,
  SkipForward,
  Loader2,
  RotateCcw,
  CheckSquare,
  Square,
  AlertCircle,
} from 'lucide-react'
import { getEquipos, iniciarTemporada, simularJornada } from '../api.js'
import { teamColors, teamMarkColor } from '../lib/format.js'
import PlayerTable from '../components/PlayerTable.jsx'
import { TeamDot } from '../components/TeamMark.jsx'

const MAX_EQUIPOS = 12
const JORNADA_DELAY_MS = 5000
const PLAYOFF_CUT = 8

export default function SimularTemporada() {
  const [equipos, setEquipos] = useState([])
  const [seleccionados, setSeleccionados] = useState([])
  const [dobleVuelta, setDobleVuelta] = useState(true)
  const [semilla, setSemilla] = useState(42)

  const [phase, setPhase] = useState('config') // 'config' | 'running' | 'complete'
  const [schedule, setSchedule] = useState([])
  const [totalJornadas, setTotalJornadas] = useState(0)
  const [currentJornadaIndex, setCurrentJornadaIndex] = useState(-1)
  const [currentStandings, setCurrentStandings] = useState([])
  const [currentPlayerStats, setCurrentPlayerStats] = useState([])
  const [jornadaHistory, setJornadaHistory] = useState([])
  const [selectedJornadaIndex, setSelectedJornadaIndex] = useState(null)
  const userSelectedJornadaRef = useRef(false)
  const [isPaused, setIsPaused] = useState(false)
  const [isLoadingJornada, setIsLoadingJornada] = useState(false)
  const [error, setError] = useState(null)

  const timerRef = useRef(null)

  useEffect(() => {
    getEquipos()
      .then((data) => {
        const eqs = data.equipos || []
        setEquipos(eqs)
        setSeleccionados(eqs.filter((e) => e.categoria === 'actual').map((e) => e.nombre))
      })
      .catch((err) => setError(err.message))
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const equiposActuales = useMemo(
    () => equipos.filter((e) => e.categoria === 'actual'),
    [equipos],
  )
  const equiposHistoricos = useMemo(
    () => equipos.filter((e) => e.categoria === 'historico'),
    [equipos],
  )

  const colorDe = useCallback(
    (nombre) => teamMarkColor(teamColors(equipos, nombre)),
    [equipos],
  )
  const coloresDe = useCallback((nombre) => teamColors(equipos, nombre), [equipos])

  const toggleEquipo = (nombre) => {
    setSeleccionados((prev) => {
      if (prev.includes(nombre)) return prev.filter((n) => n !== nombre)
      if (prev.length >= MAX_EQUIPOS) return prev
      return [...prev, nombre]
    })
  }

  const totalPartidos = seleccionados.length * (seleccionados.length - 1) * (dobleVuelta ? 1 : 0.5)

  // ─────────────────────────────────────────────────────────
  // Simulación jornada a jornada
  // ─────────────────────────────────────────────────────────

  const iniciarSimulacion = async () => {
    if (seleccionados.length < 2) return
    setError(null)
    setIsLoadingJornada(true)
    try {
      const init = await iniciarTemporada({
        equipos: seleccionados,
        doble_vuelta: dobleVuelta,
        semilla,
      })
      setSchedule(init.schedule)
      setTotalJornadas(init.total_jornadas)
      setCurrentStandings(init.initial_standings)
      setCurrentPlayerStats(init.initial_player_stats)
      setJornadaHistory([])
      setSelectedJornadaIndex(null)
      userSelectedJornadaRef.current = false
      setCurrentJornadaIndex(-1)
      setPhase('running')
      setIsPaused(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoadingJornada(false)
    }
  }

  const fetchJornada = useCallback(
    async (jornadaIdx) => {
      if (jornadaIdx < 0 || jornadaIdx >= totalJornadas) return
      setIsLoadingJornada(true)
      setError(null)
      try {
        const resp = await simularJornada({
          equipos: seleccionados,
          doble_vuelta: dobleVuelta,
          schedule,
          jornada_index: jornadaIdx,
          current_standings: currentStandings,
          current_player_stats: currentPlayerStats,
          semilla,
        })
        setCurrentStandings(resp.updated_standings)
        setCurrentPlayerStats(resp.updated_player_stats)
        setCurrentJornadaIndex(resp.jornada_index)

        const snapshot = {
          jornada_index: resp.jornada_index,
          jornada_num: resp.jornada_num ?? resp.jornada_index + 1,
          matches: resp.matches,
        }
        setJornadaHistory((prev) => {
          const filtered = prev.filter((j) => j.jornada_index !== snapshot.jornada_index)
          return [...filtered, snapshot].sort((a, b) => a.jornada_index - b.jornada_index)
        })
        if (!userSelectedJornadaRef.current) {
          setSelectedJornadaIndex(resp.jornada_index)
        }
        if (resp.is_complete) {
          setPhase('complete')
          setIsPaused(false)
        }
      } catch (err) {
        setError(err.message)
        setIsPaused(true)
      } finally {
        setIsLoadingJornada(false)
      }
    },
    [
      seleccionados,
      dobleVuelta,
      schedule,
      totalJornadas,
      currentStandings,
      currentPlayerStats,
      semilla,
    ],
  )

  useEffect(() => {
    if (phase !== 'running' || isPaused || isLoadingJornada) return
    if (currentJornadaIndex >= totalJornadas - 1) return

    timerRef.current = setTimeout(() => {
      fetchJornada(currentJornadaIndex + 1)
    }, JORNADA_DELAY_MS)

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [phase, isPaused, isLoadingJornada, currentJornadaIndex, totalJornadas, fetchJornada])

  // Nota: aquí NO hay scrollIntoView. La versión anterior arrastraba la
  // página al principio cada 5 s, así que era imposible leer la
  // clasificación mientras avanzaba la temporada. Ahora los resultados y la
  // clasificación se ven a la vez y la página no se mueve sola.

  const siguienteJornada = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    if (currentJornadaIndex < totalJornadas - 1) fetchJornada(currentJornadaIndex + 1)
  }

  const nuevaSimulacion = () => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setPhase('config')
    setSchedule([])
    setTotalJornadas(0)
    setCurrentJornadaIndex(-1)
    setCurrentStandings([])
    setCurrentPlayerStats([])
    setJornadaHistory([])
    setSelectedJornadaIndex(null)
    userSelectedJornadaRef.current = false
    setIsPaused(false)
    setIsLoadingJornada(false)
    setError(null)
  }

  const selectJornada = (idx) => {
    userSelectedJornadaRef.current = true
    setSelectedJornadaIndex(idx)
  }

  // ─────────────────────────────────────────────────────────
  // Configuración
  // ─────────────────────────────────────────────────────────

  if (phase === 'config') {
    const grupo = (titulo, lista) =>
      lista.length > 0 && (
        <>
          <div className="kicker mt-4" style={{ marginBottom: 'var(--s2)' }}>
            {titulo}
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
              gap: 'var(--s1)',
            }}
          >
            {lista.map((eq) => {
              const selected = seleccionados.includes(eq.nombre)
              const disabled = !selected && seleccionados.length >= MAX_EQUIPOS
              return (
                <button
                  key={eq.nombre}
                  type="button"
                  aria-pressed={selected}
                  disabled={disabled}
                  onClick={() => toggleEquipo(eq.nombre)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--s2)',
                    minHeight: 44,
                    padding: '0 var(--s3)',
                    background: selected ? 'var(--surface-3)' : 'var(--bg)',
                    border: 'none',
                    borderLeft: `4px solid ${
                      selected ? teamMarkColor(eq.colores) : 'var(--line-strong)'
                    }`,
                    color: disabled
                      ? 'var(--text-ghost)'
                      : selected
                        ? 'var(--text-strong)'
                        : 'var(--text-dim)',
                    fontFamily: 'var(--font-cond)',
                    fontSize: 18,
                    fontWeight: selected ? 700 : 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.02em',
                    cursor: disabled ? 'not-allowed' : 'pointer',
                    opacity: disabled ? 0.4 : 1,
                    textAlign: 'left',
                  }}
                >
                  {eq.nombre}
                </button>
              )
            })}
          </div>
        </>
      )

    return (
      <section className="page">
        <div className="page-head">
          <h1 className="page-title">Simular temporada</h1>
          <p className="page-sub">
            Una liga completa jornada a jornada, con pausa y avance manual.
          </p>
        </div>

        <div className="panel panel-pad">
          <div className="section-head" style={{ marginBottom: 'var(--s3)' }}>
            <h2 className="cond" style={{ fontSize: 24, fontWeight: 800, textTransform: 'uppercase' }}>
              Equipos{' '}
              <span
                className="mono"
                style={{
                  fontSize: 15,
                  color: seleccionados.length >= MAX_EQUIPOS ? 'var(--warn)' : 'var(--lime)',
                }}
              >
                {seleccionados.length}/{MAX_EQUIPOS}
              </span>
            </h2>
            <div className="btn-group">
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => setSeleccionados(equiposActuales.map((e) => e.nombre))}
              >
                <CheckSquare size={14} aria-hidden="true" /> Actuales
              </button>
              <button type="button" className="btn btn-sm" onClick={() => setSeleccionados([])}>
                <Square size={14} aria-hidden="true" /> Ninguno
              </button>
            </div>
          </div>

          {seleccionados.length >= MAX_EQUIPOS && (
            <p className="notice">
              <AlertCircle size={15} aria-hidden="true" />
              Límite de {MAX_EQUIPOS} equipos alcanzado. Deselecciona uno para añadir otro.
            </p>
          )}

          {grupo('Temporada actual', equiposActuales)}
          {grupo('Históricos', equiposHistoricos)}

          <div
            className="mt-6"
            style={{ display: 'flex', gap: 'var(--s7)', alignItems: 'center', flexWrap: 'wrap' }}
          >
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={dobleVuelta}
                onChange={(e) => setDobleVuelta(e.target.checked)}
              />
              Doble vuelta (ida y vuelta)
            </label>
            <label className="checkbox-row" htmlFor="semilla">
              Semilla
              <input
                id="semilla"
                type="number"
                min="0"
                value={semilla}
                onChange={(e) => setSemilla(Math.max(0, parseInt(e.target.value, 10) || 0))}
                style={{ width: 96, minHeight: 38 }}
              />
            </label>
            <span style={{ fontSize: 12.5, color: 'var(--text-faint)' }}>
              La semilla fija el calendario y los resultados.
            </span>
          </div>

          {error && (
            <p className="notice notice-error mt-4" role="alert">
              {error}
            </p>
          )}

          <div className="mt-6">
            <button
              type="button"
              className="btn btn-primary btn-lg"
              style={{ width: '100%' }}
              onClick={iniciarSimulacion}
              disabled={isLoadingJornada || seleccionados.length < 2}
            >
              {isLoadingJornada ? (
                <>
                  <Loader2 size={18} className="spin-icon" aria-hidden="true" /> Inicializando…
                </>
              ) : (
                <>
                  <Play size={18} fill="currentColor" aria-hidden="true" /> Iniciar simulación (
                  {seleccionados.length} equipos, {Math.round(totalPartidos)} partidos)
                </>
              )}
            </button>
          </div>
        </div>
      </section>
    )
  }

  // ─────────────────────────────────────────────────────────
  // En curso / completada
  // ─────────────────────────────────────────────────────────

  const isComplete = phase === 'complete'
  const standings = currentStandings
  const topPlayers = [...currentPlayerStats]
    .sort((a, b) => (b.puntos || 0) - (a.puntos || 0))
    .slice(0, 10)

  const selectedJornada =
    selectedJornadaIndex !== null
      ? jornadaHistory.find((j) => j.jornada_index === selectedJornadaIndex) || null
      : jornadaHistory[jornadaHistory.length - 1] || null

  const displayIdx = selectedJornadaIndex ?? currentJornadaIndex
  const viendoPasada =
    selectedJornadaIndex !== null && selectedJornadaIndex !== currentJornadaIndex

  const progressPct =
    totalJornadas > 0 ? Math.round(((currentJornadaIndex + 1) / totalJornadas) * 100) : 0

  const faseLabel = dobleVuelta
    ? displayIdx < totalJornadas / 2
      ? 'Ida'
      : 'Vuelta'
    : null

  return (
    <>
      {/* ── Barra de estado ── */}
      <div className="livebar">
        <div className="livebar-inner">
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--s5)', flexWrap: 'wrap' }}>
            <span
              className="kicker kicker-lime live-pill"
              style={{
                background: isComplete ? 'var(--lime)' : isPaused ? 'var(--warn)' : 'var(--lime)',
              }}
            >
              <span className="dot" aria-hidden="true" />
              {isComplete ? 'Finalizada' : isPaused ? 'En pausa' : 'En directo'}
            </span>
            <span
              className="cond"
              style={{
                fontSize: 30,
                fontWeight: 800,
                textTransform: 'uppercase',
                lineHeight: 1,
                color: 'var(--text-strong)',
              }}
            >
              Jornada {displayIdx + 1}{' '}
              <span style={{ color: 'var(--text-ghost)', fontWeight: 600 }}>
                / {totalJornadas}
                {faseLabel ? ` · ${faseLabel}` : ''}
              </span>
            </span>
          </div>

          <div className="btn-group">
            {!isComplete && (
              <>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setIsPaused((v) => !v)}
                  disabled={isLoadingJornada && currentJornadaIndex === -1}
                >
                  {isPaused ? (
                    <>
                      <Play size={14} fill="currentColor" aria-hidden="true" /> Reanudar
                    </>
                  ) : (
                    <>
                      <Pause size={14} fill="currentColor" aria-hidden="true" /> Pausar
                    </>
                  )}
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={siguienteJornada}
                  disabled={isLoadingJornada || currentJornadaIndex >= totalJornadas - 1}
                >
                  <SkipForward size={14} aria-hidden="true" /> Siguiente
                </button>
              </>
            )}
            <button type="button" className="btn btn-sm btn-ghost" onClick={nuevaSimulacion}>
              <RotateCcw size={14} aria-hidden="true" /> Reiniciar
            </button>
          </div>
        </div>
        <div className="progress">
          <span className="progress-fill" style={{ width: `${progressPct}%` }} />
        </div>
      </div>

      <div className="sr-only" aria-live="polite">
        Jornada {currentJornadaIndex + 1} de {totalJornadas} simulada.
      </div>

      <section className="page" style={{ paddingTop: 'var(--s6)' }}>
        {error && (
          <p className="notice notice-error" role="alert" style={{ marginBottom: 'var(--s5)' }}>
            {error}
          </p>
        )}

        <div className="split">
          {/* ── Resultados de la jornada ── */}
          <div>
            <div className="section-head">
              <h2 className="section-title">Resultados</h2>
              {isLoadingJornada && (
                <Loader2 size={16} className="spin-icon" aria-label="Simulando jornada" />
              )}
            </div>

            {jornadaHistory.length > 0 && (
              <div
                className="chip-scroll"
                role="group"
                aria-label="Elegir jornada"
                style={{ marginBottom: 'var(--s4)' }}
              >
                {jornadaHistory.map((j) => (
                  <button
                    key={j.jornada_index}
                    type="button"
                    className="chip"
                    aria-pressed={j.jornada_index === selectedJornadaIndex}
                    onClick={() => selectJornada(j.jornada_index)}
                  >
                    J{j.jornada_num}
                  </button>
                ))}
              </div>
            )}

            {viendoPasada && (
              <p className="notice" style={{ marginBottom: 'var(--s3)' }}>
                <AlertCircle size={15} aria-hidden="true" />
                Estás viendo una jornada anterior. La clasificación de la derecha es siempre la
                actual.
              </p>
            )}

            {selectedJornada ? (
              <div className="stack" style={{ gap: 'var(--s1)' }}>
                {selectedJornada.matches.map((m, i) => {
                  const localGano = m.ganador === m.local
                  return (
                    <div key={`${m.local}-${m.visitante}-${i}`} className="fixture fade-in">
                      <span aria-hidden="true" style={{ background: colorDe(m.local) }} />
                      <span className={`fixture-team home ${localGano ? 'winner' : ''}`}>
                        <span>{m.local}</span>
                      </span>
                      <span className="fixture-score">{m.resultado?.replace('-', '–')}</span>
                      <span className={`fixture-team ${!localGano ? 'winner' : ''}`}>
                        <span>{m.visitante}</span>
                      </span>
                      <span aria-hidden="true" style={{ background: colorDe(m.visitante) }} />
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="loading">
                <span className="spinner" aria-hidden="true" />
                Simulando la primera jornada…
              </div>
            )}

            {topPlayers.length > 0 && (
              <section className="mt-7">
                <h2 className="section-title" style={{ marginBottom: 'var(--s4)' }}>
                  Máximos anotadores
                </h2>
                <PlayerTable
                  players={topPlayers}
                  colorFor={coloresDe}
                  showRank
                  showTeam
                  showMatches
                  caption="Jugadores con más puntos en la temporada simulada"
                />
              </section>
            )}
          </div>

          {/* ── Clasificación: siempre visible, al lado ── */}
          <div>
            <div className="section-head">
              <h2 className="section-title">Clasificación</h2>
              <span className="kicker">
                tras {currentJornadaIndex + 1} jornada{currentJornadaIndex === 0 ? '' : 's'}
              </span>
            </div>

            <div className="table-wrap">
              <table>
                <caption className="sr-only">
                  Clasificación acumulada de la temporada simulada
                </caption>
                <thead>
                  <tr>
                    <th scope="col" style={{ width: 32 }}>
                      #
                    </th>
                    <th scope="col">Equipo</th>
                    <th scope="col" className="num">
                      Pts
                    </th>
                    <th scope="col" className="num">
                      PJ
                    </th>
                    <th scope="col" className="num">
                      PG
                    </th>
                    <th scope="col" className="num">
                      PP
                    </th>
                    <th scope="col" className="num">
                      SG–SP
                    </th>
                    <th scope="col" className="num">
                      Ratio
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {standings.map((row, i) => (
                    <tr
                      key={row.equipo}
                      className={i === PLAYOFF_CUT ? 'cut-row' : undefined}
                      title={i === PLAYOFF_CUT ? 'Corte de playoff' : undefined}
                    >
                      <td
                        className="cond"
                        style={{
                          fontSize: 19,
                          fontWeight: 800,
                          color: i < 3 ? 'var(--lime)' : 'var(--text-faint)',
                        }}
                      >
                        {i + 1}
                      </td>
                      <td>
                        <span
                          style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s2)' }}
                        >
                          <TeamDot colores={coloresDe(row.equipo)} />
                          <span className="t-name">{row.equipo}</span>
                        </span>
                      </td>
                      <td className="num" style={{ fontWeight: 700, color: 'var(--lime)' }}>
                        {row.puntos}
                      </td>
                      <td className="num" style={{ color: 'var(--text-dim)' }}>
                        {row.pj}
                      </td>
                      <td className="num" style={{ color: 'var(--text-dim)' }}>
                        {row.pg}
                      </td>
                      <td
                        className="num"
                        style={{ color: row.pp > 0 ? 'var(--text-dim)' : 'var(--text-ghost)' }}
                      >
                        {row.pp}
                      </td>
                      <td className="num" style={{ color: 'var(--text-dim)' }}>
                        {row.sg}–{row.sp}
                      </td>
                      <td className="num">{row.sr}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div
              className="mt-4"
              style={{ display: 'flex', alignItems: 'center', gap: 'var(--s2)' }}
            >
              <span aria-hidden="true" style={{ width: 22, height: 2, background: 'var(--lime)' }} />
              <span className="kicker">Corte de playoff ({PLAYOFF_CUT} primeros)</span>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
