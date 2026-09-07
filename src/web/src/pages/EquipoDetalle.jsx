import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Swords } from 'lucide-react'
import { getEquipo } from '../api.js'
import { fmtRatio, formatPlayerName, readableOn, teamMarkColor } from '../lib/format.js'

export default function EquipoDetalle() {
  const { nombre } = useParams()
  // Un único estado que recuerda a qué equipo corresponde. Así no hace falta
  // un setState síncrono dentro del efecto para volver a "cargando" cuando
  // cambia el parámetro de la ruta.
  const [carga, setCarga] = useState({ para: null, equipo: null, error: null })

  useEffect(() => {
    let vivo = true
    getEquipo(nombre)
      .then((data) => vivo && setCarga({ para: nombre, equipo: data, error: null }))
      .catch((err) => vivo && setCarga({ para: nombre, equipo: null, error: err.message }))
    return () => {
      vivo = false
    }
  }, [nombre])

  const loading = carga.para !== nombre
  const equipo = loading ? null : carga.equipo
  const error = loading ? null : carga.error

  if (loading) {
    return (
      <section className="page">
        <div className="loading">
          <span className="spinner" aria-hidden="true" />
          Cargando equipo…
        </div>
      </section>
    )
  }

  if (error || !equipo) {
    return (
      <section className="page">
        <div className="page-head">
          <h1 className="page-title">Equipo no encontrado</h1>
          <p className="page-sub">{error || `No hay datos para «${nombre}».`}</p>
        </div>
        <Link to="/" className="btn">
          <ArrowLeft size={16} aria-hidden="true" /> Volver al inicio
        </Link>
      </section>
    )
  }

  const color = teamMarkColor(equipo.colores)
  const jugadores = equipo.jugadores || []

  return (
    <>
      <div
        className="page-bleed"
        style={{ background: color, color: readableOn(color) }}
      >
        <div style={{ padding: 'var(--s8) var(--gutter)' }}>
          <div className="kicker" style={{ color: 'currentColor', opacity: 0.7 }}>
            SuperLega {equipo.temporada}
          </div>
          <h1
            className="cond"
            style={{
              fontSize: 'clamp(40px, 6vw, 72px)',
              fontWeight: 800,
              lineHeight: 0.9,
              textTransform: 'uppercase',
              letterSpacing: '-0.01em',
              marginTop: 'var(--s2)',
            }}
          >
            {equipo.nombre}
          </h1>
          <div
            className="mono"
            style={{ marginTop: 'var(--s3)', fontSize: 14, opacity: 0.85 }}
          >
            Fuerza {fmtRatio(equipo.fuerza)} · {jugadores.length} jugadores
          </div>
        </div>
      </div>

      <section className="page" style={{ paddingTop: 'var(--s6)' }}>
        <div className="btn-group" style={{ marginBottom: 'var(--s7)' }}>
          <Link to="/simular-partido" className="btn btn-primary">
            <Swords size={16} aria-hidden="true" /> Simular partido
          </Link>
          <Link to="/" className="btn">
            <ArrowLeft size={16} aria-hidden="true" /> Volver
          </Link>
        </div>

        <div className="section-head">
          <h2 className="section-title">Plantilla</h2>
          <span className="kicker">Medias por set de la temporada</span>
        </div>

        {jugadores.length === 0 ? (
          <div className="empty">No hay datos de jugadores para este equipo</div>
        ) : (
          <div className="table-wrap">
            <table>
              <caption className="sr-only">Plantilla de {equipo.nombre} con medias por set</caption>
              <thead>
                <tr>
                  <th scope="col" style={{ width: 32 }}>
                    #
                  </th>
                  <th scope="col">Jugador</th>
                  <th scope="col" className="num">
                    Pts/set
                  </th>
                  <th scope="col" className="num">
                    Ace/set
                  </th>
                  <th scope="col" className="num">
                    Atq/set
                  </th>
                  <th scope="col" className="num">
                    Blq/set
                  </th>
                  <th scope="col" className="num">
                    Pts total
                  </th>
                </tr>
              </thead>
              <tbody>
                {jugadores.map((j, i) => (
                  <tr key={j.nombre || i}>
                    <td
                      className="cond"
                      style={{
                        fontSize: 18,
                        fontWeight: 800,
                        color: i === 0 ? 'var(--lime)' : 'var(--text-faint)',
                      }}
                    >
                      {i + 1}
                    </td>
                    <td>
                      <span className="t-name">{formatPlayerName(j.nombre)}</span>
                    </td>
                    <td
                      className="num"
                      style={{ fontWeight: 700, color: i === 0 ? 'var(--lime)' : 'var(--text)' }}
                    >
                      {j.puntos_por_set?.toFixed(2) ?? '—'}
                    </td>
                    <td className="num" style={{ color: 'var(--text-dim)' }}>
                      {j.aces_por_set?.toFixed(2) ?? '—'}
                    </td>
                    <td className="num" style={{ color: 'var(--text-dim)' }}>
                      {j.ataques_ganados_por_set?.toFixed(2) ?? '—'}
                    </td>
                    <td className="num" style={{ color: 'var(--text-dim)' }}>
                      {j.bloqueos_por_set?.toFixed(2) ?? '—'}
                    </td>
                    <td className="num" style={{ color: 'var(--text-dim)' }}>
                      {j.puntos_total ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  )
}
