/**
 * PlayerTable.jsx — Tabla de estadísticas de jugador.
 *
 * Antes esta tabla estaba escrita cuatro veces (detalle por set, acumulado
 * del partido, top de temporada y plantilla) con estilos en línea distintos
 * en cada copia. Aquí vive una sola vez.
 */
import { formatPlayerName } from '../lib/format.js'
import { TeamDot } from './TeamMark.jsx'

export default function PlayerTable({
  players = [],
  colorFor,
  showRank = false,
  showTeam = false,
  showMatches = false,
  showBar = false,
  limit,
  caption,
  emptyText = 'Sin datos de jugadores',
}) {
  const rows = typeof limit === 'number' ? players.slice(0, limit) : players

  if (!rows.length) {
    return <div className="empty">{emptyText}</div>
  }

  return (
    <div className="table-wrap">
      <table>
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr>
            {showBar && <th scope="col" style={{ width: 5 }} aria-label="Club" />}
            {showRank && (
              <th scope="col" style={{ width: 32 }}>
                #
              </th>
            )}
            <th scope="col">Jugador</th>
            {showTeam && <th scope="col">Equipo</th>}
            <th scope="col" className="num">
              Pts
            </th>
            <th scope="col" className="num">
              Ace
            </th>
            <th scope="col" className="num">
              Atq
            </th>
            <th scope="col" className="num">
              Blq
            </th>
            <th scope="col" className="num">
              Rec
            </th>
            {showMatches && (
              <th scope="col" className="num">
                PJ
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((p, i) => {
            const colores = colorFor ? colorFor(p.equipo) : null
            return (
              <tr key={`${p.equipo || ''}|${p.jugador}|${i}`}>
                {showBar && (
                  <td style={{ padding: 0 }}>
                    <span
                      aria-hidden="true"
                      style={{
                        display: 'block',
                        width: 5,
                        height: 20,
                        background: colores ? undefined : 'transparent',
                        backgroundColor: p.barColor,
                      }}
                    />
                  </td>
                )}
                {showRank && (
                  <td
                    className="num mono"
                    style={{
                      textAlign: 'left',
                      color: i === 0 ? 'var(--lime)' : 'var(--text-faint)',
                      fontWeight: 700,
                    }}
                  >
                    {i + 1}
                  </td>
                )}
                <td>
                  <span className="t-name">{formatPlayerName(p.jugador)}</span>
                </td>
                {showTeam && (
                  <td style={{ color: 'var(--text-dim)' }}>
                    {colores ? (
                      <span
                        style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s2)' }}
                      >
                        <TeamDot colores={colores} />
                        {p.equipo}
                      </span>
                    ) : (
                      p.equipo
                    )}
                  </td>
                )}
                <td
                  className="num"
                  style={{ fontWeight: 700, color: i === 0 ? 'var(--lime)' : 'var(--text)' }}
                >
                  {p.puntos ?? 0}
                </td>
                <td className="num" style={{ color: 'var(--text-dim)' }}>
                  {p.aces ?? 0}
                </td>
                <td className="num" style={{ color: 'var(--text-dim)' }}>
                  {p.ataques_ganados ?? 0}
                </td>
                <td className="num" style={{ color: 'var(--text-dim)' }}>
                  {p.bloqueos ?? 0}
                </td>
                <td className="num" style={{ color: 'var(--text-dim)' }}>
                  {p.recepciones_exc ?? 0}
                </td>
                {showMatches && (
                  <td className="num" style={{ color: 'var(--text-faint)' }}>
                    {p.partidos ?? 0}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
