/**
 * TeamMark.jsx — Marcas de club reutilizables.
 *
 * Todas resuelven el color por `teamMarkColor` y el texto por `readableOn`,
 * para que ningún escudo quede invisible sobre el fondo negro.
 */
import { readableOn, teamAbbr, teamMarkColor } from '../lib/format.js'

/** Escudo cuadrado con las iniciales del club. */
export function Crest({ nombre, colores, size = 30 }) {
  const bg = teamMarkColor(colores)
  return (
    <span
      className="crest"
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        background: bg,
        color: readableOn(bg),
        fontSize: Math.round(size * 0.36),
      }}
    >
      {teamAbbr(nombre)}
    </span>
  )
}

/** Punto de color para tablas y listas densas. */
export function TeamDot({ colores }) {
  return (
    <span className="team-dot" aria-hidden="true" style={{ background: teamMarkColor(colores) }} />
  )
}

/** Barra vertical de club (banda lateral de un resultado). */
export function TeamBar({ colores }) {
  return (
    <span className="team-bar" aria-hidden="true" style={{ background: teamMarkColor(colores) }} />
  )
}

/** Nombre con punto de color delante, para celdas de tabla. */
export function TeamLabel({ nombre, colores }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 'var(--s2)' }}>
      <TeamDot colores={colores} />
      {nombre}
    </span>
  )
}
