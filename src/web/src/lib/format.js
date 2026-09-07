/**
 * format.js — Helpers de presentación compartidos por todas las páginas.
 */

const FALLBACK_COLOR = '#607D8B'

/**
 * El backend entrega los jugadores como «Apellido Nombre»
 * («Bovolenta Alessandro», «Ferreira Souza Darlan»). Tomar el último token
 * mostraba el nombre de pila suelto y dos jugadores del mismo equipo podían
 * aparecer ambos como «Alessandro». Devolvemos apellido completo + inicial.
 */
export function formatPlayerName(fullName) {
  if (!fullName) return '—'
  const parts = String(fullName).trim().split(/\s+/)
  if (parts.length === 1) return parts[0]
  const given = parts[parts.length - 1]
  const surname = parts.slice(0, -1).join(' ')
  return `${surname} ${given.charAt(0).toUpperCase()}.`
}

/** Sólo el apellido, para tablas muy estrechas. */
export function playerSurname(fullName) {
  if (!fullName) return '—'
  const parts = String(fullName).trim().split(/\s+/)
  return parts.length === 1 ? parts[0] : parts.slice(0, -1).join(' ')
}

/** Abreviatura de tres letras: única para los 16 equipos del dataset. */
export function teamAbbr(name) {
  if (!name) return '???'
  return String(name).slice(0, 3).toUpperCase()
}

function hexToRgb(hex) {
  const clean = String(hex || '').replace('#', '')
  if (clean.length !== 6) return null
  const int = parseInt(clean, 16)
  if (Number.isNaN(int)) return null
  return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 }
}

/** Luminancia relativa WCAG. */
function luminance(hex) {
  const rgb = hexToRgb(hex)
  if (!rgb) return 0
  const [r, g, b] = [rgb.r, rgb.g, rgb.b].map((v) => {
    const s = v / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/**
 * Color de texto legible sobre un color de club.
 * Perugia es #000000 y Trento #FFD700: no vale un blanco fijo.
 */
export function readableOn(bg) {
  return luminance(bg) > 0.45 ? '#0A0A0C' : '#FFFFFF'
}

/**
 * Color de club utilizable como marca sobre fondo negro. Un club cuyo color
 * primario es casi negro (Perugia) se representa con su color secundario,
 * porque sobre #0A0A0C sería invisible.
 */
export function teamMarkColor(colores) {
  const primary = colores?.primary || FALLBACK_COLOR
  if (luminance(primary) < 0.045) {
    const secondary = colores?.secondary
    if (secondary && luminance(secondary) >= 0.045) return secondary
  }
  return primary
}

/** Mezcla un color con negro. `amount` 0 = igual, 1 = negro. */
export function shade(hex, amount) {
  const rgb = hexToRgb(hex)
  if (!rgb) return hex
  const k = 1 - Math.min(Math.max(amount, 0), 1)
  const to2 = (v) =>
    Math.round(v * k)
      .toString(16)
      .padStart(2, '0')
  return `#${to2(rgb.r)}${to2(rgb.g)}${to2(rgb.b)}`
}

/** Distancia euclídea en RGB: suficiente para detectar colores confundibles. */
function colorDistance(a, b) {
  const ra = hexToRgb(a)
  const rb = hexToRgb(b)
  if (!ra || !rb) return Infinity
  return Math.sqrt((ra.r - rb.r) ** 2 + (ra.g - rb.g) ** 2 + (ra.b - rb.b) ** 2)
}

/**
 * Devuelve `candidato` oscurecido si es demasiado parecido a `referencia`.
 *
 * Hace falta porque hay enfrentamientos con dos clubes casi del mismo color:
 * Trento es #FFD700 y Perugia cae en su dorado secundario #D4AF37, así que
 * los dos campos del marcador quedaban indistinguibles.
 */
export function ensureDistinct(referencia, candidato, umbral = 110) {
  if (colorDistance(referencia, candidato) >= umbral) return candidato
  return shade(candidato, 0.5)
}

/** Busca un equipo en la lista y devuelve su paleta. */
export function teamColors(equipos, nombre) {
  const eq = equipos?.find((e) => e.nombre === nombre)
  return eq?.colores || { primary: FALLBACK_COLOR, secondary: '#FFFFFF' }
}

/** 0.833 → «.833» — sin cero inicial, como los ratios deportivos. */
export function fmtRatio(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toFixed(3).replace(/^0/, '')
}

/** 0.467 → «46.7 %» */
export function fmtPct(value, decimals = 1) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(decimals)}%`
}

/** 1322 -> "1 322", con espacio fino no separable (U+202F), como en la memoria. */
export function fmtInt(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toLocaleString('es-ES').replace(/[\s.]/g, ' ')
}

/** 0.7624 → «0.762» */
export function fmtMetric(value, decimals = 3) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return value.toFixed(decimals)
}
