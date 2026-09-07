/**
 * api.js — Utility para comunicación con el backend FastAPI.
 */

const API_BASE = '/api'

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Error ${res.status}`)
  }
  return res.json()
}

export async function getEquipos() {
  return fetchJSON('/equipos')
}

export async function getEquipo(nombre) {
  return fetchJSON(`/equipos/${encodeURIComponent(nombre)}`)
}

export async function simularPartido(data) {
  return fetchJSON('/simular/partido', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function simularTemporada(data) {
  return fetchJSON('/simular/temporada', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function iniciarTemporada(data) {
  return fetchJSON('/simular/temporada/iniciar', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function simularJornada(data) {
  return fetchJSON('/simular/temporada/jornada', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function getModeloInfo() {
  return fetchJSON('/modelo/info')
}
