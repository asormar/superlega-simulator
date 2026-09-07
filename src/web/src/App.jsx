import { useState } from 'react'
import { Routes, Route, NavLink, Link } from 'react-router-dom'
import { Volleyball, Menu, X } from 'lucide-react'
import Dashboard from './pages/Dashboard.jsx'
import SimularPartido from './pages/SimularPartido.jsx'
import SimularTemporada from './pages/SimularTemporada.jsx'
import EquipoDetalle from './pages/EquipoDetalle.jsx'

const NAV = [
  { to: '/', label: 'Inicio', end: true },
  { to: '/simular-partido', label: 'Partido', end: false },
  { to: '/simular-temporada', label: 'Temporada', end: false },
]

function App() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="app-layout">
      <a className="skip-link" href="#contenido">
        Saltar al contenido
      </a>

      <nav className="navbar" aria-label="Navegación principal">
        <div className="navbar-inner">
          <Link to="/" className="navbar-logo">
            <Volleyball size={20} aria-hidden="true" />
            SuperLega
          </Link>

          <button
            type="button"
            className="nav-toggle"
            aria-expanded={menuOpen}
            aria-controls="nav-principal"
            onClick={() => setMenuOpen((v) => !v)}
          >
            {menuOpen ? <X size={18} aria-hidden="true" /> : <Menu size={18} aria-hidden="true" />}
            Menú
          </button>

          <ul id="nav-principal" className={`navbar-nav ${menuOpen ? 'open' : ''}`}>
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
                  onClick={() => setMenuOpen(false)}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      <main id="contenido">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/simular-partido" element={<SimularPartido />} />
          <Route path="/simular-temporada" element={<SimularTemporada />} />
          <Route path="/equipo/:nombre" element={<EquipoDetalle />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
