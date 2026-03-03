import { Link, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from '../hooks/useCards';

export default function Layout() {
  const location = useLocation();
  const { data: health } = useHealth();

  const navItems = [
    { path: '/', label: 'Загрузить' },
    { path: '/cards', label: 'Карточки' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link to="/" className="flex items-center gap-2">
              <span className="text-2xl font-bold text-indigo-600">SnapCard</span>
            </Link>
            <nav className="flex items-center gap-6">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`text-sm font-medium transition-colors ${
                    location.pathname === item.path
                      ? 'text-indigo-600'
                      : 'text-gray-500 hover:text-gray-900'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    health?.models_loaded ? 'bg-green-500' : 'bg-yellow-500'
                  }`}
                />
                <span className="text-xs text-gray-400">
                  {health?.models_loaded ? 'ML Ready' : 'ML Offline'}
                </span>
              </div>
            </nav>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}
