import { useState } from 'react';
import { Link } from 'react-router-dom';
import ProductCard from '../components/ProductCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { useCards } from '../hooks/useCards';

export default function CardsListPage() {
  const [page, setPage] = useState(1);
  const { data, isLoading, isError } = useCards(page);

  if (isLoading) return <LoadingSpinner />;
  if (isError) return <div className="text-red-600">Ошибка загрузки карточек</div>;
  if (!data || data.items.length === 0) {
    return (
      <div className="text-center py-16 space-y-4">
        <p className="text-gray-500 text-lg">Карточки ещё не созданы</p>
        <Link
          to="/"
          className="inline-block px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
        >
          Создать первую
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Карточки товаров</h1>
        <span className="text-sm text-gray-500">Всего: {data.total}</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {data.items.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>

      {data.pages > 1 && (
        <div className="flex justify-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-4 py-2 text-sm border border-gray-300 rounded-lg disabled:opacity-50 hover:bg-gray-50"
          >
            Назад
          </button>
          <span className="px-4 py-2 text-sm text-gray-500">
            {page} / {data.pages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(data.pages, p + 1))}
            disabled={page === data.pages}
            className="px-4 py-2 text-sm border border-gray-300 rounded-lg disabled:opacity-50 hover:bg-gray-50"
          >
            Вперёд
          </button>
        </div>
      )}
    </div>
  );
}
