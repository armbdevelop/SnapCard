import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useCard, useUpdateCard, useDeleteCard } from '../hooks/useCards';
import LoadingSpinner from '../components/LoadingSpinner';
import SEOPreview from '../components/SEOPreview';
import CharacteristicsTable from '../components/CharacteristicsTable';

export default function CardDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: product, isLoading, isError } = useCard(Number(id));
  const updateCard = useUpdateCard();
  const deleteCard = useDeleteCard();
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState<Record<string, string>>({});

  if (isLoading) return <LoadingSpinner />;
  if (isError || !product) return <div className="text-red-600">Карточка не найдена</div>;

  const startEditing = () => {
    setEditData({
      title: product.title,
      description: product.description,
      category: product.category,
      seo_title: product.seo_title,
      seo_description: product.seo_description,
      seo_keywords: product.seo_keywords,
    });
    setIsEditing(true);
  };

  const saveEdits = () => {
    updateCard.mutate(
      { id: product.id, data: editData },
      { onSuccess: () => setIsEditing(false) }
    );
  };

  const handleDelete = () => {
    if (confirm('Удалить эту карточку?')) {
      deleteCard.mutate(product.id, {
        onSuccess: () => navigate('/cards'),
      });
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => navigate(-1)}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          &larr; Назад
        </button>
        <div className="flex gap-2">
          {isEditing ? (
            <>
              <button
                onClick={() => setIsEditing(false)}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Отмена
              </button>
              <button
                onClick={saveEdits}
                disabled={updateCard.isPending}
                className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                Сохранить
              </button>
            </>
          ) : (
            <>
              <Link
                to={`/demo-product/${product.id}`}
                className="px-4 py-2 text-sm border border-green-300 text-green-700 rounded-lg hover:bg-green-50"
              >
                Демо-страница
              </Link>
              <button
                onClick={startEditing}
                className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Редактировать
              </button>
              <button
                onClick={handleDelete}
                className="px-4 py-2 text-sm border border-red-300 text-red-600 rounded-lg hover:bg-red-50"
              >
                Удалить
              </button>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Image */}
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <img
              src={`/${product.image_path}`}
              alt={product.title}
              className="w-full object-contain max-h-[500px]"
            />
          </div>
          <div className="text-xs text-gray-400 space-y-1">
            <p>Файл: {product.original_filename}</p>
            {product.caption_ru && <p>Описание фото: {product.caption_ru}</p>}
            {product.confidence_score > 0 && (
              <p>Уверенность: {Math.round(product.confidence_score * 100)}%</p>
            )}
          </div>
        </div>

        {/* Details */}
        <div className="space-y-6">
          {/* Title */}
          <div>
            {isEditing ? (
              <input
                value={editData.title}
                onChange={(e) => setEditData({ ...editData, title: e.target.value })}
                className="w-full text-2xl font-bold border border-gray-300 rounded-lg px-3 py-2"
              />
            ) : (
              <h1 className="text-2xl font-bold text-gray-900">{product.title}</h1>
            )}
          </div>

          {/* Category */}
          <div>
            {isEditing ? (
              <input
                value={editData.category}
                onChange={(e) => setEditData({ ...editData, category: e.target.value })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                placeholder="Категория"
              />
            ) : (
              <span className="inline-block text-sm font-medium text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
                {product.category || 'Без категории'}
              </span>
            )}
          </div>

          {/* Description */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2">
              Описание
            </h3>
            {isEditing ? (
              <textarea
                value={editData.description}
                onChange={(e) => setEditData({ ...editData, description: e.target.value })}
                rows={4}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              />
            ) : (
              <p className="text-gray-600">{product.description}</p>
            )}
          </div>

          {/* Tags */}
          {product.tags.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2">
                Теги
              </h3>
              <div className="flex flex-wrap gap-2">
                {product.tags.map((tag) => (
                  <span
                    key={tag}
                    className="text-sm bg-gray-100 text-gray-700 px-3 py-1 rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Characteristics */}
          <div>
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider mb-2">
              Характеристики
            </h3>
            <CharacteristicsTable characteristics={product.characteristics} />
          </div>

          {/* SEO */}
          <div>
            {isEditing ? (
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider">SEO</h3>
                <input
                  value={editData.seo_title}
                  onChange={(e) => setEditData({ ...editData, seo_title: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="SEO заголовок"
                />
                <textarea
                  value={editData.seo_description}
                  onChange={(e) => setEditData({ ...editData, seo_description: e.target.value })}
                  rows={2}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="SEO описание"
                />
                <input
                  value={editData.seo_keywords}
                  onChange={(e) => setEditData({ ...editData, seo_keywords: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  placeholder="Ключевые слова (через запятую)"
                />
              </div>
            ) : (
              <SEOPreview
                title={product.seo_title}
                description={product.seo_description}
                keywords={product.seo_keywords}
                url={`http://localhost:5173/demo-product/${product.id}`}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
