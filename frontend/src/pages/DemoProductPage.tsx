import { useParams } from 'react-router-dom';
import { useCard } from '../hooks/useCards';
import SEOPreview from '../components/SEOPreview';
import LoadingSpinner from '../components/LoadingSpinner';

export default function DemoProductPage() {
  const { id } = useParams<{ id: string }>();
  const { data: product, isLoading, isError } = useCard(Number(id));

  if (isLoading) return <LoadingSpinner />;
  if (isError || !product) return <div className="text-red-600">Товар не найден</div>;

  const price = '1 299 ₽';
  const sku = product.id.toString().padStart(5, '0');

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="p-8 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Фото товара */}
            <div className="bg-gray-100 rounded-lg overflow-hidden">
              <img
                src={`/${product.image_path}`}
                alt={product.title}
                className="w-full h-full object-contain max-h-[400px]"
              />
            </div>

            {/* Информация о товаре */}
            <div className="space-y-4">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">{product.title}</h1>
                <p className="text-sm text-gray-500 mt-1">Артикул: {sku}</p>
              </div>
              <div className="text-3xl font-bold text-indigo-600">{price}</div>
              <p className="text-gray-600 leading-relaxed">{product.description}</p>
              <button className="px-8 py-3 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors">
                В корзину
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* SEO Preview */}
      <SEOPreview
        title={product.seo_title || product.title}
        description={product.seo_description || product.description}
        keywords={product.seo_keywords}
        url={`http://localhost:5173/demo-product/${product.id}`}
      />
    </div>
  );
}
