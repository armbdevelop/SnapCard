import { useNavigate } from 'react-router-dom';
import ImageUploader from '../components/ImageUploader';
import { useGenerateCard } from '../hooks/useCards';

export default function UploadPage() {
  const navigate = useNavigate();
  const generateCard = useGenerateCard();

  const handleUpload = (file: File) => {
    generateCard.mutate(file, {
      onSuccess: (product) => {
        navigate(`/cards/${product.id}`);
      },
    });
  };

  return (
    <div className="max-w-2xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">Создать карточку товара</h1>
        <p className="text-gray-500">
          Загрузите фото товара, и мы автоматически создадим полную карточку с описанием, категорией и SEO
        </p>
      </div>

      <ImageUploader onUpload={handleUpload} isLoading={generateCard.isPending} />

      {generateCard.isError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
          Ошибка генерации: {(generateCard.error as Error).message}
        </div>
      )}
    </div>
  );
}
