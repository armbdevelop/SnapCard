interface Props {
  title: string;
  description: string;
  keywords: string;
}

export default function SEOPreview({ title, description, keywords }: Props) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wider">SEO превью</h3>

      {/* Google SERP mockup */}
      <div className="bg-white border border-gray-200 rounded-lg p-4 space-y-1">
        <p className="text-sm text-green-700 truncate">example.com/products/...</p>
        <p className="text-xl text-blue-800 hover:underline cursor-pointer leading-tight">
          {title || 'SEO заголовок не задан'}
        </p>
        <p className="text-sm text-gray-600 line-clamp-2">
          {description || 'SEO описание не задано'}
        </p>
      </div>

      {/* Character counts */}
      <div className="grid grid-cols-2 gap-4 text-xs">
        <div>
          <span className="text-gray-500">Заголовок: </span>
          <span className={title.length > 70 ? 'text-red-500 font-medium' : 'text-green-600'}>
            {title.length}/70
          </span>
        </div>
        <div>
          <span className="text-gray-500">Описание: </span>
          <span className={description.length > 160 ? 'text-red-500 font-medium' : 'text-green-600'}>
            {description.length}/160
          </span>
        </div>
      </div>

      {/* Keywords */}
      {keywords && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Ключевые слова:</p>
          <div className="flex flex-wrap gap-1">
            {keywords.split(',').map((kw, i) => (
              <span key={i} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                {kw.trim()}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
