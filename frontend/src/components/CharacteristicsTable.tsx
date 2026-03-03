interface Props {
  characteristics: Record<string, string>;
}

export default function CharacteristicsTable({ characteristics }: Props) {
  const entries = Object.entries(characteristics);

  if (entries.length === 0) {
    return (
      <p className="text-sm text-gray-400 italic">Характеристики не определены</p>
    );
  }

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <tbody>
          {entries.map(([key, value], i) => (
            <tr key={key} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
              <td className="px-4 py-2 font-medium text-gray-700 w-1/3">{key}</td>
              <td className="px-4 py-2 text-gray-600">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
