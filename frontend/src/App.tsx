import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import CardsListPage from './pages/CardsListPage';
import CardDetailPage from './pages/CardDetailPage';
import DemoProductPage from './pages/DemoProductPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<UploadPage />} />
            <Route path="/cards" element={<CardsListPage />} />
            <Route path="/cards/:id" element={<CardDetailPage />} />
            <Route path="/demo-product/:id" element={<DemoProductPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
