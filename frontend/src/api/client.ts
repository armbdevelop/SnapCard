import axios from 'axios';
import type { Product, ProductListResponse, ProductUpdate, HealthResponse } from '../types';

const api = axios.create({
  baseURL: '/api/v1',
});

export const cardsApi = {
  generate: async (file: File): Promise<Product> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<Product>('/cards/generate', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  list: async (page = 1, perPage = 20): Promise<ProductListResponse> => {
    const { data } = await api.get<ProductListResponse>('/cards', {
      params: { page, per_page: perPage },
    });
    return data;
  },

  get: async (id: number): Promise<Product> => {
    const { data } = await api.get<Product>(`/cards/${id}`);
    return data;
  },

  update: async (id: number, updates: ProductUpdate): Promise<Product> => {
    const { data } = await api.put<Product>(`/cards/${id}`, updates);
    return data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/cards/${id}`);
  },

  health: async (): Promise<HealthResponse> => {
    const { data } = await api.get<HealthResponse>('/health');
    return data;
  },
};
