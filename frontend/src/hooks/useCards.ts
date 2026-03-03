import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { cardsApi } from '../api/client';
import type { ProductUpdate } from '../types';

export function useCards(page = 1, perPage = 20) {
  return useQuery({
    queryKey: ['cards', page, perPage],
    queryFn: () => cardsApi.list(page, perPage),
  });
}

export function useCard(id: number) {
  return useQuery({
    queryKey: ['cards', id],
    queryFn: () => cardsApi.get(id),
    enabled: !!id,
  });
}

export function useGenerateCard() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => cardsApi.generate(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cards'] });
    },
  });
}

export function useUpdateCard() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: ProductUpdate }) =>
      cardsApi.update(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['cards', id] });
      queryClient.invalidateQueries({ queryKey: ['cards'] });
    },
  });
}

export function useDeleteCard() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => cardsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cards'] });
    },
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => cardsApi.health(),
    refetchInterval: 30000,
  });
}
