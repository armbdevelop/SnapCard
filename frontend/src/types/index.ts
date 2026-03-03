export interface Product {
  id: number;
  image_path: string;
  original_filename: string;
  title: string;
  description: string;
  category: string;
  characteristics: Record<string, string>;
  tags: string[];
  seo_title: string;
  seo_description: string;
  seo_keywords: string;
  caption: string;
  confidence_score: number;
  created_at: string;
  updated_at: string;
}

export interface ProductListResponse {
  items: Product[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

export interface ProductUpdate {
  title?: string;
  description?: string;
  category?: string;
  characteristics?: Record<string, string>;
  tags?: string[];
  seo_title?: string;
  seo_description?: string;
  seo_keywords?: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: boolean;
  models_status: Record<string, boolean>;
}
