// API configuration
// Prefer VITE_API_URL when provided; otherwise use '/api' so Vite proxy handles dev
// and a relative path works in production deployments
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export default {
  API_BASE_URL
};
