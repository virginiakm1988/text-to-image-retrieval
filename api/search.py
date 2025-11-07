from http.server import BaseHTTPRequestHandler
import json
import os
import requests
from typing import List, Dict, Any

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle text search requests using external APIs"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            
            if not query:
                self.send_json_response({'error': 'Query is required'}, 400)
                return
            
            # Option 1: Use NVIDIA NIM API (recommended)
            nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
            if nvidia_api_key:
                results = self.search_with_nvidia_nim(query, top_k, nvidia_api_key)
            else:
                # Option 2: Use Hugging Face Inference API
                hf_api_key = os.environ.get('HUGGINGFACE_API_KEY')
                if hf_api_key:
                    results = self.search_with_huggingface(query, top_k, hf_api_key)
                else:
                    # Option 3: Mock results for demo
                    results = self.get_demo_results(query, top_k)
            
            self.send_json_response({'results': results})
            
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def search_with_nvidia_nim(self, query: str, top_k: int, api_key: str) -> List[Dict[str, Any]]:
        """Search using NVIDIA NIM API"""
        try:
            # This is a simplified example - you'd need to adapt based on your indexed data
            # For a real implementation, you'd need to:
            # 1. Store your image embeddings in a database (like Pinecone, Weaviate, or Supabase)
            # 2. Encode the query text using NVIDIA NIM
            # 3. Search the database for similar embeddings
            
            # Example NVIDIA NIM API call for text encoding
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Encode the query text
            text_embedding_response = requests.post(
                'https://integrate.api.nvidia.com/v1/embeddings',
                headers=headers,
                json={
                    'input': [query],
                    'model': 'nvidia/nvclip',
                    'encoding_format': 'float'
                },
                timeout=30
            )
            
            if text_embedding_response.status_code == 200:
                # In a real implementation, you'd use this embedding to search your vector database
                # For now, return demo results
                return self.get_demo_results(query, top_k)
            else:
                return self.get_demo_results(query, top_k)
                
        except Exception as e:
            print(f"NVIDIA NIM API error: {e}")
            return self.get_demo_results(query, top_k)
    
    def search_with_huggingface(self, query: str, top_k: int, api_key: str) -> List[Dict[str, Any]]:
        """Search using Hugging Face Inference API"""
        try:
            # Example using Hugging Face Inference API
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # You could use sentence-transformers models via HF API
            response = requests.post(
                'https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32',
                headers=headers,
                json={'inputs': query},
                timeout=30
            )
            
            if response.status_code == 200:
                # Process the response and search your vector database
                return self.get_demo_results(query, top_k)
            else:
                return self.get_demo_results(query, top_k)
                
        except Exception as e:
            print(f"Hugging Face API error: {e}")
            return self.get_demo_results(query, top_k)
    
    def get_demo_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return demo results based on query keywords"""
        
        # Sample image database with categories
        image_database = {
            'cat': [
                {'url': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300', 'filename': 'orange_cat.jpg', 'score': 0.95},
                {'url': 'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=300', 'filename': 'white_cat.jpg', 'score': 0.89},
                {'url': 'https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=300', 'filename': 'black_cat.jpg', 'score': 0.85},
            ],
            'dog': [
                {'url': 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=300', 'filename': 'golden_dog.jpg', 'score': 0.92},
                {'url': 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=300', 'filename': 'puppy.jpg', 'score': 0.88},
                {'url': 'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=300', 'filename': 'small_dog.jpg', 'score': 0.82},
            ],
            'mountain': [
                {'url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300', 'filename': 'snow_mountain.jpg', 'score': 0.91},
                {'url': 'https://images.unsplash.com/photo-1464822759844-d150baec3e5e?w=300', 'filename': 'mountain_lake.jpg', 'score': 0.87},
                {'url': 'https://images.unsplash.com/photo-1551632811-561732d1e306?w=300', 'filename': 'mountain_peak.jpg', 'score': 0.83},
            ],
            'city': [
                {'url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=300', 'filename': 'city_skyline.jpg', 'score': 0.90},
                {'url': 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1f?w=300', 'filename': 'night_city.jpg', 'score': 0.86},
                {'url': 'https://images.unsplash.com/photo-1514565131-fce0801e5785?w=300', 'filename': 'urban_street.jpg', 'score': 0.81},
            ],
            'airplane': [
                {'url': 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=300', 'filename': 'airplane_sky.jpg', 'score': 0.93},
                {'url': 'https://images.unsplash.com/photo-1569629743817-70d8db6c323b?w=300', 'filename': 'jet_plane.jpg', 'score': 0.89},
                {'url': 'https://images.unsplash.com/photo-1544636331-e26879cd4d9b?w=300', 'filename': 'plane_wing.jpg', 'score': 0.85},
            ]
        }
        
        # Simple keyword matching
        query_lower = query.lower()
        results = []
        
        for category, images in image_database.items():
            if category in query_lower:
                results.extend([
                    {
                        'image_url': img['url'],
                        'filename': img['filename'],
                        'similarity_score': img['score']
                    }
                    for img in images
                ])
        
        # If no category matches, return a mix of all categories
        if not results:
            all_images = []
            for images in image_database.values():
                all_images.extend(images)
            
            # Sort by score and take top results
            all_images.sort(key=lambda x: x['score'], reverse=True)
            results = [
                {
                    'image_url': img['url'],
                    'filename': img['filename'],
                    'similarity_score': img['score'] * 0.7  # Lower score for generic results
                }
                for img in all_images[:top_k]
            ]
        
        return results[:top_k]
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data)
        self.wfile.write(response.encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
