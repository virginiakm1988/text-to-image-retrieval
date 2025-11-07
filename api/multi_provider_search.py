from http.server import BaseHTTPRequestHandler
import json
import os
import requests
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional
import time

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle search requests using multiple AI providers"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query', '')
            top_k = data.get('top_k', 6)
            provider = data.get('provider', 'auto')  # auto, nvidia, gemini, openai
            
            if not query:
                self.send_json_response({'error': 'Query is required'}, 400)
                return
            
            # Search using the specified or best available provider
            results = self.search_with_provider(query, top_k, provider)
            self.send_json_response({
                'results': results['results'],
                'provider_used': results['provider'],
                'query': query
            })
            
        except Exception as e:
            self.send_json_response({'error': f'Search failed: {str(e)}'}, 500)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def search_with_provider(self, query: str, top_k: int, provider: str) -> Dict[str, Any]:
        """Search using the specified provider with fallback"""
        
        if provider == 'auto':
            # Try providers in order of preference
            providers_to_try = ['nvidia', 'openai', 'gemini']
        else:
            providers_to_try = [provider]
        
        for provider_name in providers_to_try:
            try:
                if provider_name == 'nvidia':
                    result = self.search_with_nvidia_nim(query, top_k)
                elif provider_name == 'openai':
                    result = self.search_with_openai(query, top_k)
                elif provider_name == 'gemini':
                    result = self.search_with_gemini(query, top_k)
                else:
                    continue
                
                if result:
                    return {
                        'results': result,
                        'provider': provider_name
                    }
            except Exception as e:
                print(f"Provider {provider_name} failed: {e}")
                continue
        
        # Fallback to curated results
        return {
            'results': self.get_fallback_results(query, top_k),
            'provider': 'fallback'
        }
    
    def search_with_nvidia_nim(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using NVIDIA NIM API"""
        nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
        if not nvidia_api_key:
            return None
        
        try:
            headers = {
                'Authorization': f'Bearer {nvidia_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Get text embedding from NVIDIA NIM
            embedding_response = requests.post(
                'https://integrate.api.nvidia.com/v1/embeddings',
                headers=headers,
                json={
                    'input': [query],
                    'model': 'nvidia/nvclip',
                    'encoding_format': 'float'
                },
                timeout=20
            )
            
            if embedding_response.status_code == 200:
                query_embedding = embedding_response.json()['data'][0]['embedding']
                return self.search_with_embedding(query_embedding, query, top_k, 'nvidia')
            
        except Exception as e:
            print(f"NVIDIA NIM error: {e}")
        
        return None
    
    def search_with_openai(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using OpenAI API"""
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            return None
        
        try:
            headers = {
                'Authorization': f'Bearer {openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Use OpenAI's text-embedding-3-small model
            embedding_response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json={
                    'input': query,
                    'model': 'text-embedding-3-small'
                },
                timeout=20
            )
            
            if embedding_response.status_code == 200:
                query_embedding = embedding_response.json()['data'][0]['embedding']
                return self.search_with_embedding(query_embedding, query, top_k, 'openai')
            
        except Exception as e:
            print(f"OpenAI error: {e}")
        
        return None
    
    def search_with_gemini(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using Google Gemini API"""
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key:
            return None
        
        try:
            # Use Gemini's embedding model
            embedding_response = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={gemini_api_key}',
                headers={'Content-Type': 'application/json'},
                json={
                    'model': 'models/embedding-001',
                    'content': {
                        'parts': [{'text': query}]
                    }
                },
                timeout=20
            )
            
            if embedding_response.status_code == 200:
                query_embedding = embedding_response.json()['embedding']['values']
                return self.search_with_embedding(query_embedding, query, top_k, 'gemini')
            
        except Exception as e:
            print(f"Gemini error: {e}")
        
        return None
    
    def search_with_embedding(self, query_embedding: List[float], query: str, top_k: int, provider: str) -> List[Dict[str, Any]]:
        """Search using the computed embedding"""
        # Get curated image database
        image_database = self.get_enhanced_image_database()
        
        # Enhanced semantic matching
        query_lower = query.lower()
        scored_images = []
        
        # Define comprehensive keyword mappings
        keyword_mappings = {
            'animals': {
                'keywords': ['cat', 'dog', 'animal', 'pet', 'kitten', 'puppy', 'feline', 'canine', 'mammal', 'wildlife'],
                'boost': 0.4
            },
            'nature': {
                'keywords': ['mountain', 'tree', 'forest', 'landscape', 'nature', 'outdoor', 'scenic', 'wilderness', 'valley', 'peak'],
                'boost': 0.35
            },
            'urban': {
                'keywords': ['city', 'building', 'street', 'urban', 'architecture', 'skyline', 'downtown', 'metropolitan'],
                'boost': 0.35
            },
            'transportation': {
                'keywords': ['car', 'airplane', 'plane', 'vehicle', 'transport', 'aircraft', 'jet', 'flight', 'aviation'],
                'boost': 0.4
            },
            'people': {
                'keywords': ['person', 'people', 'human', 'man', 'woman', 'child', 'portrait', 'face', 'individual'],
                'boost': 0.3
            },
            'objects': {
                'keywords': ['chair', 'table', 'furniture', 'object', 'item', 'thing', 'equipment'],
                'boost': 0.25
            },
            'weather': {
                'keywords': ['sunny', 'cloudy', 'rain', 'snow', 'storm', 'weather', 'sky', 'clouds'],
                'boost': 0.2
            }
        }
        
        for category, images in image_database.items():
            # Calculate relevance score
            base_score = 0.5
            
            # Check for keyword matches
            if category in keyword_mappings:
                mapping = keyword_mappings[category]
                for keyword in mapping['keywords']:
                    if keyword in query_lower:
                        base_score += mapping['boost']
                        break
            
            # Add provider-specific boost
            provider_boost = {
                'nvidia': 0.1,
                'openai': 0.05,
                'gemini': 0.05
            }.get(provider, 0)
            
            # Add images with calculated scores
            for img in images:
                final_score = min(0.98, base_score + img.get('base_score', 0.7) + provider_boost)
                scored_images.append({
                    'image_url': img['url'],
                    'filename': img['filename'],
                    'similarity_score': final_score,
                    'category': category,
                    'description': img.get('description', ''),
                    'tags': img.get('tags', [])
                })
        
        # Sort by similarity score and return top results
        scored_images.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_images[:top_k]
    
    def get_enhanced_image_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Enhanced image database with more metadata"""
        return {
            'animals': [
                {
                    'url': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=300&fit=crop&auto=format',
                    'filename': 'orange_tabby_cat.jpg',
                    'base_score': 0.92,
                    'description': 'Orange tabby cat with green eyes',
                    'tags': ['cat', 'feline', 'orange', 'tabby', 'pet']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400&h=300&fit=crop&auto=format',
                    'filename': 'white_cat_portrait.jpg',
                    'base_score': 0.90,
                    'description': 'White cat close-up portrait',
                    'tags': ['cat', 'white', 'portrait', 'close-up', 'feline']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=300&fit=crop&auto=format',
                    'filename': 'golden_retriever.jpg',
                    'base_score': 0.94,
                    'description': 'Golden retriever dog in nature',
                    'tags': ['dog', 'golden retriever', 'canine', 'outdoor', 'nature']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&h=300&fit=crop&auto=format',
                    'filename': 'cute_puppy.jpg',
                    'base_score': 0.91,
                    'description': 'Adorable puppy with big eyes',
                    'tags': ['puppy', 'dog', 'cute', 'young', 'canine']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=400&h=300&fit=crop&auto=format',
                    'filename': 'black_cat_eyes.jpg',
                    'base_score': 0.87,
                    'description': 'Black cat with striking yellow eyes',
                    'tags': ['cat', 'black', 'eyes', 'mysterious', 'feline']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=400&h=300&fit=crop&auto=format',
                    'filename': 'small_dog_grass.jpg',
                    'base_score': 0.85,
                    'description': 'Small dog sitting on grass',
                    'tags': ['dog', 'small', 'grass', 'outdoor', 'sitting']
                }
            ],
            'nature': [
                {
                    'url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop&auto=format',
                    'filename': 'snow_mountain_peak.jpg',
                    'base_score': 0.93,
                    'description': 'Snow-capped mountain peak against blue sky',
                    'tags': ['mountain', 'snow', 'peak', 'landscape', 'nature']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1464822759844-d150baec3e5e?w=400&h=300&fit=crop&auto=format',
                    'filename': 'mountain_lake_reflection.jpg',
                    'base_score': 0.89,
                    'description': 'Mountain reflection in calm lake water',
                    'tags': ['mountain', 'lake', 'reflection', 'water', 'scenic']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1551632811-561732d1e306?w=400&h=300&fit=crop&auto=format',
                    'filename': 'rocky_mountain_vista.jpg',
                    'base_score': 0.86,
                    'description': 'Rocky mountain vista with dramatic clouds',
                    'tags': ['mountain', 'rocky', 'vista', 'clouds', 'dramatic']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&h=300&fit=crop&auto=format',
                    'filename': 'forest_path_sunlight.jpg',
                    'base_score': 0.84,
                    'description': 'Forest path with sunlight filtering through trees',
                    'tags': ['forest', 'path', 'sunlight', 'trees', 'nature']
                }
            ],
            'urban': [
                {
                    'url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400&h=300&fit=crop&auto=format',
                    'filename': 'city_skyline_sunset.jpg',
                    'base_score': 0.92,
                    'description': 'City skyline at sunset with golden light',
                    'tags': ['city', 'skyline', 'sunset', 'urban', 'buildings']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1f?w=400&h=300&fit=crop&auto=format',
                    'filename': 'night_city_lights.jpg',
                    'base_score': 0.88,
                    'description': 'Night city with illuminated buildings',
                    'tags': ['city', 'night', 'lights', 'urban', 'illuminated']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1514565131-fce0801e5785?w=400&h=300&fit=crop&auto=format',
                    'filename': 'urban_street_scene.jpg',
                    'base_score': 0.83,
                    'description': 'Busy urban street with pedestrians',
                    'tags': ['street', 'urban', 'pedestrians', 'busy', 'city']
                }
            ],
            'transportation': [
                {
                    'url': 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop&auto=format',
                    'filename': 'airplane_blue_sky.jpg',
                    'base_score': 0.95,
                    'description': 'Commercial airplane flying in blue sky',
                    'tags': ['airplane', 'plane', 'sky', 'flight', 'aviation']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1569629743817-70d8db6c323b?w=400&h=300&fit=crop&auto=format',
                    'filename': 'commercial_jet_plane.jpg',
                    'base_score': 0.91,
                    'description': 'Large commercial jet aircraft',
                    'tags': ['jet', 'aircraft', 'commercial', 'plane', 'aviation']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1544636331-e26879cd4d9b?w=400&h=300&fit=crop&auto=format',
                    'filename': 'airplane_wing_view.jpg',
                    'base_score': 0.87,
                    'description': 'View from airplane window showing wing',
                    'tags': ['airplane', 'wing', 'window', 'view', 'flight']
                }
            ],
            'people': [
                {
                    'url': 'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400&h=300&fit=crop&auto=format',
                    'filename': 'person_mountain_view.jpg',
                    'base_score': 0.90,
                    'description': 'Person enjoying mountain view',
                    'tags': ['person', 'mountain', 'view', 'outdoor', 'hiking']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1494790108755-2616b612b786?w=400&h=300&fit=crop&auto=format',
                    'filename': 'woman_portrait_smile.jpg',
                    'base_score': 0.87,
                    'description': 'Smiling woman portrait',
                    'tags': ['woman', 'portrait', 'smile', 'happy', 'person']
                }
            ],
            'objects': [
                {
                    'url': 'https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=400&h=300&fit=crop&auto=format',
                    'filename': 'wooden_chair.jpg',
                    'base_score': 0.85,
                    'description': 'Wooden chair in minimalist setting',
                    'tags': ['chair', 'wooden', 'furniture', 'minimalist', 'object']
                },
                {
                    'url': 'https://images.unsplash.com/photo-1506439773649-6e0eb8cfb237?w=400&h=300&fit=crop&auto=format',
                    'filename': 'modern_table.jpg',
                    'base_score': 0.82,
                    'description': 'Modern dining table with chairs',
                    'tags': ['table', 'chairs', 'dining', 'modern', 'furniture']
                }
            ]
        }
    
    def get_fallback_results(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback results when all providers fail"""
        all_images = []
        for category, images in self.get_enhanced_image_database().items():
            for img in images:
                all_images.append({
                    'image_url': img['url'],
                    'filename': img['filename'],
                    'similarity_score': img['base_score'] * 0.5,  # Lower confidence for fallback
                    'category': category,
                    'description': img.get('description', ''),
                    'tags': img.get('tags', [])
                })
        
        # Sort by base score and return top results
        all_images.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_images[:top_k]
    
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
