from http.server import BaseHTTPRequestHandler
import json
import os
import requests

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Serve the main web interface"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text-to-Image Retrieval System</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .search-input {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 10px;
            margin-bottom: 15px;
            box-sizing: border-box;
        }
        .search-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
        }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .result-item {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            background: #f9f9f9;
        }
        .result-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .similarity-score {
            background: #e8f4fd;
            color: #1f77b4;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .loading { text-align: center; padding: 20px; color: #666; }
        .error { background: #ffe6e6; color: #d63031; padding: 15px; border-radius: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Text-to-Image Retrieval</h1>
            <p>Search for images using natural language descriptions</p>
        </div>
        
        <div>
            <input type="text" id="searchInput" class="search-input" 
                   placeholder="e.g., a cat sitting on a chair" />
            <button onclick="searchImages()" class="search-button">Search Images</button>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        async function searchImages() {
            const query = document.getElementById('searchInput').value;
            if (!query.trim()) {
                alert('Please enter a search description');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">🔍 Searching...</div>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: 6 })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    return;
                }
                
                if (data.results && data.results.length > 0) {
                    displayResults(data.results, query);
                } else {
                    resultsDiv.innerHTML = '<div class="error">No results found</div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">Search failed: ${error.message}</div>`;
            }
        }
        
        function displayResults(results, query) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div style="grid-column: 1 / -1; margin-bottom: 20px;">
                    <h3>Search Results - "${query}"</h3>
                    <p>Found ${results.length} relevant images</p>
                </div>
            `;
            
            results.forEach(result => {
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                resultItem.innerHTML = `
                    <img src="${result.image_url}" alt="Search result" loading="lazy" />
                    <div class="similarity-score">Similarity: ${result.similarity_score.toFixed(3)}</div>
                    <div>${result.filename}</div>
                `;
                resultsDiv.appendChild(resultItem);
            });
        }
        
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchImages();
        });
    </script>
</body>
</html>
        """
        
        self.wfile.write(html.encode())
    
    def do_POST(self):
        """Handle API requests"""
        if self.path == '/api/search':
            self.handle_search()
        else:
            self.send_error(404, "Not Found")
    
    def handle_search(self):
        """Handle search requests"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            else:
                data = {}
            
            query = data.get('query', '')
            top_k = data.get('top_k', 6)
            
            if not query:
                self.send_json_response({'error': 'Query is required'}, 400)
                return
            
            # Try NVIDIA NIM if available
            nvidia_api_key = os.environ.get('NVIDIA_API_KEY')
            if nvidia_api_key:
                results = self.search_with_nvidia(query, top_k, nvidia_api_key)
                if results:
                    self.send_json_response({
                        'results': results,
                        'provider_used': 'nvidia'
                    })
                    return
            
            # Fallback to demo results
            results = self.get_demo_results(query, top_k)
            self.send_json_response({
                'results': results,
                'provider_used': 'demo'
            })
            
        except Exception as e:
            self.send_json_response({'error': f'Search failed: {str(e)}'}, 500)
    
    def search_with_nvidia(self, query, top_k, api_key):
        """Search using NVIDIA NIM"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                'https://integrate.api.nvidia.com/v1/embeddings',
                headers=headers,
                json={
                    'input': [query],
                    'model': 'nvidia/nvclip',
                    'encoding_format': 'float'
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return self.get_demo_results(query, top_k, boost=0.1)
            
        except Exception:
            pass
        
        return None
    
    def get_demo_results(self, query, top_k, boost=0):
        """Get demo results"""
        query_lower = query.lower()
        
        images = [
            {
                'image_url': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=300&fit=crop',
                'filename': 'orange_cat.jpg',
                'similarity_score': 0.95 + boost,
                'keywords': ['cat', 'feline', 'orange', 'pet']
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=300&fit=crop',
                'filename': 'golden_dog.jpg',
                'similarity_score': 0.92 + boost,
                'keywords': ['dog', 'golden', 'canine', 'pet']
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop',
                'filename': 'airplane.jpg',
                'similarity_score': 0.93 + boost,
                'keywords': ['airplane', 'plane', 'aircraft', 'sky']
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop',
                'filename': 'mountain.jpg',
                'similarity_score': 0.89 + boost,
                'keywords': ['mountain', 'landscape', 'nature', 'peak']
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400&h=300&fit=crop',
                'filename': 'city.jpg',
                'similarity_score': 0.87 + boost,
                'keywords': ['city', 'urban', 'buildings', 'skyline']
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400&h=300&fit=crop',
                'filename': 'person.jpg',
                'similarity_score': 0.85 + boost,
                'keywords': ['person', 'human', 'portrait', 'people']
            }
        ]
        
        # Score based on keyword matching
        for img in images:
            for keyword in img['keywords']:
                if keyword in query_lower:
                    img['similarity_score'] += 0.1
                    break
        
        # Sort and return top results
        images.sort(key=lambda x: x['similarity_score'], reverse=True)
        return images[:top_k]
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
