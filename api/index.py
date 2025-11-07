from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.encoders import NVIDIANIMEncoder
except ImportError:
    # Fallback if modules are not available
    NVIDIANIMEncoder = None

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - serve the web interface"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html_content = """
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
                .header p {
                    color: #666;
                    font-size: 1.1rem;
                }
                .search-section {
                    margin-bottom: 30px;
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
                    transition: transform 0.2s;
                }
                .search-button:hover {
                    transform: translateY(-2px);
                }
                .upload-section {
                    border: 2px dashed #ddd;
                    border-radius: 10px;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 30px;
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
                .loading {
                    text-align: center;
                    padding: 20px;
                    color: #666;
                }
                .error {
                    background: #ffe6e6;
                    color: #d63031;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
                .info {
                    background: #e8f4fd;
                    color: #0984e3;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Text-to-Image Retrieval</h1>
                    <p>Search for images using natural language descriptions</p>
                </div>
                
                <div class="info">
                    <strong>Note:</strong> This is a demo version running on Vercel. 
                    For full functionality with local models and file uploads, 
                    please run the system locally using the instructions in the README.
                </div>
                
                <div class="search-section">
                    <h3>🔤 Text Search</h3>
                    <input type="text" id="searchInput" class="search-input" 
                           placeholder="e.g., a cat sitting on a chair" />
                    <div style="margin-bottom: 15px;">
                        <label for="providerSelect" style="margin-right: 10px;">AI Provider:</label>
                        <select id="providerSelect" style="padding: 8px; border-radius: 5px; border: 1px solid #ddd;">
                            <option value="auto">Auto (Best Available)</option>
                            <option value="nvidia">NVIDIA NIM</option>
                            <option value="openai">OpenAI</option>
                            <option value="gemini">Google Gemini</option>
                        </select>
                    </div>
                    <button onclick="searchImages()" class="search-button">Search Images</button>
                </div>
                
                <div class="upload-section">
                    <h3>📷 Image-to-Image Search</h3>
                    <p>Upload an image to find similar images</p>
                    <input type="file" id="imageUpload" accept="image/*" onchange="uploadImage()" />
                </div>
                
                <div id="results" class="results"></div>
            </div>
            
            <script>
                async function searchImages() {
                    const query = document.getElementById('searchInput').value;
                    const provider = document.getElementById('providerSelect').value;
                    
                    if (!query.trim()) {
                        alert('Please enter a search description');
                        return;
                    }
                    
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = '<div class="loading">🔍 Searching with ' + provider + ' provider...</div>';
                    
                    try {
                        const response = await fetch('/api/multi_provider_search', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ 
                                query: query, 
                                top_k: 6,
                                provider: provider
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                            return;
                        }
                        
                        if (data.results && data.results.length > 0) {
                            displayResults(data.results, query, data.provider_used);
                        } else {
                            resultsDiv.innerHTML = '<div class="error">No results found</div>';
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Search failed: ${error.message}</div>`;
                    }
                }
                
                function displayResults(results, query, provider_used) {
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = `
                        <div style="grid-column: 1 / -1; margin-bottom: 20px;">
                            <h3>Search Results - "${query}"</h3>
                            <p>Found ${results.length} relevant images using <strong>${provider_used}</strong> provider</p>
                        </div>
                    `;
                    
                    results.forEach(result => {
                        const resultItem = document.createElement('div');
                        resultItem.className = 'result-item';
                        resultItem.innerHTML = `
                            <img src="${result.image_url}" alt="Search result" loading="lazy" />
                            <div class="similarity-score">Similarity: ${result.similarity_score.toFixed(3)}</div>
                            <div style="font-weight: bold; margin: 5px 0;">${result.filename}</div>
                            ${result.description ? `<div style="font-size: 0.9em; color: #666;">${result.description}</div>` : ''}
                            ${result.tags ? `<div style="font-size: 0.8em; color: #888; margin-top: 5px;">${result.tags.join(', ')}</div>` : ''}
                        `;
                        resultsDiv.appendChild(resultItem);
                    });
                }
                
                async function uploadImage() {
                    const fileInput = document.getElementById('imageUpload');
                    const file = fileInput.files[0];
                    
                    if (!file) return;
                    
                    const resultsDiv = document.getElementById('results');
                    resultsDiv.innerHTML = '<div class="loading">🔍 Searching for similar images...</div>';
                    
                    try {
                        const formData = new FormData();
                        formData.append('image', file);
                        
                        const response = await fetch('/api/search-by-image', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                            return;
                        }
                        
                        if (data.results && data.results.length > 0) {
                            displayResults(data.results, 'Similar images');
                        } else {
                            resultsDiv.innerHTML = '<div class="error">No similar images found</div>';
                        }
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="error">Image search failed: ${error.message}</div>`;
                    }
                }
                
                // Allow Enter key to trigger search
                document.getElementById('searchInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchImages();
                    }
                });
            </script>
        </body>
        </html>
        """
        
        self.wfile.write(html_content.encode())
    
    def do_POST(self):
        """Handle POST requests for API endpoints"""
        if self.path == '/api/search':
            self.handle_text_search()
        elif self.path == '/api/search-by-image':
            self.handle_image_search()
        else:
            self.send_error(404, "Not Found")
    
    def handle_text_search(self):
        """Handle text-based image search"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            
            if not query:
                self.send_json_response({'error': 'Query is required'}, 400)
                return
            
            # For demo purposes, return mock results
            # In a real deployment, you would use the actual retrieval system
            mock_results = self.get_mock_search_results(query, top_k)
            
            self.send_json_response({'results': mock_results})
            
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def handle_image_search(self):
        """Handle image-based search"""
        try:
            # For demo purposes, return mock results
            mock_results = self.get_mock_search_results("similar images", 5)
            self.send_json_response({'results': mock_results})
            
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def get_mock_search_results(self, query, top_k):
        """Generate mock search results for demo"""
        # Sample images from Unsplash for demo
        sample_images = [
            {
                'image_url': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=300',
                'filename': 'cat_1.jpg',
                'similarity_score': 0.95
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=300',
                'filename': 'cat_2.jpg',
                'similarity_score': 0.89
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=300',
                'filename': 'dog_1.jpg',
                'similarity_score': 0.82
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=300',
                'filename': 'dog_2.jpg',
                'similarity_score': 0.78
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=300',
                'filename': 'mountain_1.jpg',
                'similarity_score': 0.71
            },
            {
                'image_url': 'https://images.unsplash.com/photo-1464822759844-d150baec3e5e?w=300',
                'filename': 'mountain_2.jpg',
                'similarity_score': 0.68
            }
        ]
        
        # Return top_k results
        return sample_images[:min(top_k, len(sample_images))]
    
    def send_json_response(self, data, status_code=200):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data)
        self.wfile.write(response.encode())
