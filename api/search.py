import json
import os

def handler(event, context):
    """Vercel serverless function handler"""
    
    # Handle CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': ''
        }
    
    try:
        # Parse request body
        body = event.get('body', '{}')
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
        
        query = data.get('query', '')
        top_k = data.get('top_k', 6)
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Query is required'})
            }
        
        # Get demo results
        results = get_demo_results(query, top_k)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'results': results,
                'provider_used': 'demo',
                'query': query
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': f'Search failed: {str(e)}'})
        }

def get_demo_results(query, top_k):
    """Get demo search results"""
    query_lower = query.lower()
    
    images = [
        {
            'image_url': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=300&fit=crop',
            'filename': 'orange_cat.jpg',
            'similarity_score': 0.95,
            'keywords': ['cat', 'feline', 'orange', 'pet']
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=300&fit=crop',
            'filename': 'golden_dog.jpg',
            'similarity_score': 0.92,
            'keywords': ['dog', 'golden', 'canine', 'pet']
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=300&fit=crop',
            'filename': 'airplane.jpg',
            'similarity_score': 0.93,
            'keywords': ['airplane', 'plane', 'aircraft', 'sky']
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400&h=300&fit=crop',
            'filename': 'mountain.jpg',
            'similarity_score': 0.89,
            'keywords': ['mountain', 'landscape', 'nature', 'peak']
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400&h=300&fit=crop',
            'filename': 'city.jpg',
            'similarity_score': 0.87,
            'keywords': ['city', 'urban', 'buildings', 'skyline']
        },
        {
            'image_url': 'https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?w=400&h=300&fit=crop',
            'filename': 'person.jpg',
            'similarity_score': 0.85,
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