# Deployment Guide

## Vercel Deployment Options

### Option 1: Demo Version (Current Implementation)
- ✅ **Pros**: Quick deployment, no model hosting needed
- ❌ **Cons**: Limited functionality, mock results only
- **Best for**: Demonstrations, prototypes

### Option 2: API-Based Approach (Recommended)
- ✅ **Pros**: Real ML capabilities, scalable
- ❌ **Cons**: Requires external services, API costs
- **Best for**: Production applications

### Option 3: Hybrid Approach
- ✅ **Pros**: Best of both worlds
- ❌ **Cons**: More complex setup
- **Best for**: Full-featured applications

## Recommended Architecture for Production

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vercel Web    │    │  Vector DB      │    │  ML API Service │
│   Frontend      │───▶│  (Pinecone/     │───▶│  (NVIDIA NIM/   │
│                 │    │   Weaviate)     │    │   HuggingFace)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Step-by-Step Deployment

### 1. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from project root
vercel

# Set environment variables
vercel env add NVIDIA_API_KEY
vercel env add HUGGINGFACE_API_KEY
```

### 2. Set Up Vector Database

#### Option A: Pinecone (Recommended)
```python
# Install: pip install pinecone-client
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
index = pinecone.Index("image-retrieval")

# Store embeddings
index.upsert(vectors=[
    ("img1", embedding_vector, {"filename": "cat.jpg", "url": "..."})
])

# Search
results = index.query(vector=query_embedding, top_k=10)
```

#### Option B: Weaviate
```python
# Install: pip install weaviate-client
import weaviate

client = weaviate.Client("https://your-cluster.weaviate.network")

# Store and search embeddings
```

#### Option C: Supabase with pgvector
```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table for embeddings
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    image_url TEXT,
    embedding vector(512)
);

-- Search similar embeddings
SELECT filename, image_url, 1 - (embedding <=> $1) as similarity
FROM image_embeddings
ORDER BY embedding <=> $1
LIMIT 10;
```

### 3. Configure ML API Services

#### NVIDIA NIM Setup
```bash
# Get API key from https://catalog.ngc.nvidia.com/
export NVIDIA_API_KEY="your_api_key"

# Test API
curl -X POST "https://integrate.api.nvidia.com/v1/embeddings" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["a cat sitting on a chair"],
    "model": "nvidia/nvclip",
    "encoding_format": "float"
  }'
```

#### Hugging Face Inference API
```bash
# Get API key from https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="your_api_key"

# Test API
curl -X POST "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32" \
  -H "Authorization: Bearer $HUGGINGFACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "a cat sitting on a chair"}'
```

## Alternative Hosting Solutions

### 1. Railway (Good for ML models)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### 2. Render (Free tier available)
```yaml
# render.yaml
services:
  - type: web
    name: image-retrieval
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
```

### 3. Google Cloud Run
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

### 4. AWS Lambda + API Gateway (Serverless)
```python
# Use AWS SAM or Serverless Framework
# Good for API endpoints, challenging for Streamlit
```

## Performance Optimization

### 1. Model Optimization
- Use quantized models (ONNX, TensorRT)
- Model distillation for smaller models
- Caching frequently used embeddings

### 2. Infrastructure Optimization
- CDN for image serving
- Redis for caching search results
- Load balancing for high traffic

### 3. Cost Optimization
- Batch processing for embeddings
- Smart caching strategies
- API rate limiting

## Environment Variables

Create a `.env` file for local development:
```bash
NVIDIA_API_KEY=your_nvidia_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-west1-gcp
DATABASE_URL=your_database_url
```

## Monitoring and Analytics

### 1. Vercel Analytics
```javascript
// Add to your frontend
import { Analytics } from '@vercel/analytics/react';

export default function App() {
  return (
    <>
      <YourApp />
      <Analytics />
    </>
  );
}
```

### 2. Error Tracking
```python
# Add Sentry for error tracking
import sentry_sdk
sentry_sdk.init(dsn="your-sentry-dsn")
```

## Security Considerations

1. **API Key Management**: Use environment variables
2. **Rate Limiting**: Implement request throttling
3. **Input Validation**: Sanitize user inputs
4. **CORS Configuration**: Proper cross-origin settings
5. **Authentication**: Add user authentication if needed

## Scaling Considerations

1. **Database Sharding**: For large image collections
2. **Microservices**: Separate encoding and search services
3. **Caching Layers**: Redis/Memcached for performance
4. **CDN Integration**: For global image delivery

## Troubleshooting

### Common Issues
1. **Cold Start Delays**: Use serverless warming
2. **Memory Limits**: Optimize model loading
3. **Timeout Errors**: Implement async processing
4. **CORS Issues**: Check headers configuration

### Debug Commands
```bash
# Check Vercel logs
vercel logs

# Test API endpoints
curl -X POST https://your-app.vercel.app/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'
```
