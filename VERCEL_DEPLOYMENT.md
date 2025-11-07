# Vercel Deployment Guide

## 🚀 Quick Deploy to Vercel

This guide will help you deploy the Text-to-Image Retrieval System to Vercel with multi-provider AI support (NVIDIA NIM, OpenAI, Google Gemini).

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **AI Provider API Keys** (at least one):
   - NVIDIA NIM API Key from [NVIDIA NGC](https://catalog.ngc.nvidia.com/)
   - OpenAI API Key from [OpenAI Platform](https://platform.openai.com/)
   - Google Gemini API Key from [Google AI Studio](https://makersuite.google.com/)

## Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

## Step 2: Deploy to Vercel

```bash
# Clone or navigate to your project directory
cd image-retrieval

# Login to Vercel
vercel login

# Deploy (follow the prompts)
vercel

# Or deploy directly
vercel --prod
```

## Step 3: Set Environment Variables

After deployment, set your API keys as environment variables:

```bash
# NVIDIA NIM API Key (recommended)
vercel env add NVIDIA_API_KEY

# OpenAI API Key (optional)
vercel env add OPENAI_API_KEY

# Google Gemini API Key (optional)
vercel env add GEMINI_API_KEY
```

Or set them via the Vercel Dashboard:
1. Go to your project dashboard
2. Navigate to Settings → Environment Variables
3. Add the following variables:

| Variable Name | Value | Environment |
|---------------|-------|-------------|
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key | Production, Preview, Development |
| `OPENAI_API_KEY` | Your OpenAI API key | Production, Preview, Development |
| `GEMINI_API_KEY` | Your Gemini API key | Production, Preview, Development |

## Step 4: Redeploy

After setting environment variables, redeploy:

```bash
vercel --prod
```

## API Endpoints

Your deployed app will have these endpoints:

- **Main App**: `https://your-app.vercel.app/`
- **Multi-Provider Search**: `https://your-app.vercel.app/api/multi_provider_search`
- **NVIDIA-Only Search**: `https://your-app.vercel.app/api/nvidia_search`

## Features

### 🤖 Multi-Provider AI Support
- **Auto Mode**: Automatically tries providers in order (NVIDIA → OpenAI → Gemini)
- **Manual Selection**: Choose specific provider
- **Fallback System**: Falls back to curated results if all providers fail

### 🔍 Search Capabilities
- Natural language text search
- Semantic similarity matching
- Enhanced metadata and descriptions
- Provider-specific optimizations

### 🎨 Modern Web Interface
- Responsive design
- Provider selection dropdown
- Real-time search results
- Image lazy loading
- Error handling

## Getting API Keys

### NVIDIA NIM API Key
1. Visit [NVIDIA NGC](https://catalog.ngc.nvidia.com/)
2. Sign up/login with your NVIDIA account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key (starts with `nvapi-`)

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login
3. Go to API Keys section
4. Create new secret key
5. Copy the key (starts with `sk-`)

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with Google account
3. Create new API key
4. Copy the key

## Testing Your Deployment

### Test the Web Interface
1. Visit your Vercel app URL
2. Try different search queries:
   - "a cat sitting on a chair"
   - "mountain landscape"
   - "airplane in the sky"
3. Test different AI providers
4. Check the results and provider used

### Test API Directly
```bash
# Test multi-provider search
curl -X POST "https://your-app.vercel.app/api/multi_provider_search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a cat sitting on a chair",
    "top_k": 5,
    "provider": "auto"
  }'
```

## Troubleshooting

### Common Issues

1. **"API key not configured" Error**
   - Ensure environment variables are set correctly
   - Redeploy after setting variables
   - Check variable names match exactly

2. **Timeout Errors**
   - Vercel has 30s timeout limit
   - API calls are optimized for 20s timeout
   - Fallback system activates on timeout

3. **Provider Not Working**
   - Check API key validity
   - Verify API quotas/limits
   - System will fallback to other providers

### Debug Commands
```bash
# Check deployment logs
vercel logs

# Check environment variables
vercel env ls

# Test specific provider
curl -X POST "https://your-app.vercel.app/api/multi_provider_search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "provider": "nvidia"}'
```

## Customization

### Adding More Images
Edit `api/multi_provider_search.py` and expand the `get_enhanced_image_database()` function:

```python
def get_enhanced_image_database(self):
    return {
        'your_category': [
            {
                'url': 'https://your-image-url.com/image.jpg',
                'filename': 'your_image.jpg',
                'base_score': 0.9,
                'description': 'Your image description',
                'tags': ['tag1', 'tag2', 'tag3']
            }
        ]
    }
```

### Modifying Search Logic
Update the `search_with_embedding()` function to customize:
- Keyword mappings
- Scoring algorithms
- Category weights
- Provider-specific boosts

## Performance Optimization

### Image Optimization
- All images use Unsplash's optimization parameters
- Lazy loading implemented
- Responsive image sizing

### API Optimization
- 20s timeout for API calls
- Automatic fallback system
- Efficient error handling
- Minimal response payloads

## Cost Considerations

### API Costs (Approximate)
- **NVIDIA NIM**: Free tier available, then pay-per-use
- **OpenAI**: ~$0.0001 per 1K tokens for embeddings
- **Gemini**: Free tier available, then pay-per-use

### Vercel Costs
- **Hobby Plan**: Free (10GB bandwidth, 100GB-hours)
- **Pro Plan**: $20/month (100GB bandwidth, 1000GB-hours)

## Security Best Practices

1. **Environment Variables**: Never commit API keys to code
2. **Rate Limiting**: Implement if needed for high traffic
3. **Input Validation**: Already implemented in the API
4. **CORS**: Properly configured for web access

## Next Steps

1. **Vector Database Integration**: For production, consider:
   - [Pinecone](https://www.pinecone.io/) - Managed vector database
   - [Weaviate](https://weaviate.io/) - Open source vector database
   - [Supabase](https://supabase.com/) - PostgreSQL with pgvector

2. **Image Upload**: Add image upload functionality
3. **User Authentication**: Add user accounts and saved searches
4. **Analytics**: Track usage and popular searches

## Support

If you encounter issues:
1. Check the [Vercel Documentation](https://vercel.com/docs)
2. Review API provider documentation
3. Check the project's GitHub issues
4. Test locally first: `vercel dev`

---

🎉 **Congratulations!** Your Text-to-Image Retrieval System is now live on Vercel with multi-provider AI support!
