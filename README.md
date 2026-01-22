# ToneIQ Sentiment Backend Engine 🚀

An advanced, hybrid Sentiment Analysis API built with Python, Flask, and Machine Learning. 

## Features
- **Hybrid Analysis**: Combines Lexical (VADER), Statistical (TextBlob), and Machine Learning (Logistic Regression).
- **Sarcasm Detection**: Custom expert rules for detecting sarcasm, including Hinglish support.
- **Aspect-Based Analysis**: Automatically extracts key aspects (e.g., Battery, Performance) and provides sentiment for each.
- **URL Analysis**: Analyze sentiment directly from web page content.
- **CORS Enabled**: Ready to connect with modern frontend frameworks (React, Vue, etc.).

## Endpoints
- `POST /api/analyze/text`: Primary analysis endpoint.
- `GET /api/analyze/text?text=...`: Quick test via browser.
- `POST /api/analyze/url`: URL-based content analysis.

## Deployment
Optimized for **Render** and **Vercel** with local memory management support.
