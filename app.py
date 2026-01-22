from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pickle
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# User-defined Core Aspects for specialized sarcasm mapping
CORE_ASPECT_KEYWORDS = ["app", "update", "feature", "camera", "battery", "performance", "price", "design"]

# Refined Aspect Extraction: Groups compound nouns (e.g., "battery life")
def extract_aspects(sentence):
    try:
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        aspects = []
        current_aspect = []
        
        for word, pos in tagged:
            # Check for noun tags
            if pos in ["NN", "NNS", "NNP", "NNPS"]:
                current_aspect.append(word.lower())
            else:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
        if current_aspect:
            aspects.append(" ".join(current_aspect))
            
        # Filter: Remove stop words, too short words, and duplicates
        stop_words = set(stopwords.words("english"))
        filtered_aspects = []
        for a in aspects:
            if a not in stop_words and len(a) > 2:
                filtered_aspects.append(a)
        
        return list(set(filtered_aspects))
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# Load Deep Learning tools safely (handles broken torch DLLs on some Windows systems)
# NOTE: RoBERTa models require ~1GB RAM. On Render Free Tier (512MB), we skip them to avoid crash.
IS_LOW_MEM = os.environ.get("LOW_MEMORY_MODE", "false").lower() == "true"

bert_analyzer = None
sarcasm_analyzer = None

if not IS_LOW_MEM:
    try:
        from transformers import pipeline as hf_pipeline
        print("Transformers library loaded.")
        
        print("Initializing Dual RoBERTa Intelligence (Sentiment + Sarcasm)...")
        # 1. Sentiment Model
        SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        bert_analyzer = hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, tokenizer=SENTIMENT_MODEL, device=-1) # Force CPU
        
        # 2. Sarcasm Detection Model
        SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-sarcasm"
        sarcasm_analyzer = hf_pipeline("sentiment-analysis", model=SARCASM_MODEL, tokenizer=SARCASM_MODEL, device=-1) # Force CPU
        
        print("Dual RoBERTa Models ready for deployment.")
    except Exception as e:
        print(f"Deep Learning load error or memory limit: {e}")
        bert_analyzer = None
        sarcasm_analyzer = None
else:
    print("LOW_MEMORY_MODE active. Skipping RoBERTa models to prevent crash.")
    bert_analyzer = None
    sarcasm_analyzer = None

# Load our high-performance ML pipeline (Logistic Regression)
MODEL_PATH = "sentiment_pipeline.pkl"
pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = pickle.load(open(MODEL_PATH, "rb"))
        print("ML Pipeline loaded successfully.")
    except Exception as e:
        print(f"Model load error: {e}")

# Robust NLTK Data Management for Render/Cloud
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def ensure_nltk_resources():
    resources = [
        'punkt',
        'stopwords', 
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4',
        'punkt_tab',
        'averaged_perceptron_tagger_eng'
    ]
    for res in resources:
        try:
            print(f"Ensuring NLTK resource: {res}")
            nltk.download(res, download_dir=nltk_data_dir, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {res}: {e}")

ensure_nltk_resources()

# VADER for lexical sentiment analysis - ENHANCED for sarcasm
vader_analyzer = SentimentIntensityAnalyzer()

# CUSTOM VADER LEXICON: Upgrade with sarcastic and backhanded triggers
vader_analyzer.lexicon.update({
    'wow': -0.5,           # Wow is often sarcastic in negative reviews
    'great': -0.2,         # Context-dependent
    'sure': -0.8,          # "Sure..."
    'impressive': -0.5,    # "Impressive how you broke it"
    'zabardast': -1.0,     # Hinglish sarcasm
    'kamaal': -1.0,        # Hinglish sarcasm
    'wah': -1.2,           # Hinglish sarcasm
    'magic': -0.8,         # "Pure magic"
    'thanks': -1.0,        # Sarcastic thanks
    'lipstick': -1.5,      # "Lipstick on a pig"
    'expected': -1.5,      # "I expected more"
    'nothing': -1.0        # "Nothing special"
})

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    # Replace non-letters/numbers/punctuation with SPACE to avoid gluing words (like easier—highly)
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    # Keep important sentiment markers
    tokens = [w for w in tokens if w not in stop_words or w in ['not', 'no', 'never', 'but', 'however', 'very', 'highly']]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return " ".join(tokens)


# Finds descriptive words (adjectives) near the aspect to explain "Why" the sentiment is such
def get_aspect_opinion(sentence, aspect):
    try:
        tokens = word_tokenize(sentence.lower())
        tagged = nltk.pos_tag(tokens)
        aspect_tokens = aspect.lower().split()
        
        # Find index of aspect
        idx = -1
        for i in range(len(tokens) - len(aspect_tokens) + 1):
            if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
                idx = i
                break
        
        if idx == -1: return "not specified"
        
        # Search for adjectives (JJ) in a window around the aspect
        opinions = []
        start = max(0, idx - 3)
        end = min(len(tokens), idx + len(aspect_tokens) + 3)
        
        for i in range(start, end):
            word, pos = tagged[i]
            if pos.startswith("JJ") or pos in ["RB", "RBR", "RBS"]: # Adjectives or Adverbs
                opinions.append(word)
        
        return ", ".join(opinions) if opinions else "general context"
    except:
        return "general context"

def get_context_phrase(sentence, aspect):
    words = sentence.split()
    aspect_words = aspect.split()
    
    for i in range(len(words)):
        window = " ".join(words[i:i+len(aspect_words)]).lower()
        if aspect.lower() in window or window in aspect.lower():
            # Get a tighter context for specific aspect sentiment
            start = max(i - 3, 0)
            end = min(i + len(aspect_words) + 3, len(words))
            return " ".join(words[start:end])
    return sentence

def detect_sarcasm_expert(text, v_compound, subjectivity):
    text_lower = text.lower()
    cues = []
    tone = "sincere"
    
    # 1. Advanced Pattern Matchers
    sarcasm_triggers = [
        "wonderful", "masterpiece", "pure magic", "glorified", "smart yet",
        "busy right now", "love how", "so glad", "truly", "expert at",
        "nice going", "great job", "yeah right", "thanks for"
    ]

    idiom_negatives = [
        "lipstick on a pig", "drop in the ocean", "waste of", 
        "head on a wall", "paperweight", "zero", "broken", "useless"
    ]

    hinglish_sarcasm = [
        "wah bhai", "kya baat hai", "zabardast update", "gajab",
        "maza aa gaya", "great hai bhai", "kamaal", "bahut achha",
        "aisi hai ki", "itna clear", "heater ban gaya", "mubarak ho",
        "shabash", "waah", "bhala ho", "mehenga", "kya dimaag",
        "use kyun nahi", "dhanyawad", "shukriya"
    ]

    contrast_outcomes = [
        "battery lives next to a socket", "zero speed", "blur", "hang",
        "crash", "gayab", "slow", "broken", "kidney bechni", "open nahi",
        "dies in 2 hours", "dies in 1 hour", "not what i wanted",
        "security footage", "waiting", "seconds", "maza saste wala",
        "until they speak", "bright until they speak", "both be wrong",
        "unplug your life support", "ignore you", "one dollar", 
        "restart kar deta hai", "computer hang", "zero speed",
        "deleted all my data", "deleted my data"
    ]

    # 2. Logic: Detection via Cultural & Contextual Irony
    
    # Check for Ironic Triggers
    if any(p in text_lower for p in sarcasm_triggers):
        cues.append("ironic_trigger_phrase")
        tone = "sarcastic"

    # Check for Contrast (Positive claim vs Negative Outcome)
    pos_claims = ["best", "great", "amazing", "love", "impressive", "clear", "fastest", "bright"]
    neg_outcomes = ["worse", "slower", "crash", "0%", "zero", "blur", "hang", "broke", "heater", "until they speak"]
    
    if any(p in text_lower for p in pos_claims) and any(n in text_lower for n in neg_outcomes):
        cues.append("positive_claim_negative_outcome_contrast")
        tone = "ironic"

    if any(h in text_lower for h in hinglish_sarcasm):
        cues.append("hinglish_irony")
        tone = "sarcastic (Hinglish)"

    if any(c in text_lower for c in contrast_outcomes):
        cues.append("extreme_negative_outcome")
        tone = "critical"

    # 3. VADER & TextBlob Synergy
    # If VADER is negative but TextBlob says it's emotional (subjective), it's likely sarcasm
    if v_compound < -0.1 and subjectivity > 0.6:
        cues.append("high_subjectivity_negative_context")
        if tone == "sincere": tone = "emotional_critique"

    is_sarcastic = len(cues) > 0
    return is_sarcastic, tone, cues

def analyze_sentiment_hybrid(text):
    text_lower = text.lower()
    clean = preprocess_text(text)
    
    # 1. VADER Score
    v_scores = vader_analyzer.polarity_scores(text)
    v_compound = v_scores['compound']
    
    # 2. TextBlob for Subjectivity & Polarity
    # UPGRADED: Weighing subjectivity higher for irony detection
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    blob_polarity = blob.sentiment.polarity
    
    # If high subjectivity but neutral polarity, it often hides a backhanded tone
    if subjectivity > 0.6 and abs(blob_polarity) < 0.2:
        blob_polarity -= 0.3 # Penalize neutral-sounding emotional text
    
    # 3. Deep Learning BERT Model (High Precision)
    bert_score = 0
    ml_label = None
    if bert_analyzer:
        try:
            # Analyze using Transformer
            res = bert_analyzer(text)[0]
            # Cardiff RoBERTa Mapping: LABEL_0 -> Negative, LABEL_1 -> Neutral, LABEL_2 -> Positive
            label = res['label'] 
            score = res['score']
            
            if label == 'LABEL_2' or label == 'POSITIVE':
                bert_score = score
                ml_label = "Positive"
            elif label == 'LABEL_0' or label == 'NEGATIVE':
                bert_score = -score
                ml_label = "Negative"
            else: # LABEL_1 or neutral
                bert_score = 0
                ml_label = "Neutral"
            confidence = float(score)
        except:
            bert_score = 0

    # Fallback to ML Pipeline if BERT is too slow or fails
    if bert_score == 0 and pipeline:
        probs = pipeline.predict_proba([clean])[0]
        ml_res = pipeline.predict([clean])[0]
        ml_label = ["Negative", "Neutral", "Positive"][ml_res]
        bert_score = [ -1, 0, 1 ][ml_res]
        confidence = float(max(probs))
    
    # 4. Sarcasm detection (Hybrid: RoBERTa + Expert Rules)
    dl_sarcastic = False
    if sarcasm_analyzer:
        try:
            s_res = sarcasm_analyzer(text)[0]
            # RoBERTa Sarcasm Mapping: LABEL_1 is Sarcastic, LABEL_0 is Not Sarcastic
            if s_res['label'] == 'LABEL_1':
                dl_sarcastic = True
                confidence = max(confidence, s_res['score'])
        except:
            pass

    # expert rule detection
    is_sarcastic, tone, cues = detect_sarcasm_expert(text, v_compound, subjectivity)
    
    # Final Sarcasm Decision: If either the Transformer or our Expert Patterns agree
    final_is_sarcastic = dl_sarcastic or is_sarcastic
    if dl_sarcastic: cues.append("transformers_sarcasm_detection")

    # 5. Hybrid Industry-Grade Override Logic
    base_score = (0.4 * v_compound) + (0.3 * blob_polarity) + (0.3 * bert_score)
    final_sentiment = "neutral"
    
    # Overrides based on expert cues
    if final_is_sarcastic:
        final_sentiment = "Negative"
        tone = "sarcastic"
        base_score = -0.7
    elif "corporate_criticism" in cues:
        final_sentiment = "Negative"
        tone = "constructive"
        base_score = -0.6
    elif "backhanded_compliment" in cues:
        final_sentiment = "Mixed"
        tone = "backhanded"
        base_score = -0.3
    elif "low_enthusiasm" in cues:
        final_sentiment = "Negative"
        tone = "unimpressed"
        base_score = -0.4
    elif "sentiment_contrast" in cues:
        final_sentiment = "Mixed"
        tone = "constructive" if base_score > 0 else "critical"
    else:
        # Standard threshold logic
        if base_score > 0.1:
            final_sentiment = "Positive"
            tone = "sincere"
        elif base_score < -0.1:
            final_sentiment = "Negative"
            tone = "sincere"
        else:
            final_sentiment = "Neutral"
            tone = "neutral"

    # Add Emojis
    emoji_map = {
        "Positive": "Positive 😊",
        "Negative": "Negative 😡",
        "Neutral": "Neutral 😐",
        "Mixed": "Mixed 🧐"
    }
    
    display_sentiment = emoji_map.get(final_sentiment, final_sentiment)
    if final_is_sarcastic: display_sentiment += " (Sarcastic 😏)"

    if not ml_label:
        confidence = abs(base_score)
    
    # Return full data
    return {
        "sentiment": final_sentiment.lower(),
        "display_sentiment": display_sentiment,
        "tone": tone,
        "sarcasm": final_is_sarcastic,
        "hidden_sentiment": "negative" if final_is_sarcastic else final_sentiment.lower(),
        "cues": cues,
        "confidence": confidence,
        "industry_score": round(base_score, 2)
    }

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Sentiment Analysis API is running. Use /api/analyze/text for analysis.",
        "endpoints": ["/api/analyze/text", "/api/analyze/url"]
    })

@app.route("/api/analyze/text", methods=["GET", "POST"])
def analyze_text():
    try:
        if request.method == "POST":
            data = request.get_json() or {}
            sentence = data.get("text", "Great another useless feature added")
        else:
            sentence = request.args.get("text", "Great another useless feature added")
            
        if not sentence or not sentence.strip():
            return jsonify({"error": "Provide text"}), 400

        aspects = extract_aspects(sentence)
        results = {}

        # ALWAYS include Overall Sentiment Summary
        overall_res = analyze_sentiment_hybrid(sentence)
        results["Overall Summary"] = {
            "context": sentence,
            "sentiment": overall_res["display_sentiment"],
            "tone": overall_res["tone"],
            "is_sarcastic": overall_res["sarcasm"],
            "cues": overall_res["cues"],
            "confidence": f"{overall_res['confidence']:.2%}" if isinstance(overall_res["confidence"], float) else overall_res["confidence"]
        }

        # Integrate User-Requested Aspect Sarcasm Logic
        text_lower = sentence.lower()
        for asp in CORE_ASPECT_KEYWORDS:
            if asp in text_lower:
                # If the overall sentence is sarcastic, flag this specific core aspect
                if overall_res["sarcasm"]:
                    results[f"Core Aspect: {asp.capitalize()}"] = {
                        "context": f"Found '{asp}' in sarcastic context",
                        "sentiment": "Negative 😡 (Sarcastic 😏)",
                        "opinion_words": "detected via sarcastic tone",
                        "tone": "sarcastic",
                        "is_sarcastic": True,
                        "confidence": "90% (Pattern Match)"
                    }
                elif asp not in aspects: # Add it if it wasn't caught by NLTK but exists in our core list
                    aspects.append(asp)

        # Add aspect-level analysis
        if aspects:
            # Common non-aspect nouns to filter out
            noise_words = ["something", "anything", "everything", "someone", "anyone", "thing", "day", "time", "way", "lot"]
            
            for aspect in aspects:
                if aspect in noise_words: continue
                
                context = get_context_phrase(sentence, aspect)
                opinion = get_aspect_opinion(sentence, aspect)
                res = analyze_sentiment_hybrid(context)
                
                results[aspect] = {
                    "context": context,
                    "sentiment": res["display_sentiment"],
                    "opinion_words": opinion,
                    "tone": res["tone"],
                    "is_sarcastic": res["sarcasm"],
                    "confidence": f"{res['confidence']:.2%}" if isinstance(res["confidence"], float) else res["confidence"]
                }

        return jsonify({
            "status": "success",
            "sentence": sentence,
            "analysis": results
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

import requests
from bs4 import BeautifulSoup

@app.route("/api/analyze/url", methods=["GET", "POST"])
def analyze_url():
    if request.method == "POST":
        data = request.get_json() or {}
        url = data.get("url", "")
    else:
        url = request.args.get("url", "")

    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ").strip()
        text = re.sub(r"\s+", " ", text)
        
        # Take first 2000 chars for analysis to avoid timeout
        preview_text = text[:2000]
        
        aspects = extract_aspects(preview_text)
        results = {}

        # ALWAYS include Overall Summary for the page content
        overall_res = analyze_sentiment_hybrid(preview_text)
        results["Page Summary"] = {
            "context": preview_text[:200] + "...",
            "sentiment": overall_res["display_sentiment"],
            "tone": overall_res["tone"],
            "is_sarcastic": overall_res["sarcasm"],
            "cues": overall_res["cues"],
            "confidence": f"{overall_res['confidence']:.2%}" if isinstance(overall_res["confidence"], float) else overall_res["confidence"]
        }

        # Add aspect-level analysis if found
        if aspects:
            # Analyze top 8 aspects for brevity
            noise_words = ["something", "anything", "everything", "someone", "anyone", "thing", "day", "time", "way", "lot"]
            count = 0
            for aspect in aspects:
                if count >= 8: break
                if aspect in noise_words: continue
                
                context = get_context_phrase(preview_text, aspect)
                opinion = get_aspect_opinion(preview_text, aspect)
                res = analyze_sentiment_hybrid(context)
                
                results[aspect] = {
                    "context": context,
                    "sentiment": res["display_sentiment"],
                    "opinion_words": opinion,
                    "tone": res["tone"],
                    "is_sarcastic": res["sarcasm"],
                    "confidence": f"{res['confidence']:.2%}" if isinstance(res["confidence"], float) else res["confidence"]
                }
                count += 1
        
        return jsonify({
            "status": "success",
            "url": url,
            "analysis": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
