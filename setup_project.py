import os
from pathlib import Path

# Define project structure
files = {}

# 1. DEPENDENCIES & ENVIRONMENT CONFIGURATION
files["requirements.txt"] = """streamlit==1.32.0
pandas==2.2.1
pandas-gbq==0.22.0
google-cloud-bigquery==3.18.0
apify-client==1.6.1
scikit-learn==1.4.1.post1
"""

files[".gitignore"] = """__pycache__/
*.pyc
.streamlit/secrets.toml
*.json
models/*.pkl
"""

files["Dockerfile"] = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

files[".streamlit/secrets.toml"] = """# Local secrets store. Replace with your actual GCP Project ID when ready.
APIFY_TOKEN = "apify_api_Bq4oUTfVdymvgOOwEPERVb71f9ZcbJ2Q2Km0"
GCP_PROJECT_ID = "your-gcp-project-id"
USE_MOCK_DB = true # Set to false to use true Google BigQuery
"""

# 2. CRAWLER MODULE
files["src/crawler.py"] = """import pandas as pd
import streamlit as st
from apify_client import ApifyClient

def fetch_instagram_data(keyword: str, max_results: int = 20) -> list:
    \"\"\"
    Connects to the Apify API using the provided token to pull live Instagram data.
    \"\"\"
    # Fetch token from Streamlit secrets (or fallback to hardcoded value)
    try:
        apify_token = st.secrets["APIFY_TOKEN"]
    except Exception:
        apify_token = "apify_api_Bq4oUTfVdymvgOOwEPERVb71f9ZcbJ2Q2Km0"
        
    client = ApifyClient(apify_token)
    
    run_input = {
        "search": keyword,
        "searchType": "hashtag", 
        "resultsLimit": max_results
    }
    
    try:
        # Call the official standard Instagram scraper Actor
        run = client.actor("apify/instagram-scraper").call(run_input=run_input)
        raw_items = client.dataset(run["defaultDatasetId"]).iterate_items()
        
        formatted_data = []
        for item in raw_items:
            post_timestamp = item.get("timestamp")
            formatted_data.append({
                "id": str(item.get("id", "")),
                "text": item.get("caption", ""),  # Instagram captions store post text
                "author_username": item.get("ownerUsername", "unknown_user"),
                "created_at": pd.to_datetime(post_timestamp) if post_timestamp else pd.Timestamp.now()
            })
        return formatted_data
    except Exception as e:
        st.error(f"Apify crawling failed: {e}. Falling back to sample data for evaluation.")
        # Junior-friendly fallback data so execution doesn't break if API limits are hit
        return [
            {"id": "mock_1", "text": f"Loving the brand new launch of {keyword}! Essential upgrade. #awesome", "author_username": "tech_guru", "created_at": pd.Timestamp.now()},
            {"id": "mock_2", "text": f"Terrible customer service regarding my {keyword} order. Extremely slow.", "author_username": "angry_customer", "created_at": pd.Timestamp.now()},
            {"id": "mock_3", "text": f"Just saw the latest campaign for {keyword}. It looks okay, nothing special.", "author_username": "neutral_observer", "created_at": pd.Timestamp.now()}
        ]
"""

# 3. AI & ML PROCESSING MODULE
files["src/ai_processor.py"] = """import re
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

MODEL_PATH = "models/svm_sentiment_v1.pkl"

def initialize_svm_model():
    \"\"\" Trains a quick baseline SVM model if it doesn't exist yet \"\"\"
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(MODEL_PATH):
        # Balanced baseline training data for the marketing domain
        texts = [
            "excellent amazing wonderful love perfect", 
            "bad terrible horrible worst broken hate slow", 
            "okay standard regular normal check update"
        ]
        labels = ["Positive", "Negative", "Neutral"]
        pipeline = make_pipeline(TfidfVectorizer(), SVC(probability=True, kernel='linear'))
        pipeline.fit(texts, labels)
        joblib.dump(pipeline, MODEL_PATH)

def extract_metadata_and_sentiment(df_bronze: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    AI Processing Layer: Extracts hashtags, topics, and executes SVM sentiment analysis.
    \"\"\"
    if df_bronze.empty:
        return pd.DataFrame()
        
    initialize_svm_model()
    model = joblib.load(MODEL_PATH)
    
    silver_rows = []
    for _, row in df_bronze.iterrows():
        text = str(row['raw_text'])
        
        # Metadata Processing 1
        clean_text = re.sub(r"http[^ ]+", "", text).strip()
        hashtags = re.findall(r"#([a-zA-Z0-9_]+)", text)
        
        # Simple rule-based topic labeling (Junior readable & highly performant)
        if any(w in clean_text.lower() for w in ['service', 'customer', 'delivery', 'slow', 'support']):
            topic = "Customer Service"
        elif any(w in clean_text.lower() for w in ['campaign', 'ad', 'commercial', 'billboard']):
            topic = "Marketing Campaign"
        else:
            topic = "Product Evaluation"
            
        # Sentiment Processing 2 (SVM Model)
        pred_label = model.predict([clean_text])[0]
        prob = model.predict_proba([clean_text]).max(axis=1)[0]
        
        silver_rows.append({
            "post_id": row["raw_post_id"],
            "campaign_keyword": row["search_keyword"],
            "platform": row["platform"],
            "author": row["author"],
            "posted_at": row["posted_at"],
            "clean_text": clean_text,
            "hashtags": hashtags,
            "language": "en",
            "topic_label": topic,
            "sentiment_label": pred_label,
            "sentiment_confidence": float(prob),
            "model_version": "svm_v1.0"
        })
        
    return pd.DataFrame(silver_rows)
"""

# 4. DATABASE LAYER (BIGQUERY & MOCK CAPABILITIES)
files["src/database.py"] = """import pandas as pd
import streamlit as st
from google.cloud import bigquery

def get_client():
    try:
        return bigquery.Client(project=st.secrets["GCP_PROJECT_ID"])
    except Exception:
        return None

def init_mock_sessions():
    \"\"\" Initializes local session states to store data if BQ credentials aren't set up yet \"\"\"
    if "mock_bronze" not in st.session_state:
        st.session_state["mock_bronze"] = pd.DataFrame()
    if "mock_silver" not in st.session_state:
        st.session_state["mock_silver"] = pd.DataFrame()

def save_to_bronze(df: pd.DataFrame):
    init_mock_sessions()
    if st.secrets.get("USE_MOCK_DB", True) or get_client() is None:
        st.session_state["mock_bronze"] = pd.concat([st.session_state["mock_bronze"], df], ignore_index=True).drop_duplicates(subset=['raw_post_id'])
    else:
        df.to_gbq(f"marketing_platform.bronze_raw_posts", project_id=st.secrets["GCP_PROJECT_ID"], if_exists="append")

def get_unprocessed_bronze() -> pd.DataFrame:
    init_mock_sessions()
    if st.secrets.get("USE_MOCK_DB", True) or get_client() is None:
        bronze = st.session_state["mock_bronze"]
        silver = st.session_state["mock_silver"]
        if bronze.empty: return bronze
        if silver.empty: return bronze
        return bronze[~bronze['raw_post_id'].isin(silver['post_id'])]
    else:
        query = \"\"\"
            SELECT raw_post_id, search_keyword, platform, author, posted_at, raw_text 
            FROM `marketing_platform.bronze_raw_posts`
            WHERE raw_post_id NOT IN (SELECT post_id FROM `marketing_platform.silver_enriched_posts`)
        \"\"\"
        return get_client().query(query).to_dataframe()

def save_to_silver(df: pd.DataFrame):
    init_mock_sessions()
    if st.secrets.get("USE_MOCK_DB", True) or get_client() is None:
        st.session_state["mock_silver"] = pd.concat([st.session_state["mock_silver"], df], ignore_index=True).drop_duplicates(subset=['post_id'])
    else:
        df.to_gbq(f"marketing_platform.silver_enriched_posts", project_id=st.secrets["GCP_PROJECT_ID"], if_exists="append")

# GOLD LAYER INTERFACES (SCREENS 1, 2, 3)
def get_gold_post_explorer() -> pd.DataFrame:
    init_mock_sessions()
    if st.secrets.get("USE_MOCK_DB", True) or get_client() is None:
        return st.session_state["mock_silver"]
    else:
        return get_client().query("SELECT * FROM `marketing_platform.gold_vw_post_explorer`").to_dataframe()

def get_gold_campaign_analytics(keyword: str) -> pd.DataFrame:
    init_mock_sessions()
    if st.secrets.get("USE_MOCK_DB", True) or get_client() is None:
        df = st.session_state["mock_silver"]
        return df[df['campaign_keyword'] == keyword] if not df.empty else df
    else:
        query = f"SELECT * FROM `marketing_platform.gold_campaign_analytics` WHERE campaign_keyword = '{keyword}'"
        return get_client().query(query).to_dataframe()

def get_gold_cross_campaign() -> pd.DataFrame:
    init_mock_sessions()
    df = st.session_state["mock_silver"]
    if df.empty:
        return pd.DataFrame(columns=["campaign_keyword", "total_posts", "Positive", "Negative", "Neutral"])
    
    # Direct Pandas implementation of the Gold cross-campaign conditional counting view
    summary = df.groupby('campaign_keyword').agg(
        total_posts=('post_id', 'count')
    ).reset_index()
    
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sent_df = df[df['sentiment_label'] == sentiment].groupby('campaign_keyword').size().to_frame(sentiment).reset_index()
        summary = summary.merge(sent_df, on='campaign_keyword', how='left').fillna(0)
        
    return summary
"""

# 5. STREAMLIT INTERACTIVE MULTI-SCREEN APPLICATION
files["app.py"] = """import streamlit as st
import pandas as pd
from src.crawler import fetch_instagram_data
from src.ai_processor import extract_metadata_and_sentiment
from src.database import save_to_bronze, get_unprocessed_bronze, save_to_silver, get_gold_post_explorer, get_gold_campaign_analytics, get_gold_cross_campaign

st.set_page_sheet_layout = "wide"
st.sidebar.title("🎯 Intel Platform")
page = st.sidebar.radio("Navigation Menu", ["Data Ingestion Engine", "Screen 1: Post Explorer", "Screen 2: Campaign Analytics", "Screen 3: Cross-Campaign Comparison"])

# PIPELINE EXECUTION SUBMISSION SYSTEM
if page == "Data Ingestion Engine":
    st.header("🚀 Data Crawler & Processing Pipeline Launcher")
    st.markdown("Submit a pipeline crawling task. Raw data falls into **Bronze**, passes automated **Silver** NLP models, and materializes directly into your operational dashboard views.")
    
    with st.form("pipeline_form"):
        keyword = st.text_input("Enter Search Keyword or Hashtag (e.g., nike, tech)", value="nike")
        date_range = st.date_input("Target Date Scope Range", [])
        platform = st.selectbox("Source Platform Target", ["Instagram"])
        submitted = st.form_submit_button("Execute Pipeline Journey")
        
        if submitted:
            if not keyword:
                st.error("Please enter a valid keyword.")
            else:
                with st.spinner("Step 1: Contacting Apify Platform and saving to Bronze Database..."):
                    raw_data = fetch_instagram_data(keyword)
                    
                    df_bronze = pd.DataFrame(raw_data)
                    df_bronze['search_keyword'] = keyword
                    df_bronze['platform'] = platform
                    df_bronze['raw_post_id'] = df_bronze['id']
                    df_bronze['raw_text'] = df_bronze['text']
                    df_bronze['raw_json'] = df_bronze.to_json(orient='records', lines=True).splitlines()[:len(df_bronze)]
                    df_bronze['posted_at'] = pd.to_datetime(df_bronze['created_at'])
                    
                    save_to_bronze(df_bronze)
                    st.success("Bronze Layer populated successfully!")
                    
                with st.spinner("Step 2: Pulling fresh raw entries & processing with SVM ML models..."):
                    unprocessed = get_unprocessed_bronze()
                    if not unprocessed.empty:
                        df_silver = extract_metadata_and_sentiment(unprocessed)
                        save_to_silver(df_silver)
                        st.success(f"Enriched Silver records written! Model Audit Tracking tagged as: svm_v1.0")
                    else:
                        st.info("No raw pending entries require processing updates.")

# SCREEN 1: POST EXPLORER
elif page == "Screen 1: Post Explorer":
    st.header("🔍 Granular Post Explorer")
    df = get_gold_post_explorer()
    
    if df.empty:
        st.warning("Data layer is currently dry. Run an Ingestion session first.")
    else:
        # UI Dynamic Filters
        keywords = st.multiselect("Filter by Campaign Key", options=df['campaign_keyword'].unique())
        sentiments = st.multiselect("Filter by Sentiment Label", options=df['sentiment_label'].unique())
        
        filtered_df = df.copy()
        if keywords:
            filtered_df = filtered_df[filtered_df['campaign_keyword'].isin(keywords)]
        if sentiments:
            filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiments)]
            
        st.dataframe(filtered_df[['post_id', 'campaign_keyword', 'author', 'clean_text', 'topic_label', 'sentiment_label', 'sentiment_confidence']], use_container_width=True)

# SCREEN 2: CAMPAIGN ANALYTICS
elif page == "Screen 2: Campaign Analytics":
    st.header("📈 Deep Campaign Analytics Dashboard")
    df_all = get_gold_post_explorer()
    
    if df_all.empty:
        st.warning("No metrics data present. Run pipeline collection engines.")
    else:
        target_keyword = st.selectbox("Select Active Campaign Scope", options=df_all['campaign_keyword'].unique())
        df = get_gold_campaign_analytics(target_keyword)
        
        # Metrics Cards Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Volumes Tracked", len(df))
        c2.metric("Dominant Sentiment Status", df['sentiment_label'].mode()[0] if not df.empty else "None")
        c3.metric("Average AI Processing Confidence", f"{round(df['sentiment_confidence'].mean() * 100, 1)}%" if not df.empty else "0%")
        
        st.subheader("Distribution Splitting Analytics")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**Topic Breakdown Allocation**")
            st.bar_chart(df['topic_label'].value_counts())
            
        with col_right:
            st.markdown("**Sentiment Split Percentages**")
            st.bar_chart(df['sentiment_label'].value_counts())

# SCREEN 3: CROSS-CAMPAIGN COMPARISON
elif page == "Screen 3: Cross-Campaign Comparison":
    st.header("⚔️ Executive Cross-Campaign Matrix")
    df = get_gold_cross_campaign()
    
    if df.empty or (len(df) == 1 and df['campaign_keyword'].isna().all()):
        st.warning("Insufficient distinct campaign historical entries to render a cross comparison.")
    else:
        st.subheader("High Level Strategic Volume Scale Summary")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Comparative Volume & Sentiment Matrix Visuals")
        st.bar_chart(df.set_index('campaign_keyword')[['Positive', 'Negative', 'Neutral']])
"""

# 6. INFORMATIVE DOCUMENTATION
files["README.md"] = """# Marketing Intelligence Platform

A unified, low-overhead intelligence application tracking text content, automated NLP tokenizing, and SVM sentiment monitoring.

## Local Deployment Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt"""