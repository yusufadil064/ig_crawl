import streamlit as st
import pandas as pd
from src.crawler import fetch_instagram_data
from src.ai_processor import extract_metadata_and_sentiment
from src.database import (
    save_to_bronze, 
    get_unprocessed_bronze, 
    save_to_silver, 
    get_gold_post_explorer, 
    get_gold_campaign_analytics, 
    get_gold_cross_campaign
)

st.set_page_config(layout="wide", page_title="Marketing Intelligence Platform")

st.sidebar.title("🎯 Platform Navigation")
page = st.sidebar.radio(
    "Select Interface", 
    [
        "🚀 Ingestion & Processing Engine", 
        "🔍 Screen 1: Post Explorer", 
        "📈 Screen 2: Campaign Analytics", 
        "⚔️ Screen 3: Cross-Campaign Comparison"
    ]
)

if page == "🚀 Ingestion & Processing Engine":
    st.header("Data Crawler & Processing Pipeline Launcher")
    st.markdown("Trigger data acquisition from social media APIs, route it into raw processing layers, and execute predictive analytics models.")
    
    with st.form("pipeline_form"):
        keyword = st.text_input("Enter Search Keyword or Hashtag (e.g., nike, tech)", value="nike")
        platform = st.selectbox("Source Platform Target", ["Instagram"])
        date_range = st.date_input("Crawl Date Range (Optional)", [])
        submitted = st.form_submit_button("Launch Data Pipeline")
        
        if submitted:
            if not keyword:
                st.error("Error: Please specify a search keyword before proceeding.")
            else:
                # ----------------------------------------------------
                # STEP 1: BRONZE LAYER CRAWLING
                # ----------------------------------------------------
                st.subheader("📥 Step 1: Raw Data Ingestion (Bronze Layer)")
                
                with st.spinner("Extracting posts from source API..."):
                    raw_data = fetch_instagram_data(keyword)
                    
                    df_bronze = pd.DataFrame(raw_data)
                    df_bronze['search_keyword'] = keyword
                    df_bronze['platform'] = platform
                    df_bronze['raw_post_id'] = df_bronze['id']
                    df_bronze['raw_text'] = df_bronze['text']
                    df_bronze['author'] = df_bronze['author_username']  
                    df_bronze['raw_json'] = df_bronze.to_json(orient='records', lines=True).splitlines()[:len(df_bronze)]
                    df_bronze['posted_at'] = pd.to_datetime(df_bronze['created_at'])
                    
                    save_to_bronze(df_bronze)
                
                st.success("🟢 STEP 1 COMPLETE: Raw payload stored.")
                st.info("**Architectural Note:** Records are written exactly as received to prevent historical data loss. No transformations have been applied yet.")
                
                st.markdown("#### Preview: Crawled Raw Results Table (Bronze)")
                st.dataframe(df_bronze[['raw_post_id', 'search_keyword', 'author', 'raw_text', 'posted_at']], width="stretch")
                st.divider()

                # ----------------------------------------------------
                # STEP 2: SILVER LAYER AI ENRICHMENT
                # ----------------------------------------------------
                st.subheader("🤖 Step 2: AI Processing & Enrichment (Silver Layer)")
                
                with st.spinner("Analyzing text metadata and running SVM classification pipeline..."):
                    # Safe check executes flawlessly even if Silver table doesn't exist yet
                    unprocessed = get_unprocessed_bronze()
                    
                    if not unprocessed.empty:
                        df_silver = extract_metadata_and_sentiment(unprocessed)
                        save_to_silver(df_silver)
                        
                        st.success("🟢 STEP 2 COMPLETE: Natural Language Processing & Sentiment Mapping finished.")
                        st.info("**Architectural Note:** System applied cleaned text rules, auto-extracted tags, and utilized a linear SVM classifier. Audit model tracking code: `svm_id_v1.0`.")
                        
                        st.markdown("#### Preview: AI Enriched Analytical Results Table (Silver)")
                        st.dataframe(df_silver[['post_id', 'clean_text', 'topic_label', 'sentiment_label', 'sentiment_confidence', 'model_version']], width="stretch")
                    else:
                        st.warning("⚠️ Indicator Status: No pending raw posts required calculation passes.")

# SCREEN 1, 2, 3 UIs remain exactly as before...
elif page == "🔍 Screen 1: Post Explorer":
    st.header("Granular Post Explorer")
    df = get_gold_post_explorer()
    if df.empty:
        st.warning("Data layer is currently dry. Please execute an Ingestion process first.")
    else:
        col1, col2 = st.columns(2)
        with col1: keywords = st.multiselect("Filter by Campaign Key", options=df['campaign_keyword'].unique())
        with col2: sentiments = st.multiselect("Filter by Sentiment Label", options=df['sentiment_label'].unique())
        filtered_df = df.copy()
        if keywords: filtered_df = filtered_df[filtered_df['campaign_keyword'].isin(keywords)]
        if sentiments: filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiments)]
        st.dataframe(filtered_df[['post_id', 'campaign_keyword', 'author', 'clean_text', 'topic_label', 'sentiment_label', 'sentiment_confidence']], width="stretch")

elif page == "📈 Screen 2: Campaign Analytics":
    st.header("Deep Campaign Analytics Dashboard")
    df_all = get_gold_post_explorer()
    if df_all.empty:
        st.warning("Data layer is currently dry. Please execute an Ingestion process first.")
    else:
        target_keyword = st.selectbox("Select Active Campaign Scope", options=df_all['campaign_keyword'].unique())
        df = get_gold_campaign_analytics(target_keyword)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Volumes Tracked", len(df))
        c2.metric("Dominant Sentiment Status", df['sentiment_label'].mode()[0] if not df.empty else "None")
        c3.metric("Average AI Processing Confidence", f"{round(df['sentiment_confidence'].mean() * 100, 1)}%" if not df.empty else "0%")
        st.subheader("Distribution Analytics")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Topic Allocation Distribution**")
            st.bar_chart(df['topic_label'].value_counts())
        with col_right:
            st.markdown("**Sentiment Split Balances**")
            st.bar_chart(df['sentiment_label'].value_counts())

elif page == "⚔️ Screen 3: Cross-Campaign Comparison":
    st.header("Executive Cross-Campaign Matrix")
    df = get_gold_cross_campaign()
    if df.empty or (len(df) == 1 and df['campaign_keyword'].isna().all()):
        st.warning("Insufficient unique historical profiles to compare. Run multiple crawls first.")
    else:
        st.subheader("Strategic Volume Scaling Analytics")
        st.dataframe(df, width="stretch")
        st.subheader("Comparative Volume & Metric Splits")
        st.bar_chart(df.set_index('campaign_keyword')[['Positive', 'Negative', 'Neutral']])