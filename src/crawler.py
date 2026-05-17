import pandas as pd
import streamlit as st
from apify_client import ApifyClient

def fetch_instagram_data(keyword: str, max_results: int = 100) -> list:
    """
    Connects to Apify using the specialized apify/instagram-hashtag-scraper actor.
    Patched to support correct array-based schema formatting.
    """
    try:
        apify_token = st.secrets["APIFY_TOKEN"]
    except Exception:
        apify_token = "apify_api_Bq4oUTfVdymvgOOwEPERVb71f9ZcbJ2Q2Km0"
        
    client = ApifyClient(apify_token)
    
    # Strip out any accidentally included '#' symbols from the user input
    clean_hashtag = keyword.replace("#", "").strip()
    
    # FIX 1: Format run_input to match the strict apify/instagram-hashtag-scraper schema
    run_input = {
        "hashtags": [clean_hashtag],     # Must be an array list of strings
        "resultsLimit": int(max_results)  # Explicitly cast to integer 
    }
    
    try:
        # FIX 2: Point directly to the requested hashtag scraper actor string
        run = client.actor("apify/instagram-hashtag-scraper").call(run_input=run_input)
        raw_items = client.dataset(run["defaultDatasetId"]).iterate_items()
        
        formatted_data = []
        for item in raw_items:
            post_timestamp = item.get("timestamp")
            
            # FIX 3: Bulletproof parsing wrapper to handle schema casing variations
            username = item.get("ownerUsername") or item.get("owner_username") or "unknown_user"
            caption_text = item.get("caption") or item.get("text") or ""
            
            formatted_data.append({
                "id": str(item.get("id", "")),
                "text": str(caption_text),  
                "author_username": str(username),
                "created_at": pd.to_datetime(post_timestamp) if post_timestamp else pd.Timestamp.now()
            })
            
        return formatted_data
        
    except Exception as e:
        st.error(f"Apify live API crawling failed: {e}. Launching local engine backup records.")
        # UI Safe local data arrays so your application layers don't freeze up
        return [
            {"id": "mock_hash_1", "text": f"Testing the hashtag updates for #{clean_hashtag}! Works great.", "author_username": "qa_tester", "created_at": pd.Timestamp.now()},
            {"id": "mock_hash_2", "text": f"The performance metrics on this tracking loop are incredibly smooth. #{clean_hashtag}", "author_username": "dev_lead", "created_at": pd.Timestamp.now()}
        ]