import pandas as pd
import streamlit as st
from google.cloud import bigquery

def get_client():
    """Membuka koneksi ke BigQuery menggunakan Project ID dari secrets."""
    try:
        project_id = st.secrets["GCP_PROJECT_ID"]
        return bigquery.Client(project=project_id)
    except Exception as e:
        if not st.secrets.get("USE_MOCK_DB", True):
            st.error(f"❌ Gagal inisialisasi BigQuery Client: {e}")
        return None

def init_mock_sessions():
    """Inisialisasi session state lokal untuk fallback mode."""
    if "mock_bronze" not in st.session_state:
        st.session_state["mock_bronze"] = pd.DataFrame()
    if "mock_silver" not in st.session_state:
        st.session_state["mock_silver"] = pd.DataFrame()

def prepare_df_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    """Mengonversi semua kolom bertipe list/array menjadi string JSON agar aman."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].apply(lambda x: isinstance(x, list)).any():
            df_clean[col] = df_clean[col].astype(str)
    return df_clean

def save_to_bronze(df: pd.DataFrame):
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    client = get_client()
    
    if use_mock or client is None:
        st.session_state["mock_bronze"] = pd.concat(
            [st.session_state["mock_bronze"], df], ignore_index=True
        ).drop_duplicates(subset=['raw_post_id'])
    else:
        # === FIX: NATIVE GOOGLE CLOUD BIGQUERY DATA LOAD ===
        try:
            df_ready = prepare_df_for_bq(df)
            table_id = f"{st.secrets['GCP_PROJECT_ID']}.social_media.bronze_raw_posts"
            
            # Configure the ingestion behavior
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND" # Equivalent to if_exists="append"
            )
            
            # Execute native streaming load
            job = client.load_table_from_dataframe(df_ready, table_id, job_config=job_config)
            job.result()  # Wait for the API upload upload operation to complete successfully
            
            st.toast("🔥 Data berhasil disimpan ke BigQuery Bronze Layer!")
        except Exception as e:
            st.error(f"⚠️ Gagal menyimpan ke BigQuery (Bronze): {e}")

def get_unprocessed_bronze() -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    client = get_client()
    
    if use_mock or client is None:
        bronze = st.session_state["mock_bronze"]
        silver = st.session_state["mock_silver"]
        if bronze.empty: return bronze
        if silver.empty: return bronze
        return bronze[~bronze['raw_post_id'].isin(silver['post_id'])]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        query = f"""
            SELECT raw_post_id, search_keyword, platform, author, posted_at, raw_text 
            FROM `{project_id}.social_media.bronze_raw_posts`
            WHERE raw_post_id NOT IN (
                SELECT DISTINCT post_id FROM `{project_id}.social_media.silver_enriched_posts`
            )
        """
        return client.query(query).to_dataframe()

def save_to_silver(df: pd.DataFrame):
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    client = get_client()
    
    if use_mock or client is None:
        st.session_state["mock_silver"] = pd.concat(
            [st.session_state["mock_silver"], df], ignore_index=True
        ).drop_duplicates(subset=['post_id'])
    else:
        # === FIX: NATIVE GOOGLE CLOUD BIGQUERY DATA LOAD ===
        try:
            df_ready = prepare_df_for_bq(df)
            table_id = f"{st.secrets['GCP_PROJECT_ID']}.social_media.silver_enriched_posts"
            
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND"
            )
            
            job = client.load_table_from_dataframe(df_ready, table_id, job_config=job_config)
            job.result()  # Wait for the API upload operation to complete successfully
            
            st.toast("💎 Data Analitik sukses disimpan ke BigQuery Silver Layer!")
        except Exception as e:
            st.error(f"⚠️ Gagal menyimpan ke BigQuery (Silver): {e}")

def get_gold_post_explorer() -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    if use_mock or get_client() is None:
        return st.session_state["mock_silver"]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        return get_client().query(f"SELECT * FROM `{project_id}.social_media.silver_enriched_posts`").to_dataframe()

def get_gold_campaign_analytics(keyword: str) -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    if use_mock or get_client() is None:
        df = st.session_state["mock_silver"]
        return df[df['campaign_keyword'] == keyword] if not df.empty else df
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        query = f"SELECT * FROM `{project_id}.social_media.silver_enriched_posts` WHERE campaign_keyword = '{keyword}'"
        return get_client().query(query).to_dataframe()

def get_gold_cross_campaign() -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    
    if use_mock or get_client() is None:
        df = st.session_state["mock_silver"]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        df = get_client().query(f"SELECT * FROM `{project_id}.social_media.silver_enriched_posts`").to_dataframe()
        
    if df.empty:
        return pd.DataFrame(columns=["campaign_keyword", "total_posts", "Positive", "Negative", "Neutral"])
    
    summary = df.groupby('campaign_keyword').agg(total_posts=('post_id', 'count')).reset_index()
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        sent_df = df[df['sentiment_label'] == sentiment].groupby('campaign_keyword').size().to_frame(sentiment).reset_index()
        summary = summary.merge(sent_df, on='campaign_keyword', how='left').fillna(0)
        
    return summary