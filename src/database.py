import pandas as pd
import streamlit as st
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

def get_client():
    """Initializes and returns the native BigQuery client using central secrets."""
    try:
        project_id = st.secrets["GCP_PROJECT_ID"]
        return bigquery.Client(project=project_id)
    except Exception as e:
        if not st.secrets.get("USE_MOCK_DB", True):
            st.error(f"❌ BigQuery Client Initialization Failed: {e}")
        return None

def init_mock_sessions():
    """Local session mock fallback variables."""
    if "mock_bronze" not in st.session_state:
        st.session_state["mock_bronze"] = pd.DataFrame()
    if "mock_silver" not in st.session_state:
        st.session_state["mock_silver"] = pd.DataFrame()

def prepare_df_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    """Converts complex Python lists/arrays into strings to ensure BigQuery compatibility."""
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
        try:
            df_ready = prepare_df_for_bq(df)
            project_id = st.secrets["GCP_PROJECT_ID"]
            dataset_name = st.secrets["GCP_DATASET_NAME"]
            table_id = f"{project_id}.{dataset_name}.bronze_raw_posts"
            
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            job = client.load_table_from_dataframe(df_ready, table_id, job_config=job_config)
            job.result() # Wait for upload transaction to complete
            
            st.toast("🔥 Raw data written to BigQuery Bronze Layer!")
        except Exception as e:
            st.error(f"⚠️ Failed saving to BigQuery (Bronze): {e}")

def get_unprocessed_bronze() -> pd.DataFrame:
    """
    Defensively queries the Bronze layer. Checks if Silver table exists first 
    to completely eliminate the 'Table Not Found' cold-start crash.
    """
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    client = get_client()
    
    if use_mock or client is None:
        bronze = st.session_state["mock_bronze"]
        silver = st.session_state["mock_silver"]
        if bronze.empty or silver.empty: return bronze
        return bronze[~bronze['raw_post_id'].isin(silver['post_id'])]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        dataset_name = st.secrets["GCP_DATASET_NAME"]
        
        silver_table_ref = f"{project_id}.{dataset_name}.silver_enriched_posts"
        bronze_table_ref = f"{project_id}.{dataset_name}.bronze_raw_posts"
        
        # Guardrail: Check if the Silver table exists before writing the subquery
        silver_table_exists = True
        try:
            client.get_table(silver_table_ref)
        except NotFound:
            silver_table_exists = False
            
        if not silver_table_exists:
            # Cold-start case: Silver table is missing, meaning ALL raw entries are unprocessed
            query = f"""
                SELECT raw_post_id, search_keyword, platform, author, posted_at, raw_text 
                FROM `{bronze_table_ref}`
            """
        else:
            # Standard operational case: Filter out already processed unique IDs
            query = f"""
                SELECT raw_post_id, search_keyword, platform, author, posted_at, raw_text 
                FROM `{bronze_table_ref}`
                WHERE raw_post_id NOT IN (
                    SELECT DISTINCT post_id FROM `{silver_table_ref}`
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
        try:
            df_ready = prepare_df_for_bq(df)
            project_id = st.secrets["GCP_PROJECT_ID"]
            dataset_name = st.secrets["GCP_DATASET_NAME"]
            table_id = f"{project_id}.{dataset_name}.silver_enriched_posts"
            
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
            job = client.load_table_from_dataframe(df_ready, table_id, job_config=job_config)
            job.result()
            
            st.toast("💎 Enriched data written to BigQuery Silver Layer!")
        except Exception as e:
            st.error(f"⚠️ Failed saving to BigQuery (Silver): {e}")

def get_gold_post_explorer() -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    if use_mock or get_client() is None:
        return st.session_state["mock_silver"]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        dataset_name = st.secrets["GCP_DATASET_NAME"]
        table_ref = f"{project_id}.{dataset_name}.silver_enriched_posts"
        
        try:
            return get_client().query(f"SELECT * FROM `{table_ref}`").to_dataframe()
        except Exception:
            return pd.DataFrame()

def get_gold_campaign_analytics(keyword: str) -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    if use_mock or get_client() is None:
        df = st.session_state["mock_silver"]
        return df[df['campaign_keyword'] == keyword] if not df.empty else df
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        dataset_name = st.secrets["GCP_DATASET_NAME"]
        table_ref = f"{project_id}.{dataset_name}.silver_enriched_posts"
        
        try:
            query = f"SELECT * FROM `{table_ref}` WHERE campaign_keyword = '{keyword}'"
            return get_client().query(query).to_dataframe()
        except Exception:
            return pd.DataFrame()

def get_gold_cross_campaign() -> pd.DataFrame:
    init_mock_sessions()
    use_mock = st.secrets.get("USE_MOCK_DB", True)
    
    if use_mock or get_client() is None:
        df = st.session_state["mock_silver"]
    else:
        project_id = st.secrets["GCP_PROJECT_ID"]
        dataset_name = st.secrets["GCP_DATASET_NAME"]
        table_ref = f"{project_id}.{dataset_name}.silver_enriched_posts"
        try:
            df = get_client().query(f"SELECT * FROM `{table_ref}`").to_dataframe()
        except Exception:
            df = pd.DataFrame()
        
    if df.empty:
        return pd.DataFrame(columns=["campaign_keyword", "total_posts", "Positive", "Negative", "Neutral"])
    
    summary = df.groupby('campaign_keyword').agg(total_posts=('post_id', 'count')).reset_index()
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in df['sentiment_label'].values:
            sent_df = df[df['sentiment_label'] == sentiment].groupby('campaign_keyword').size().to_frame(sentiment).reset_index()
            summary = summary.merge(sent_df, on='campaign_keyword', how='left').fillna(0)
        else:
            summary[sentiment] = 0.0
        
    return summary