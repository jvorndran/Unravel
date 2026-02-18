"""UI Widget Keys Registry.

Centralized registry of all widget keys used in the Unravel UI.
This ensures consistency between implementation and tests.
"""


class WidgetKeys:
    """Registry of widget keys for Streamlit components."""

    # =========================================================================
    # Upload Step
    # =========================================================================
    UPLOAD_CRAWL_METHOD = "crawl_method"
    UPLOAD_CRAWL_SITEMAP_URL = "crawl_sitemap_url"
    UPLOAD_CRAWL_MAX_PAGES = "crawl_max_pages"
    UPLOAD_CRAWL_LANG = "crawl_lang"
    UPLOAD_CRAWL_RESPECT_ROBOTS = "crawl_respect_robots"
    UPLOAD_META_AUTHOR = "meta_author"
    UPLOAD_META_DATE = "meta_date"
    UPLOAD_META_DESC = "meta_desc"
    UPLOAD_META_TAGS = "meta_tags"
    UPLOAD_META_SITENAME = "meta_sitename"
    UPLOAD_DELETE_BTN = "delete_current_doc"

    # =========================================================================
    # Chunks Step
    # =========================================================================
    CHUNKS_APPLY_BTN = "apply_chunking_config_btn"
    CHUNKS_PARSE_BTN = "parse_document_btn"
    CHUNKS_GOTO_UPLOAD = "goto_upload_chunks"
    CHUNKS_VIEW_TAB = "chunks_view_tab"

    # =========================================================================
    # Embeddings Step
    # =========================================================================
    EMBEDDINGS_RESTART_QDRANT_BTN = "restart_qdrant_header_btn"
    EMBEDDINGS_GOTO_UPLOAD = "goto_upload_embeddings"
    EMBEDDINGS_GOTO_CHUNKS = "goto_chunks"
    EMBEDDINGS_QUERY_INPUT = "query_input"

    # =========================================================================
    # Query Step
    # =========================================================================
    QUERY_INPUT = "query_input"
    QUERY_OPEN_CONFIG_FOLDER_BTN = "open_config_folder_btn"
    QUERY_GOTO_EMBEDDINGS = "goto_embeddings"
    QUERY_ASK_BUTTON = "ask_button"
    QUERY_TOP_K_SLIDER = "top_k_slider"
    QUERY_THRESHOLD_SLIDER_PREFIX = "threshold_slider_"
    QUERY_EXPANSION_ENABLED = "query_expansion_enabled"
    QUERY_EXPANSION_COUNT = "query_expansion_count"
    QUERY_REWRITE_PROMPT = "query_rewrite_prompt"
    QUERY_SYSTEM_PROMPT = "query_system_prompt"
    QUERY_API_ENDPOINT_TOGGLE = "api_endpoint_toggle"

    # =========================================================================
    # Export Step
    # =========================================================================
    EXPORT_GOTO_CHUNKS = "goto_chunks_export"

    # =========================================================================
    # Sidebar - RAG Configuration
    # =========================================================================
    SIDEBAR_DOC_SELECTOR = "sidebar_doc_selector"
    SIDEBAR_RETRIEVAL_STRATEGY = "sidebar_retrieval_strategy"
    SIDEBAR_DENSE_WEIGHT = "sidebar_dense_weight"
    SIDEBAR_FUSION_METHOD = "sidebar_fusion_method"
    SIDEBAR_ENABLE_RERANKING = "sidebar_enable_reranking"
    SIDEBAR_RERANK_LIBRARY = "sidebar_rerank_library"
    SIDEBAR_RERANK_MODEL = "sidebar_rerank_model"
    SIDEBAR_RERANK_TOP_N = "sidebar_rerank_top_n"
    SIDEBAR_EMBEDDING_MODEL = "sidebar_embedding_model"
    SIDEBAR_SAVE_RAG_CONFIG_BTN = "save_rag_config_btn"
    SIDEBAR_CLEAR_SESSION_BTN = "clear_session_btn"

    # =========================================================================
    # Sidebar - LLM Configuration
    # =========================================================================
    SIDEBAR_PROVIDER = "sidebar_provider"
    SIDEBAR_MODEL_INPUT = "sidebar_model_input"
    SIDEBAR_MODEL_SELECT = "sidebar_model_select"
    SIDEBAR_TEMPERATURE = "sidebar_temperature"
    SIDEBAR_BASE_URL = "sidebar_base_url"
    SIDEBAR_SAVE_CONFIG_BTN = "save_config_btn"


# Convenience functions for dynamic keys
def get_threshold_slider_key(strategy: str, fusion_method: str | None = None) -> str:
    """Generate threshold slider key based on retrieval strategy.

    Args:
        strategy: Retrieval strategy name
        fusion_method: Optional fusion method for hybrid retrieval

    Returns:
        Complete widget key for threshold slider
    """
    if fusion_method:
        return f"{WidgetKeys.QUERY_THRESHOLD_SLIDER_PREFIX}{strategy}_{fusion_method}"
    return f"{WidgetKeys.QUERY_THRESHOLD_SLIDER_PREFIX}{strategy}"
