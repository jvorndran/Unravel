from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.storage import (
    clear_documents,
    get_current_document,
    save_document,
)
from unravel.utils.web_scraper import (
    crawl_url,
    generate_filename_from_url,
    scrape_url_to_markdown,
)


@st.fragment
def _render_url_scraping() -> None:
    st.caption("Enter a URL to scrape and extract content")

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/article",
        help="Enter the full URL including https://",
        label_visibility="collapsed",
    )

    use_browser = st.checkbox(
        "Enable JavaScript rendering (slower)",
        value=False,
        help="Use a headless browser (Chrome) to render JavaScript-heavy pages. This is slower but works with dynamic content.",
    )

    crawl_mode = st.checkbox(
        "Crawl multiple pages",
        value=False,
        help="Discover and scrape multiple pages from this site",
    )

    if crawl_mode:
        with st.container(border=True):
            crawl_method = st.radio(
                "Discovery Method",
                options=["Crawler", "Sitemap", "Feeds"],
                key="crawl_method",
                horizontal=True,
                help="Crawler: follows internal links. Sitemap: uses sitemap.xml. Feeds: discovers articles from RSS/Atom feeds.",
            )

            if crawl_method == "Sitemap":
                sitemap_url_override = st.text_input(
                    "Sitemap URL (optional)",
                    placeholder="e.g. https://example.com/en-us/sitemap.xml",
                    key="crawl_sitemap_url",
                    help="Override the auto-detected sitemap location. Useful when the sitemap is at a non-standard path.",
                )
            else:
                sitemap_url_override = ""

            col_pages, col_lang = st.columns(2)
            with col_pages:
                max_pages = st.number_input(
                    "Max pages",
                    min_value=1,
                    max_value=100,
                    value=10,
                    key="crawl_max_pages",
                )
            with col_lang:
                lang_filter = st.text_input(
                    "Language filter",
                    placeholder="e.g. en, fr",
                    key="crawl_lang",
                    help="Optional ISO 639-1 code",
                )

            respect_robots = st.checkbox(
                "Respect robots.txt",
                value=True,
                key="crawl_respect_robots",
                disabled=(crawl_method != "Crawler"),
                help="Follow robots.txt crawling rules (Crawler only)",
            )

            st.caption("Per-page metadata to include:")
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                meta_author = st.checkbox("Author", value=True, key="meta_author")
                meta_date = st.checkbox("Publication date", value=True, key="meta_date")
                meta_description = st.checkbox("Description", value=False, key="meta_desc")
            with meta_col2:
                meta_tags = st.checkbox("Tags / categories", value=False, key="meta_tags")
                meta_sitename = st.checkbox("Site name", value=False, key="meta_sitename")
    else:
        crawl_method = "Crawler"
        max_pages = 10
        lang_filter = ""
        sitemap_url_override = ""
        respect_robots = True
        meta_author = True
        meta_date = True
        meta_description = False
        meta_tags = False
        meta_sitename = False

    with st.expander("Advanced Extraction Options"):
        st.caption("Configure how content is extracted from the page")

        output_format = st.selectbox(
            "Output Format",
            options=["Markdown", "TXT", "CSV", "JSON", "HTML", "XML", "XML-TEI"],
            index=0,
            help="Format for the extracted content",
        )

        st.write("")

        extraction_mode = st.radio(
            "Extraction Strategy",
            options=["Balanced", "Favor Precision", "Favor Recall"],
            index=0,
            help="Balanced: Standard extraction. Precision: Less text but cleaner. Recall: More comprehensive text.",
            horizontal=True,
        )

        st.write("")

        col1, col2 = st.columns(2)
        with col1:
            include_links = st.checkbox("Include links", value=True, help="Keep hyperlinks with their targets")
            include_images = st.checkbox("Include images", value=True, help="Include image alt text and references")
            include_tables = st.checkbox("Include tables", value=True, help="Extract content from HTML tables")
        with col2:
            include_formatting = st.checkbox("Include formatting", value=False, help="Preserve text formatting (bold, italic)")
            deduplicate = st.checkbox("Deduplicate content", value=False, help="Remove duplicate content segments")

    submit_button = st.button(
        "Crawl Site" if crawl_mode else "Scrape URL",
        type="primary",
        use_container_width=True,
    )

    if submit_button:
        if not url_input or not url_input.strip():
            st.error("Please enter a URL")
            return

        try:
            format_map = {
                "Markdown": ("markdown", ".md"),
                "TXT": ("txt", ".txt"),
                "CSV": ("csv", ".csv"),
                "JSON": ("json", ".json"),
                "HTML": ("html", ".html"),
                "XML": ("xml", ".xml"),
                "XML-TEI": ("xmltei", ".xml"),
            }
            trafilatura_format, file_ext = format_map[output_format]

            favor_precision = extraction_mode == "Favor Precision"
            favor_recall = extraction_mode == "Favor Recall"

            extraction_params = {
                "output_format": trafilatura_format,
                "favor_precision": favor_precision,
                "favor_recall": favor_recall,
                "include_links": include_links,
                "include_images": include_images,
                "include_formatting": include_formatting,
                "include_tables": include_tables,
                "deduplicate": deduplicate,
            }

            if crawl_mode:
                from urllib.parse import urlparse
                crawl_domain = urlparse(url_input).netloc or url_input
                with st.status(f"Crawling {crawl_domain}...", expanded=True) as crawl_status:
                    st.write(f"Discovering pages via {crawl_method.lower()}...")
                    content_bytes, metadata = crawl_url(
                        url_input,
                        method=crawl_method.lower(),
                        max_pages=int(max_pages),
                        lang=lang_filter or None,
                        respect_robots=respect_robots,
                        sitemap_url=sitemap_url_override or None,
                        use_browser=use_browser,
                        include_author=meta_author,
                        include_date=meta_date,
                        include_description=meta_description,
                        include_tags=meta_tags,
                        include_sitename=meta_sitename,
                        **extraction_params,
                    )
                    page_count = metadata.get("page_count", 0)
                    failed_count = metadata.get("failed_count", 0)
                    if failed_count > 0:
                        crawl_status.update(
                            label=f"Crawled {page_count} pages — {failed_count} failed",
                            state="complete",
                        )
                    else:
                        crawl_status.update(
                            label=f"Crawled {page_count} pages successfully",
                            state="complete",
                        )
                file_ext = ".md"
                timestamp = datetime.now().strftime("%Y-%m-%d")
                safe_domain = metadata.get("domain", "site").replace(".", "_")
                filename = f"{safe_domain}_crawl_{timestamp}.md"
            else:
                spinner_msg = "Rendering page with JavaScript..." if use_browser else "Scraping URL..."
                with st.spinner(spinner_msg):
                    content_bytes, metadata = scrape_url_to_markdown(
                        url_input,
                        use_browser=use_browser,
                        **extraction_params,
                    )
                filename = generate_filename_from_url(url_input, metadata)
                filename = Path(filename).stem + file_ext

            doc_path = save_document(filename, content_bytes)

            doc_meta = {
                "name": filename,
                "format": file_ext.lstrip(".").upper(),
                "size_bytes": len(content_bytes),
                "path": str(doc_path),
                "source": "url",
                "source_url": url_input,
                "scraped_at": datetime.now().isoformat(),
                "domain": metadata.get("domain", ""),
                "scraping_method": metadata.get("scraping_method", ""),
            }
            if crawl_mode:
                doc_meta["crawl_method"] = metadata.get("crawl_method", "")
                doc_meta["page_count"] = metadata.get("page_count", 0)
                doc_meta["failed_count"] = metadata.get("failed_count", 0)
                doc_meta["page_results"] = metadata.get("page_results", [])

            st.session_state.document_metadata = doc_meta
            st.session_state.doc_name = filename

            for key in ["chunks", "last_embeddings_result", "search_results", "bm25_index"]:
                if key in st.session_state:
                    del st.session_state[key]

            st.rerun(scope="app")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to scrape URL: {str(e)}")


def render_upload_step() -> None:
    # Initialize session state for document metadata
    if "document_metadata" not in st.session_state:
        st.session_state.document_metadata = None

    # Check if stored document matches session state
    current_doc = get_current_document()
    if current_doc:
        doc_name, _ = current_doc
        if st.session_state.document_metadata and st.session_state.document_metadata.get("name") != doc_name:
            st.session_state.document_metadata = None
    else:
        st.session_state.document_metadata = None

    # Source selection (File Upload or URL Scraping)
    with st.container(border=True):
        st.markdown("### Add Document")

        source_mode = st.radio(
            "Source",
            options=["File Upload", "URL Scraping"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.write("")

        if source_mode == "File Upload":
            st.caption("Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, PNG, JPG")

            uploaded_file = st.file_uploader(
                "Choose a file to upload",
                type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                is_new_file = (
                    st.session_state.document_metadata is None
                    or st.session_state.document_metadata.get("name") != uploaded_file.name
                )

                if is_new_file:
                    try:
                        content = uploaded_file.read()
                        doc_path = save_document(uploaded_file.name, content)

                        file_format = Path(uploaded_file.name).suffix.upper().lstrip(".")
                        size_bytes = len(content)

                        st.session_state.document_metadata = {
                            "name": uploaded_file.name,
                            "format": file_format,
                            "size_bytes": size_bytes,
                            "path": str(doc_path),
                            "source": "file",
                        }
                        st.session_state.doc_name = uploaded_file.name

                        for key in ["chunks", "last_embeddings_result", "search_results", "bm25_index"]:
                            if key in st.session_state:
                                del st.session_state[key]

                        st.success(f"Uploaded: {uploaded_file.name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to upload {uploaded_file.name}: {str(e)}")

        else:
            _render_url_scraping()

    # Load metadata for existing document if not in session state
    if current_doc and st.session_state.document_metadata is None:
        doc_name, content = current_doc
        file_format = Path(doc_name).suffix.upper().lstrip(".")
        size_bytes = len(content)

        st.session_state.document_metadata = {
            "name": doc_name,
            "format": file_format,
            "size_bytes": size_bytes,
            "path": "",
            "source": "file",
        }
        st.session_state.doc_name = doc_name

    # Display current document
    metadata = st.session_state.document_metadata
    if metadata:
        st.write("")
        st.markdown("### Current Document")

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                file_format = metadata.get("format", "Unknown")
                size_bytes = metadata.get("size_bytes", 0)

                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

                st.markdown(f"**{metadata.get('name', 'Unknown')}**")

                if metadata.get("source") == "url":
                    source_url = metadata.get("source_url", "")
                    st.caption(f"{file_format} | {size_str}")
                    if metadata.get("crawl_method"):
                        page_count = metadata.get("page_count", 0)
                        failed_count = metadata.get("failed_count", 0)
                        if failed_count > 0:
                            st.caption(f"Crawled {page_count} pages via {metadata['crawl_method']} — {failed_count} failed")
                        else:
                            st.caption(f"Crawled {page_count} pages via {metadata['crawl_method']}")
                        page_results = metadata.get("page_results", [])
                        if page_results:
                            with st.expander("Page results", expanded=failed_count > 0):
                                for p in page_results:
                                    if p["status"] == "ok":
                                        st.caption(f"[ok] {p.get('title') or p['url']}")
                                    else:
                                        st.caption(f"[failed] {p['url']} — {p.get('reason', 'unknown error')}")
                    elif source_url:
                        display_url = source_url if len(source_url) <= 60 else source_url[:57] + "..."
                        st.caption(f"Source: {display_url}")
                else:
                    st.caption(f"{file_format} | {size_str}")

            with col2:
                if ui.button("Delete", variant="destructive", key="delete_current_doc"):
                    clear_documents()
                    st.session_state.document_metadata = None
                    st.session_state.doc_name = None
                    for key in ["chunks", "last_embeddings_result", "search_results"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    else:
        st.info("No document uploaded yet. Use the file uploader above to add a document.")
