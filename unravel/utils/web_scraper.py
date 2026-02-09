"""Web scraping utilities for converting URLs to markdown."""

import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import trafilatura
from trafilatura import feeds, sitemaps, spider

try:
    from curl_cffi import requests
    _IMPERSONATE = "chrome120"
except ImportError:
    import requests
    _IMPERSONATE = None
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

_BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}


def _fetch(url: str, timeout: int = 30) -> str:
    """Fetch URL HTML using curl_cffi (browser TLS) when available, else requests."""
    if _IMPERSONATE:
        resp = requests.get(url, impersonate=_IMPERSONATE, timeout=timeout, allow_redirects=True)
    else:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text


def validate_url(url: str) -> str:
    """Validate and normalize a URL.

    Args:
        url: The URL string to validate

    Returns:
        Normalized URL with scheme

    Raises:
        ValueError: If URL is invalid or uses unsupported protocol
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    url = url.strip()

    # Add https:// if no scheme provided
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError("Invalid URL format")

    # Check for valid scheme
    if parsed.scheme not in ("http", "https"):
        if parsed.scheme == "file":
            raise ValueError("Local file paths are not supported. Please use the file upload feature instead.")
        raise ValueError(f"Unsupported protocol: {parsed.scheme}. Only HTTP and HTTPS are supported.")

    # Check for valid domain
    if not parsed.netloc:
        raise ValueError("Invalid URL format: missing domain")

    return url


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL.

    Args:
        url: The URL to extract domain from

    Returns:
        Clean domain name (e.g., "example.com" from "https://example.com/path")
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    # Remove www. prefix if present
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _sanitize_filename_component(text: str) -> str:
    """Sanitize text for use in filename.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for filenames
    """
    # Replace path separators and special characters
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    # Replace multiple underscores with single
    text = re.sub(r"_+", "_", text)
    # Remove leading/trailing underscores and spaces
    text = text.strip("_ ")
    # Limit length
    if len(text) > 100:
        text = text[:100]
    return text or "untitled"


def generate_filename_from_url(url: str, metadata: dict) -> str:
    """Generate sanitized filename from URL and metadata.

    Args:
        url: The source URL
        metadata: Metadata dict containing optional 'title' key

    Returns:
        Sanitized filename in format: {domain}_{title}_{timestamp}.md
    """
    domain = extract_domain(url)
    domain = _sanitize_filename_component(domain.replace(".", "_"))

    title = metadata.get("title", "")
    if title:
        title = _sanitize_filename_component(title.lower().replace(" ", "_"))
        title = f"_{title}"
    else:
        title = ""

    timestamp = datetime.now().strftime("%Y-%m-%d")

    filename = f"{domain}{title}_{timestamp}.md"

    # Ensure total length is reasonable
    if len(filename) > 150:
        # Truncate title part if too long
        max_title_len = 150 - len(domain) - len(timestamp) - 5  # 5 for underscores and .md
        if title:
            title = title[:max_title_len]
        filename = f"{domain}{title}_{timestamp}.md"

    return filename


def scrape_with_trafilatura(
    url: str,
    output_format: str = "markdown",
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_links: bool = True,
    include_images: bool = True,
    include_formatting: bool = False,
    include_tables: bool = True,
    deduplicate: bool = False,
) -> tuple[bytes, dict]:
    """Scrape URL using Trafilatura (fast, static content).

    Note: Uses requests for downloading with proper headers, then trafilatura for extraction.
    This is the recommended approach since trafilatura.fetch_url() doesn't accept custom headers.

    Args:
        url: The URL to scrape
        output_format: Output format: markdown, txt, csv, json, html, xml, xmltei
        favor_precision: Prefer less text but cleaner extraction
        favor_recall: Prefer more text even when uncertain
        include_links: Keep hyperlinks with their targets
        include_images: Include image references and alt text
        include_formatting: Preserve text formatting (bold, italic)
        include_tables: Include table content
        deduplicate: Remove duplicate content segments

    Returns:
        Tuple of (content_bytes, metadata_dict)

    Raises:
        ValueError: If scraping fails or content cannot be extracted
    """
    try:
        downloaded = _fetch(url, timeout=30)

        if not downloaded:
            raise ValueError(
                "Failed to download content from URL. "
                "The site may be blocking requests or require JavaScript rendering."
            )

        # Extract content in requested format
        content = trafilatura.extract(
            downloaded,
            output_format=output_format,
            include_comments=False,
            include_tables=include_tables,
            include_links=include_links,
            include_images=include_images,
            include_formatting=include_formatting,
            favor_precision=favor_precision,
            favor_recall=favor_recall,
            deduplicate=deduplicate,
        )

        if not content or len(content.strip()) < 100:
            raise ValueError(
                "No content extracted or content is too short. "
                "The page may be empty or require JavaScript rendering. "
                "Try enabling 'JavaScript rendering' option."
            )

        # Extract metadata
        metadata_dict = trafilatura.extract_metadata(downloaded)

        metadata = {
            "title": metadata_dict.title if metadata_dict and metadata_dict.title else "",
            "author": metadata_dict.author if metadata_dict and metadata_dict.author else "",
            "domain": extract_domain(url),
            "scraping_method": "trafilatura",
            "output_format": output_format,
        }

        return content.encode("utf-8"), metadata

    except ValueError:
        raise
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "Unknown"
        if status_code == 403:
            raise ValueError(f"Access forbidden (403). The site may be blocking automated requests.")
        elif status_code == 404:
            raise ValueError(f"Page not found (404). Please check the URL.")
        elif status_code == 429:
            raise ValueError(f"Too many requests (429). The site is rate limiting.")
        else:
            raise ValueError(f"HTTP error {status_code} while accessing {extract_domain(url)}")
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out while accessing {extract_domain(url)}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Failed to connect to {extract_domain(url)}. Please check the URL and your internet connection.")
    except requests.exceptions.SSLError:
        raise ValueError(f"SSL certificate error while accessing {extract_domain(url)}")
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            raise ValueError(f"Request timed out while accessing {extract_domain(url)}")
        elif "connection" in error_msg or "network" in error_msg:
            raise ValueError(f"Failed to connect to {extract_domain(url)}. Please check the URL and your internet connection.")
        elif "ssl" in error_msg or "certificate" in error_msg:
            raise ValueError(f"SSL certificate error while accessing {extract_domain(url)}")
        else:
            raise ValueError(f"Failed to scrape URL: {str(e)}")


def _create_chrome_driver() -> webdriver.Chrome:
    """Create a configured headless Chrome driver."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(60)
    return driver


def _wait_for_js(driver: webdriver.Chrome) -> None:
    """Wait for the page to fully render including JavaScript."""
    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(2)


def _selenium_fetch(url: str) -> tuple[str, str]:
    """Launch headless Chrome, render the page, return (html, title)."""
    driver = None
    try:
        driver = _create_chrome_driver()
        driver.get(url)
        _wait_for_js(driver)

        return driver.page_source, driver.title
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def _scrape_selenium_with_html(url: str, **extraction_params) -> tuple[bytes, str]:
    """Render page with Selenium and return (content_bytes, raw_html).

    Returns raw HTML alongside content so callers can extract metadata
    without a second network request.
    """
    html, title = _selenium_fetch(url)
    body_text = re.sub(r"<[^>]+>", " ", html)

    if not body_text or len(body_text.strip()) < 100:
        raise ValueError("No content extracted or content is too short. The page may be empty or protected.")

    content = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        **extraction_params,
    )
    if not content or len(content.strip()) < 100:
        content = body_text.strip()

    return content.encode("utf-8"), html


def scrape_with_selenium(
    url: str,
    output_format: str = "markdown",
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_links: bool = True,
    include_images: bool = True,
    include_formatting: bool = False,
    include_tables: bool = True,
    deduplicate: bool = False,
) -> tuple[bytes, dict]:
    """Scrape URL using Selenium (handles JavaScript).

    Uses Selenium with Chrome in headless mode to render JavaScript-heavy pages.

    Args:
        url: The URL to scrape
        favor_precision: Prefer less text but cleaner extraction
        favor_recall: Prefer more text even when uncertain
        include_links: Keep hyperlinks with their targets
        include_images: Include image references and alt text
        include_formatting: Preserve text formatting (bold, italic)
        include_tables: Include table content
        deduplicate: Remove duplicate content segments

    Returns:
        Tuple of (markdown_content_bytes, metadata_dict)

    Raises:
        ValueError: If scraping fails or content cannot be extracted
    """
    try:
        html, title = _selenium_fetch(url)

        body_text = re.sub(r"<[^>]+>", " ", html).strip()

        if not body_text or len(body_text) < 100:
            raise ValueError("No content extracted or content is too short. The page may be empty or protected.")

        content = trafilatura.extract(
            html,
            output_format=output_format,
            include_comments=False,
            include_tables=include_tables,
            include_links=include_links,
            include_images=include_images,
            include_formatting=include_formatting,
            favor_precision=favor_precision,
            favor_recall=favor_recall,
            deduplicate=deduplicate,
        )

        if not content or len(content.strip()) < 100:
            content = body_text

        metadata = {
            "title": title or "",
            "author": "",
            "domain": extract_domain(url),
            "scraping_method": "selenium",
            "output_format": output_format,
        }

        return content.encode("utf-8"), metadata

    except ValueError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            raise ValueError(f"Page load timed out for {extract_domain(url)}")
        elif "connection" in error_msg or "network" in error_msg:
            raise ValueError(f"Failed to connect to {extract_domain(url)}")
        elif "ssl" in error_msg or "certificate" in error_msg:
            raise ValueError(f"SSL certificate error while accessing {extract_domain(url)}")
        elif "chromedriver" in error_msg or "chrome" in error_msg:
            raise ValueError(
                "Failed to initialize Chrome browser. "
                "Please ensure Chrome is installed on your system."
            )
        else:
            raise ValueError(f"Failed to scrape URL with browser: {str(e)}")


def _discover_links_browser(url: str, start_path: str, max_links: int) -> list:
    """Use Selenium to extract internal links from a JS-rendered page."""
    driver = None
    try:
        driver = _create_chrome_driver()
        driver.get(url)
        _wait_for_js(driver)

        parsed_base = urlparse(url)
        base = f"{parsed_base.scheme}://{parsed_base.netloc}"
        _skip_ext = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".css", ".js", ".xml"}

        links = set()
        for el in driver.find_elements(By.TAG_NAME, "a"):
            href = el.get_attribute("href")
            if not href:
                continue
            href = href.split("#")[0].split("?")[0]
            if not href.startswith(base):
                continue
            if any(href.lower().endswith(ext) for ext in _skip_ext):
                continue
            if start_path and not urlparse(href).path.startswith(start_path):
                continue
            links.add(href)

        return list(links)[:max_links]
    except Exception:
        return []
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def _format_page_section(
    url: str,
    content: str,
    page_meta,
    include_author: bool,
    include_date: bool,
    include_description: bool,
    include_tags: bool,
    include_sitename: bool,
) -> str:
    """Format a single crawled page as a markdown section."""
    title = (page_meta.title if page_meta and page_meta.title else None) or url
    section = f"# {title}\n{url}\n"

    meta_parts = []
    if include_author and page_meta and page_meta.author:
        meta_parts.append(f"**Author:** {page_meta.author}")
    if include_date and page_meta and page_meta.date:
        meta_parts.append(f"**Date:** {page_meta.date}")
    if include_description and page_meta and page_meta.description:
        meta_parts.append(f"**Description:** {page_meta.description}")
    if include_tags and page_meta and page_meta.tags:
        tags_str = ", ".join(page_meta.tags) if isinstance(page_meta.tags, (list, tuple)) else str(page_meta.tags)
        meta_parts.append(f"**Tags:** {tags_str}")
    if include_sitename and page_meta and page_meta.sitename:
        meta_parts.append(f"**Site:** {page_meta.sitename}")

    if meta_parts:
        section += " | ".join(meta_parts) + "\n"

    section += f"\n{content}\n\n---\n\n"
    return section


def _classify_fetch_error(e: Exception) -> str:
    """Return a short, user-readable reason for a fetch/extraction failure."""
    msg = str(e)
    lower = msg.lower()
    for code, label in [
        ("403", "Blocked (403 Forbidden) — site may use bot protection"),
        ("401", "Authentication required (401)"),
        ("404", "Page not found (404)"),
        ("429", "Rate limited (429) — too many requests"),
        ("500", "Server error (500)"),
        ("503", "Service unavailable (503)"),
    ]:
        if code in msg:
            return label
    if "timeout" in lower:
        return "Request timed out"
    if "ssl" in lower or "certificate" in lower:
        return "SSL/certificate error"
    if "connection" in lower:
        return "Connection failed"
    if "no content" in lower or "too short" in lower:
        return "No extractable content (page may require JS rendering)"
    return msg[:120] if len(msg) > 120 else msg


def crawl_url(
    url: str,
    method: str = "crawler",
    max_pages: int = 10,
    lang: Optional[str] = None,
    respect_robots: bool = True,
    sitemap_url: Optional[str] = None,
    use_browser: bool = False,
    include_author: bool = True,
    include_date: bool = True,
    include_description: bool = False,
    include_tags: bool = False,
    include_sitename: bool = False,
    **extraction_params,
) -> tuple[bytes, dict]:
    """Crawl multiple pages from a site and merge into a single document.

    Args:
        url: Starting URL
        method: Discovery method: "crawler", "sitemap", or "feeds"
        max_pages: Maximum number of pages to scrape (1-100)
        lang: Optional language filter (ISO 639-1 code, e.g. "en")
        respect_robots: Whether to respect robots.txt (crawler method only)
        sitemap_url: Override sitemap location (sitemap method only)
        include_author: Include author in per-page metadata header
        include_date: Include publication date in per-page metadata header
        include_description: Include description in per-page metadata header
        include_tags: Include tags/categories in per-page metadata header
        include_sitename: Include site name in per-page metadata header
        **extraction_params: Forwarded to trafilatura.extract() per page

    Returns:
        Tuple of (merged_content_bytes, metadata_dict)

    Raises:
        ValueError: If URL is invalid or crawling fails
    """
    url = validate_url(url)
    domain = extract_domain(url)
    lang_filter = lang.strip() if lang and lang.strip() else None

    parsed_start = urlparse(url)
    start_path = parsed_start.path.rstrip("/")

    # Discover URLs based on method
    if method == "crawler":
        if use_browser:
            # JS-rendered sites: extract links from the rendered DOM
            urls_to_scrape = _discover_links_browser(url, start_path, max_pages)
        else:
            rules = spider.get_rules(url) if respect_robots else None
            # Crawl a larger budget so path-filtering still yields enough results
            crawl_budget = min(max_pages * 3, max_pages + 50)
            _, known_links = spider.focused_crawler(
                url, max_seen_urls=crawl_budget, lang=lang_filter, rules=rules
            )
            if start_path:
                urls_to_scrape = [
                    u for u in known_links if urlparse(u).path.startswith(start_path)
                ][:max_pages]
            else:
                urls_to_scrape = list(known_links)[:max_pages]
    elif method == "sitemap":
        sitemap_target = sitemap_url or url
        discovered = sitemaps.sitemap_search(sitemap_target, target_lang=lang_filter)
        urls_to_scrape = discovered[:max_pages]
    elif method == "feeds":
        feed_article_urls = feeds.find_feed_urls(url, target_lang=lang_filter)
        urls_to_scrape = feed_article_urls[:max_pages]
    else:
        raise ValueError(f"Unknown crawl method: {method}")

    if not urls_to_scrape:
        raise ValueError(
            f"No pages discovered using {method} method. "
            "The site may not have a sitemap/feed, or the crawler found no internal links."
        )

    # Extraction params for per-page content — always markdown since we build a merged doc
    page_extraction = {k: v for k, v in extraction_params.items() if k != "output_format"}

    # Scrape each discovered URL with a single fetch per page
    sections = []
    page_results = []
    for page_url in urls_to_scrape:
        try:
            if use_browser:
                content_bytes, page_html = _scrape_selenium_with_html(page_url, **page_extraction)
                content = content_bytes.decode("utf-8")
                page_meta = trafilatura.extract_metadata(page_html) if page_html else None
            else:
                html = _fetch(page_url, timeout=15)
                content = trafilatura.extract(
                    html,
                    output_format="markdown",
                    include_comments=False,
                    **page_extraction,
                )
                if not content or len(content.strip()) < 100:
                    raise ValueError("No extractable content (page may require JS rendering)")
                page_meta = trafilatura.extract_metadata(html)

            title = (page_meta.title if page_meta and page_meta.title else None) or page_url

            sections.append(_format_page_section(
                page_url, content, page_meta,
                include_author, include_date, include_description,
                include_tags, include_sitename,
            ))
            page_results.append({"url": page_url, "title": title, "status": "ok"})
        except Exception as e:
            page_results.append({
                "url": page_url,
                "title": page_url,
                "status": "failed",
                "reason": _classify_fetch_error(e),
            })

        time.sleep(1)  # polite pacing between requests

    scraped_count = sum(1 for p in page_results if p["status"] == "ok")
    failed_count = len(page_results) - scraped_count

    if scraped_count == 0:
        reasons = "; ".join({p["reason"] for p in page_results if p.get("reason")})
        raise ValueError(
            f"Failed to extract content from any of the {len(page_results)} discovered pages. "
            + (f"Errors: {reasons}" if reasons else "The site may require JavaScript rendering or block automated access.")
        )

    # Build merged document
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = f"Crawl: {domain} | Date: {date_str} | Pages: {scraped_count} | Method: {method}\n\n---\n\n"
    merged = header + "".join(sections)

    metadata = {
        "title": f"Crawl: {domain}",
        "domain": domain,
        "scraping_method": f"crawl_{method}",
        "crawl_method": method,
        "page_count": scraped_count,
        "failed_count": failed_count,
        "page_results": page_results,
        "source_url": url,
    }

    return merged.encode("utf-8"), metadata


def scrape_url_to_markdown(
    url: str,
    use_browser: bool = False,
    output_format: str = "markdown",
    favor_precision: bool = False,
    favor_recall: bool = False,
    include_links: bool = True,
    include_images: bool = True,
    include_formatting: bool = False,
    include_tables: bool = True,
    deduplicate: bool = False,
) -> tuple[bytes, dict]:
    """Scrape URL and extract content.

    Main entry point for URL scraping. Routes to appropriate scraper based on options.

    Args:
        url: The URL to scrape (will be validated and normalized)
        use_browser: If True, use Selenium for JS rendering (slower)
        output_format: Output format: markdown, txt, csv, json, html, xml, xmltei
        favor_precision: Prefer less text but cleaner extraction
        favor_recall: Prefer more text even when uncertain
        include_links: Keep hyperlinks with their targets
        include_images: Include image references and alt text
        include_formatting: Preserve text formatting (bold, italic)
        include_tables: Include table content
        deduplicate: Remove duplicate content segments

    Returns:
        Tuple of (content_bytes, metadata_dict)
        Metadata includes: title, author, domain, scraping_method, output_format

    Raises:
        ValueError: If URL is invalid or scraping fails
    """
    # Validate URL first
    url = validate_url(url)

    # Common extraction parameters
    extraction_params = {
        "output_format": output_format,
        "favor_precision": favor_precision,
        "favor_recall": favor_recall,
        "include_links": include_links,
        "include_images": include_images,
        "include_formatting": include_formatting,
        "include_tables": include_tables,
        "deduplicate": deduplicate,
    }

    # Route to appropriate scraper
    if use_browser:
        return scrape_with_selenium(url, **extraction_params)
    else:
        return scrape_with_trafilatura(url, **extraction_params)
