# app_scrape_only.py
import streamlit as st
import requests
import random
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from io import StringIO

# Compact JSON helpers
import re, json, hashlib
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

# --------------------------- Config ---------------------------
RSS_FEEDS = {
    "New Straits Times": "https://www.nst.com.my/feed",
    "Malay Mail": "https://www.malaymail.com/feed/rss/malaysia",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
]
NAMESPACES = {"content": "http://purl.org/rss/1.0/modules/content/"}
_TRACKING_KEYS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "gclid","fbclid","mc_cid","mc_eid","igshid","mibextid"
}

# -------------------- Cleanup & Summarization ------------------
def _clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = str(s).replace("\xa0", " ").replace("\u200b", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url)
        q = [(k, v) for (k, v) in parse_qsl(parts.query, keep_blank_values=False)
             if k not in _TRACKING_KEYS]
        new_query = urlencode(q, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, ""))
    except Exception:
        return url

def _safe_iso(pubdate: str | None) -> str | None:
    try:
        if not pubdate or pubdate == "No date":
            return None
        dt = parsedate_to_datetime(pubdate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

def _make_id(title: str | None, link: str | None) -> str:
    base = f"{title or ''}|{link or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

def _ensure_nltk():
    import nltk
    needed = [("tokenizers/punkt", "punkt"), ("corpora/stopwords", "stopwords")]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

def _summarize_nltk(text: str, max_chars: int = 220, max_sentences: int = 2) -> str:
    text = _clean_text(text)
    if not text or len(text) <= max_chars:
        return text
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize

        _ensure_nltk()
        sw = set(stopwords.words("english")) | set(stopwords.words("indonesian"))
        sw |= {"yang","dan","atau","untuk","kepada","dalam","dengan","itu","ini","tidak","ada","bagi","oleh","terhadap","akan","kerana","juga"}

        sentences = sent_tokenize(text)
        words = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in sw]
        if not sentences or not words:
            raise ValueError("Empty after tokenization")

        from collections import Counter
        freq = Counter(words)
        scored = []
        for i, s in enumerate(sentences):
            toks = [w for w in word_tokenize(s.lower()) if w.isalpha()]
            sc = sum(freq.get(w, 0) for w in toks) / (1 + len(toks))
            sc *= 1.05 ** max(0, (len(sentences) - i))
            scored.append((i, sc))
        keep_idx = sorted(i for i, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:max(1, min(max_sentences, len(sentences)))])
        summary = _clean_text(" ".join(sentences[i] for i in keep_idx))
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
        return summary
    except Exception:
        return (text[:max_chars].rsplit(" ", 1)[0] + "…") if len(text) > max_chars else text

def articles_to_raw_json(
    articles: list[dict],
    include_content: bool = False,
    max_items: int = 80,
    max_summary_chars: int = 220,
    max_summary_sentences: int = 2,
) -> dict:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items = []
    seen = set()
    for a in articles:
        title = _clean_text(a.get("Title"))
        url = _canonical_url(a.get("Link"))
        uid = _make_id(title, url)
        if uid in seen:
            continue
        seen.add(uid)

        item = {
            "id": uid,
            "title": title,
            "url": url,
            "source": _clean_text(a.get("Source")),
            "publishedAt": _safe_iso(a.get("Date")),
            "summary": _summarize_nltk(_clean_text(a.get("Article")), max_chars=max_summary_chars, max_sentences=max_summary_sentences),
        }
        if include_content:
            item["content"] = _clean_text(a.get("Article"))
        items.append(item)

    # Keep the most recent items (where possible)
    def _key(x):
        return x["publishedAt"] or ""
    items = sorted(items, key=_key, reverse=True)[:max_items]
    return {"asOf": now_iso, "items": items}

# ------------------------ Scraping core ------------------------
def get_random_user_agent() -> str:
    return random.choice(USER_AGENTS)

def fetch_rss_feed(url: str):
    headers = {"User-Agent": get_random_user_agent()}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        st.error(f"Error fetching/parsing RSS: {url} — {e}")
        return None

def fetch_body_content(url: str) -> str:
    headers = {"User-Agent": get_random_user_agent()}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        node = (
            soup.find("article")
            or soup.find("div", class_="entry-content")
            or soup.find("div", class_="td-post-content")
            or soup.find("div", class_="g-item-content")
        )
        if node:
            for s in node(["script", "style"]):
                s.extract()
            return node.get_text(separator="\n", strip=True)
        return "Content not available (specific article tag not found)"
    except requests.RequestException as e:
        st.warning(f"Fetch article error: {url} — {e}")
        return "Content not available (network/request error)"
    except Exception as e:
        st.warning(f"Parsing error: {url} — {e}")
        return "Content not available (parsing error)"

def parse_article_content(item: ET.Element, link: str) -> str:
    content_encoded = item.find("content:encoded", NAMESPACES)
    description = item.find("description").text if item.find("description") is not None else ""
    if content_encoded is not None and content_encoded.text and content_encoded.text.strip():
        soup = BeautifulSoup(content_encoded.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    if description:
        soup = BeautifulSoup(description, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    return fetch_body_content(link)

def scrape_feed(feed_name: str, feed_url: str, status_placeholder, delay_sec: int = 1) -> list[dict]:
    out = []
    feed = fetch_rss_feed(feed_url)
    if not feed:
        return out
    items = feed.findall(".//item")
    status_placeholder.text(f"Found {len(items)} articles in '{feed_name}'.")
    for i, item in enumerate(items, 1):
        title = item.find("title").text if item.find("title") is not None else "No title"
        link = item.find("link").text if item.find("link") is not None else "No link"
        date = item.find("pubDate").text if item.find("pubDate") is not None else "No date"
        article = parse_article_content(item, link)
        out.append({"Source": feed_name, "Title": title, "Link": link, "Article": article, "Date": date})
        status_placeholder.text(f"[{feed_name}] {i}/{len(items)} — {title[:70]}…")
        sleep(delay_sec)
    return out

# -------------------------- UI (minimal) -----------------------
def main():
    st.set_page_config(layout="wide", page_title="News Scraper (Minimal)")
    st.title("📰 Malaysian News Scraper — Minimal")

    # Sidebar config
    st.sidebar.header("Configuration")
    selected = st.sidebar.multiselect("Sources:", options=list(RSS_FEEDS.keys()), default=list(RSS_FEEDS.keys()))
    delay = st.sidebar.slider("Delay per article (sec)", 1, 10, 2, 1)
    include_content = st.sidebar.checkbox("Include full text in JSON (bigger file)", value=False)
    max_items = st.sidebar.slider("Max items in JSON", 20, 200, 80, 10)
    max_summary_chars = st.sidebar.slider("Summary length (chars)", 100, 400, 220, 10)

    if "scraped_data" not in st.session_state:
        st.session_state.scraped_data = []

    if st.button("🚀 Start Scraping"):
        st.session_state.scraped_data = []
        if not selected:
            st.warning("Please select at least one source.")
        else:
            status = st.empty()
            all_rows = []
            for idx, name in enumerate(selected, 1):
                url = RSS_FEEDS[name]
                status.text(f"Feed {idx}/{len(selected)} — {name} ({url})")
                rows = scrape_feed(name, url, status, delay)
                all_rows.extend(rows)
                sleep(0.5)
            status.text("✅ Scraping complete.")
            st.session_state.scraped_data = all_rows

    # Display & downloads
    if st.session_state.scraped_data:
        df = pd.DataFrame(st.session_state.scraped_data)
        df_display = df.copy()
        df_display["Link"] = df_display["Link"].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        st.subheader("📊 Scraped Data")
        st.write(df_display.to_html(escape=False), unsafe_allow_html=True)

        # CSV download
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name="scraped_news_data.csv",
            mime="text/csv",
        )

        # JSON (compact) download
        payload = articles_to_raw_json(
            st.session_state.scraped_data,
            include_content=include_content,
            max_items=max_items,
            max_summary_chars=max_summary_chars,
            max_summary_sentences=2,
        )
        json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON (compact)",
            data=json_bytes,
            file_name="latest_raw_small.json",
            mime="application/json",
        )
    else:
        st.info("No data yet. Select sources and click **Start Scraping**.")

if __name__ == "__main__":
    main()
