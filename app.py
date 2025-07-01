import streamlit as st
import requests
import random
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
import io
from time import sleep
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# --- Configurations ---
RSS_FEEDS = [
    "https://www.nst.com.my/feed",
    "https://www.freemalaysiatoday.com/category/nation/feed/",
    "https://www.bharian.com.my/feed",
    "https://www.sinarharian.com.my/rssFeed/211",
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]
NAMESPACES = {'content': 'http://purl.org/rss/1.0/modules/content/'}

def fetch_rss_feed(url, log):
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        log.append(f"[✓] RSS fetched: {url}")
        return ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as e:
        log.append(f"[✗] Error fetching RSS feed from {url}: {e}")
        return None

def fetch_body_content(url, log):
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        article_tag = soup.find('article')
        log.append(f"[✓] Article fetched: {url}")
        return article_tag.get_text() if article_tag else "Content not available"
    except requests.RequestException as e:
        log.append(f"[✗] Error fetching article from {url}: {e}")
        return "Content not available"

st.set_page_config(page_title="RSS Feed Crawler", layout="wide")
st.title("📰 RSS Feed Crawler for Malaysian News")

selected_feeds = st.multiselect(
    "Select RSS feeds to crawl:",
    RSS_FEEDS,
    default=RSS_FEEDS
)
crawl_button = st.button("Crawl News Feeds")

data = []
debug_log = []

if crawl_button and selected_feeds:
    with st.spinner("Crawling selected RSS feeds..."):
        # Count total articles for progress bar
        # First, get number of articles for all feeds (optional, otherwise use approximation)
        article_count = 0
        temp_articles = []
        for feed_url in selected_feeds:
            feed = fetch_rss_feed(feed_url, debug_log)
            if feed:
                articles = list(feed.findall('.//item'))
                temp_articles.extend([(feed_url, item) for item in articles])
        article_count = len(temp_articles)
        progress = st.progress(0)
        completed = 0

        for feed_url, item in temp_articles:
            content_encoded = item.find('content:encoded', NAMESPACES)
            title = item.find('title').text if item.find('title') is not None else "No title"
            description = item.find('description').text if item.find('description') is not None else "No description"
            link = item.find('link').text if item.find('link') is not None else "No link"
            date = item.find('pubDate').text if item.find('pubDate') is not None else "No date"

            if content_encoded is not None and content_encoded.text and content_encoded.text.strip():
                soup = BeautifulSoup(content_encoded.text, 'html.parser')
                article_content = soup.get_text()
            elif description:
                soup = BeautifulSoup(description, 'html.parser')
                article_content = soup.get_text()
            else:
                article_content = fetch_body_content(link, debug_log)

            preview = article_content[:300] + ('...' if len(article_content) > 300 else '')
            data.append([title, link, preview, date])

            completed += 1
            progress.progress(completed / article_count)
            debug_log.append(f"[{completed}/{article_count}] '{title}' scraped from {feed_url}")
            sleep(1)  # Respectful crawling

    if data:
        df = pd.DataFrame(data, columns=['Title', 'Link', 'Article Preview', 'Date'])
        st.success("Crawling completed! See the DataFrame below.")
        st.dataframe(df, use_container_width=True)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue().encode('utf-8-sig'),
            file_name='feed_data_new.csv',
            mime='text/csv'
        )
    else:
        st.info("No articles found.")

    with st.expander("Show Debug Log"):
        st.text('\n'.join(debug_log))

else:
    st.info("Select at least one RSS feed and click 'Crawl News Feeds'.")
