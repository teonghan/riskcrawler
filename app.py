import streamlit as st
import requests
import random
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
import io
from time import sleep
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import string

nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# --- Configurations ---
RSS_FEEDS_DICT = {
    "New Straits Times (NST)": "https://www.nst.com.my/feed",
    "Free Malaysia Today (FMT)": "https://www.freemalaysiatoday.com/category/nation/feed/",
    "Berita Harian (BH)": "https://www.bharian.com.my/feed",
    "Sinar Harian": "https://www.sinarharian.com.my/rssFeed/211",
}

feed_titles = list(RSS_FEEDS_DICT.keys())

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

def get_keywords_lemmatized(texts, n=50):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'\b\w{3,}\b')
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    # Lemmatize vocabulary
    lemmatized_vocab = [lemmatizer.lemmatize(word) for word in vocab]
    sum_words = X.sum(axis=0)
    # Sum frequencies for lemmatized roots
    word_freq = {}
    for idx, lemma in enumerate(lemmatized_vocab):
        word_freq[lemma] = word_freq.get(lemma, 0) + sum_words[0, idx]
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_keywords[:n]]

def lemmatize_text(text):
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
        
def get_keywords(texts, n=20):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'\b\w{3,}\b')
    X = vectorizer.fit_transform(texts)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [w for w, f in words_freq[:n]]

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
        
def crawl_feeds(selected_feeds_tuple):
    data = []
    debug_log = []
    temp_articles = []
    for feed_url in selected_feeds_tuple:
        feed = fetch_rss_feed(feed_url, debug_log)
        if feed:
            articles = list(feed.findall('.//item'))
            temp_articles.extend([(feed_url, item) for item in articles])
    article_count = len(temp_articles)
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
        data.append([title, link, article_content, date])
        completed += 1
        debug_log.append(f"[{completed}/{article_count}] '{title}' scraped from {feed_url}")
    return data, debug_log

@st.cache_data(show_spinner=False)
def sentiment_analysis(data_tuple):
    df = pd.DataFrame(list(data_tuple), columns=['Title', 'Link', 'Article', 'Date'])
    df['Sentiment'] = df['Title'].apply(analyze_sentiment)
    df['Sentiment Score'] = df['Title'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

st.set_page_config(page_title="RSS Feed Crawler", layout="wide")
st.title("📰 RSS Feed Crawler for Malaysian News")

selected_titles = st.multiselect(
    "Select RSS feeds to crawl:",
    feed_titles,
    default=feed_titles
)
selected_feeds = [RSS_FEEDS_DICT[title] for title in selected_titles]

crawl_button = st.button("Crawl News Feeds")

# 1. User clicks "Crawl" => Store data/df in session_state
if crawl_button and selected_feeds:
    with st.spinner("Crawling selected RSS feeds..."):
        data, debug_log = crawl_feeds(tuple(selected_feeds))
    st.session_state['data'] = data
    st.session_state['debug_log'] = debug_log
    if data:
        with st.spinner("Performing sentiment analysis..."):
            df = sentiment_analysis(tuple(tuple(row) for row in data))
        st.session_state['df'] = df
        st.success("Crawling & sentiment analysis completed!")

# 2. Now display filters and DataFrame if session_state['df'] exists
if 'df' in st.session_state:
    df = st.session_state['df']
    st.dataframe(df, use_container_width=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue().encode('utf-8-sig'),
        file_name='feed_data_new.csv',
        mime='text/csv'
    )

    sentiment_options = df['Sentiment'].unique().tolist()
    selected_sentiment = st.multiselect('Filter by Sentiment', sentiment_options, default=sentiment_options)
    filtered_df = df[df['Sentiment'].isin(selected_sentiment)]

    keywords = get_keywords_lemmatized(filtered_df['Article'].tolist(), n=50)
    selected_keywords = st.multiselect('Filter by Keyword', keywords, default=[])

    if selected_keywords:
        selected_keywords = [k.lower() for k in selected_keywords]
        mask = filtered_df['Article'].apply(
            lambda x: any(lk in lemmatize_text(x).split() for lk in selected_keywords)
        )
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df, use_container_width=True)

    with st.expander("Show Debug Log"):
        st.text('\n'.join(st.session_state.get('debug_log', [])))

else:
    st.info("Select at least one RSS feed and click 'Crawl News Feeds'.")
