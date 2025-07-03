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
import re
import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import pipeline
import time

# -----------------
# A. Auto classifier
# -----------------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = [
    "This article is about a major risk to a public university",
    "This article is about a moderate or ongoing risk to a public university",
    "This article is about a low or routine issue"
]

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt_tab")

nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Predefined keyword list
predefined_keywords = []

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
    df['Sentiment'] = df['Article'].apply(analyze_sentiment)
    df['Sentiment Score'] = df['Article'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

st.set_page_config(page_title="RSS Feed Crawler", layout="wide")
st.title("📰 RSS Feed Crawler for Malaysian News")

@st.cache_data(show_spinner=True)
def extract_entities(articles):
    entity_counter = Counter()
    entity_list = []
    for text in articles:
        doc = nlp(text)
        ents = [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 2]
        entity_list.append(ents)
        for ent_text, ent_label in ents:
            entity_counter[(ent_text, ent_label)] += 1
    # Show most common, with type in label
    sorted_entities = [f"{e[0]} ({e[1]})" for e, _ in entity_counter.most_common(50)]
    return entity_list, sorted_entities

def extract_entities_spacy(text):
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 2]

def summarize_text(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

def classify_article(article):
    result = classifier(article, candidate_labels)
    return result['labels'][0], result['scores'][0]  # Top label and score

# --------------------------
# 0. List RSS to choose from
# --------------------------
selected_titles = st.multiselect(
    "Select RSS feeds to crawl:",
    feed_titles,
    default=feed_titles
)
selected_feeds = [RSS_FEEDS_DICT[title] for title in selected_titles]

crawl_button = st.button("Crawl News Feeds")

# ---------------------------------------------------------
# 1. User clicks "Crawl" => Store data/df in session_state
# ---------------------------------------------------------
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
    
        if data:
            with st.spinner("Performing named entity recognition..."):
                df['Entities'] = df['Article'].apply(extract_entities_spacy)
                st.session_state['df'] = df
            st.success("NER completed!")

            if data:
                with st.spinner("Create summary..."):
                    df['Summary'] = df['Article'].apply(lambda x: summarize_text(x, 2))
                    st.session_state['df'] = df
                st.success("Summarisation completed!")

                if data:
                    with st.spinner("Auto classify..."):

                        progress_bar = st.progress(0, text="Classifying...")
                        times = []
                        start_total = time.time()
                        
                        for i in range(2):  # Only the first two rows
                            start_article = time.time()
                            
                            label, score = classify_article(df.iloc[i]['Summary'])
                            df.at[df.index[i], 'AI_Risk'] = label
                            df.at[df.index[i], 'AI_Risk_Score'] = score

                            end_article = time.time()
                            times.append(end_article - start_article)

                            # Update progress bar
                            progress = (i + 1) / 2  # Or use total N for full loop
                            progress_bar.progress(progress, text=f"Classifying row {i+1}/2...")

                        total_time = time.time() - start_total

                        st.session_state['df'] = df
                        
                    st.success(
                        f"Auto classification completed! ⏱️ "
                        f"Avg/article: {sum(times)/len(times):.2f}s, "
                        f"Total: {total_time:.2f}s"
                    )
            
# --------------------------------------
# 2. Display the df with Sentiment + NER
# --------------------------------------
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Ensure columns exist in your permanent DataFrame
    for col in ["Relevancy", "Theme"]:
        if col not in st.session_state['df'].columns:
            st.session_state['df'][col] = ""
    
    st.dataframe(df, use_container_width=True)

    # --------------------------------
    # 3. Provide download of RAW data
    # --------------------------------
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue().encode('utf-8-sig'),
        file_name='feed_data_new.csv',
        mime='text/csv'
    )

    # --------------------------
    # 4. Add filter by sentiment
    # --------------------------
    st.subheader("Filter by Sentiment")
    sentiment_options = df['Sentiment'].unique().tolist()
    selected_sentiment = st.multiselect('Filter by Sentiment', sentiment_options, default=sentiment_options)
    
    filtered_df = st.session_state['df'][st.session_state['df']['Sentiment'].isin(selected_sentiment)] # <- filter that damn bastard

    # -------------------------------
    # 5. Add filter auto-gen keywords
    # -------------------------------
    st.subheader("Filter by Auto-generated Keywords")
    keywords = get_keywords_lemmatized(filtered_df['Article'].tolist(), n=50)
    selected_keywords = st.multiselect('Filter by Keyword', keywords, default=[])

    if selected_keywords:
        selected_keywords = [k.lower() for k in selected_keywords]
        mask = filtered_df['Article'].apply(
            lambda x: any(lk in lemmatize_text(x).split() for lk in selected_keywords)
        )
        filtered_df = filtered_df[mask]
        
    # ---------------------
    # 6. Add filter for NER
    # ---------------------
    entities_flat = []
    for ents in filtered_df['Entities']:
        entities_flat.extend(ents)
    
    # Convert to "Text (LABEL)" format and deduplicate
    all_entities = sorted(set(f"{ent[0]} ({ent[1]})" for ent in entities_flat), key=lambda x: x.lower())
    
    selected_entities = st.multiselect("Filter by Named Entity", all_entities)

    # Convert selected string to tuple
    selected_tuples = [(s.rsplit(" (", 1)[0], s.rsplit("(", 1)[1].replace(")", "")) for s in selected_entities]
    
    if selected_entities:
        mask = filtered_df['Entities'].apply(lambda ents: any(e in selected_tuples for e in ents))
        filtered_df = filtered_df[mask]

    # ---------------------------------
    # 7. Add filter for custom keywords
    # ---------------------------------
    if 'keyword_input' not in st.session_state:
        st.session_state['keyword_input'] = ", ".join(predefined_keywords)
    
    st.subheader("Filter by Your Keywords")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_keywords = st.text_area(
            "Enter keywords to filter (comma separated):",
            value=st.session_state['keyword_input'],
            key='kw_input_area'
        )
    with col2:
        if st.button("Reset to Default Keywords"):
            st.session_state['keyword_input'] = ", ".join(predefined_keywords)
            st.experimental_rerun()
    
    # Split and lowercase user input, use the actual text area value!
    input_keywords = [w.strip().lower() for w in user_keywords.split(",") if w.strip()]
    
    match_mode = st.radio("Keyword Match Logic", options=['Any (OR)', 'All (AND)'], index=0, horizontal=True)
    
    # Only show info if input_keywords is empty
    if not input_keywords:
        st.info("No keywords entered — showing all articles.")
    
    # Only filter if there are keywords
    if input_keywords:
        def keyword_match(text):
            text = str(text).lower()
            if match_mode == 'Any (OR)':
                return any(k in text for k in input_keywords)
            else:
                return all(k in text for k in input_keywords)
        mask = filtered_df['Article'].apply(keyword_match)
        filtered_df = filtered_df[mask]
        if filtered_df.empty:
            st.warning("No articles match the selected keywords.")

    # Add columns for tagging if not already present
    if 'tagged_df' not in st.session_state or st.session_state['tagged_df'].shape[0] != filtered_df.shape[0]:
        st.session_state['tagged_df'] = filtered_df.copy()
    
    # Editable grid
    editable_columns = ['Relevancy', 'Theme']
    all_columns = filtered_df.columns.tolist()
    read_only_columns = [col for col in all_columns if col not in editable_columns]
    
    edited_df = st.data_editor(
        filtered_df,
        column_config={
            "Relevancy": st.column_config.SelectboxColumn(
                options=["High", "Mid", "Low"]
            ),
            "Theme": st.column_config.SelectboxColumn(
                options=["Funding", "Governance", "Reputation", "Integrity", "Cyber", "Other"]
            ),
        },
        disabled=read_only_columns,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
    )
    st.session_state['tagged_df'] = edited_df  # Persist edits
    
    # Export
    csv = io.StringIO()
    st.session_state['tagged_df'].to_csv(csv, index=False)
    st.download_button("Download tagged CSV", csv.getvalue(), "tagged_news.csv")

    with st.expander("Show Debug Log"):
        st.text('\n'.join(st.session_state.get('debug_log', [])))

else:
    st.info("Select at least one RSS feed and click 'Crawl News Feeds'.")
