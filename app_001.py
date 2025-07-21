import streamlit as st
import requests
import random
import csv
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from io import StringIO

# Import NLTK for sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import Plotly for charting
import plotly.express as px

# Import scikit-learn for ML classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download VADER lexicon if not already downloaded
# This is a one-time download and will be cached by Streamlit's @st.cache_resource
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

# --- Configuration ---
# List of RSS feeds to scrape
RSS_FEEDS = {
    "New Straits Times": "https://www.nst.com.my/feed",
    "Free Malaysia Today": "https://www.freemalaysiatoday.com/category/nation/feed/",
    "Berita Harian": "https://www.bharian.com.my/feed",
    "Sinar Harian": "https://www.sinarharian.com.my/rssFeed/211",
    # Add more feeds here if needed, e.g.,
    # "The Star": "https://www.thestar.com.my/rss/news/nation"
}

# Example user agents to rotate, helping to avoid being blocked by websites
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0",
]

# Namespace for content:encoded, crucial for parsing certain RSS feed structures
NAMESPACES = {'content': 'http://purl.org/rss/1.0/modules/content/'}

# Keywords for Higher Education Classification (fallback if ML model not used)
HIGHER_ED_KEYWORDS = {
    "High": [
        "university", "universities", "college", "colleges", "student", "students",
        "faculty", "academic", "academics", "campus", "campuses", "higher education",
        "tertiary education", "research institution", "degree program", "postgraduate",
        "undergraduate", "enrollment", "scholarship", "scholarships", "tuition fees",
        "curriculum", "accreditation", "chancellor", "vice-chancellor", "dean",
        "professor", "lecturer", "alumni", "graduation", "convocation", "student loan",
        "PTPTN", "MOHE", "ministry of higher education", "polytechnic", "vocational college"
    ],
    "Moderate": [
        "education", "educational", "training", "vocational", "skill development",
        "youth development", "research", "innovation", "technology transfer",
        "funding", "grants", "curriculum development", "career development",
        "public-private partnership", "industry collaboration", "online learning",
        "distance learning", "e-learning", "lifelong learning"
    ],
    "Low": [
        "school", "schools", "teacher", "teachers", "primary education", "secondary education",
        "general policy", "economy", "health", "environment", "infrastructure"
    ]
}


# --- Helper Functions ---

def get_random_user_agent():
    """Returns a random user agent from the predefined list."""
    return random.choice(USER_AGENTS)

def fetch_rss_feed(url: str):
    """
    Fetches and parses an RSS feed from the given URL.

    Args:
        url (str): The URL of the RSS feed.

    Returns:
        xml.etree.ElementTree.Element or None: The parsed XML tree if successful,
                                               otherwise None.
    """
    headers = {'User-Agent': get_random_user_agent()}
    try:
        # Send a GET request with a timeout to prevent indefinite waiting
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as e:
        st.error(f"Error fetching or parsing RSS feed from {url}: {e}")
        return None

def fetch_body_content(url: str):
    """
    Fetches and extracts the main article content from a given webpage URL.
    This is used as a fallback if content is not available in the RSS feed.

    Args:
        url (str): The URL of the article webpage.

    Returns:
        str: The extracted article text, or "Content not available" if an error occurs.
    """
    headers = {'User-Agent': get_random_user_agent()}
    try:
        # Send a GET request with a timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to find common article content containers
        # This part might need further refinement for different website structures
        article_content_element = soup.find('article') or \
                                  soup.find('div', class_='entry-content') or \
                                  soup.find('div', class_='td-post-content') or \
                                  soup.find('div', class_='g-item-content') # Added for more general cases

        if article_content_element:
            # Extract text, remove script and style tags for cleaner content
            for script_or_style in article_content_element(["script", "style"]):
                script_or_style.extract()
            return article_content_element.get_text(separator='\n', strip=True)
        else:
            return "Content not available (specific article tag not found)"

    except requests.RequestException as e:
        st.warning(f"Error fetching article content from {url}: {e}")
        return "Content not available (network/request error)"
    except Exception as e:
        st.warning(f"An unexpected error occurred while parsing content from {url}: {e}")
        return "Content not available (parsing error)"

def parse_article_content(item: ET.Element, link: str):
    """
    Parses and extracts the main article content from an RSS feed item.
    It prioritizes 'content:encoded', then 'description', and finally falls back
    to fetching content directly from the article link.

    Args:
        item (ET.Element): The XML element representing an RSS feed item.
        link (str): The URL of the article.

    Returns:
        str: The extracted article content.
    """
    content_encoded = item.find('content:encoded', NAMESPACES)
    description = item.find('description').text if item.find('description') is not None else ""

    if content_encoded is not None and content_encoded.text and content_encoded.text.strip():
        # Content is directly in content:encoded, parse it as HTML
        soup = BeautifulSoup(content_encoded.text, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    elif description:
        # Fallback to description, parse it as HTML
        soup = BeautifulSoup(description, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    else:
        # Last resort: fetch content from the article's webpage
        return fetch_body_content(link)

def scrape_feed(feed_name: str, feed_url: str, article_status_placeholder, article_sleep_time: int = 1):
    """
    Scrapes articles from a single RSS feed URL.

    Args:
        feed_name (str): The name of the feed (for display purposes).
        feed_url (str): The URL of the RSS feed.
        article_status_placeholder (st.empty): Streamlit placeholder to update article progress.
        article_sleep_time (int): Seconds to sleep between processing each article.

    Returns:
        list: A list of dictionaries, where each dictionary represents an article.
    """
    articles_data = []
    feed = fetch_rss_feed(feed_url)
    if feed:
        items = feed.findall('.//item')
        total_items = len(items)

        # Update overall status text to show number of articles found for this feed
        # The main app will handle the primary overall status, this is for more detail
        article_status_placeholder.text(f"Found {total_items} articles in '{feed_name}'. Starting scrape...")

        for i, item in enumerate(items):
            title = item.find('title').text if item.find('title') is not None else "No title"
            link = item.find('link').text if item.find('link') is not None else "No link"
            date = item.find('pubDate').text if item.find('pubDate') is not None else "No date"

            # Parse article content using the modular function
            article_content = parse_article_content(item, link)

            articles_data.append({
                'Source': feed_name, # Add source information
                'Title': title,
                'Link': link,
                'Article': article_content,
                'Date': date
            })

            # Update progress for the current article directly using the placeholder
            article_status_placeholder.text(f"  Scraping article {i+1}/{total_items} from '{feed_name}': {title[:70]}...")

            sleep(article_sleep_time) # Sleep to avoid being blocked

    return articles_data

def analyze_sentiment_vader(text: str, analyzer: SentimentIntensityAnalyzer):
    """
    Performs sentiment analysis using VADER on the given text.
    Classifies sentiment as 'Positive', 'Negative', or 'Neutral' based on compound score.
    """
    if not text or not isinstance(text, str):
        return "Neutral" # Handle empty or non-string inputs

    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def classify_higher_ed_relevance_keyword(text: str):
    """
    Classifies the relevance of an article to the higher education sector
    based on predefined keywords (fallback method).
    """
    if not text or not isinstance(text, str):
        return "Low" # Default for empty or non-string inputs

    text_lower = text.lower()

    # Check for High relevance keywords first (highest priority)
    for keyword in HIGHER_ED_KEYWORDS["High"]:
        if keyword in text_lower:
            return "High"

    # If not High, check for Moderate relevance keywords
    for keyword in HIGHER_ED_KEYWORDS["Moderate"]:
        if keyword in text_lower:
            return "Moderate"

    # If neither High nor Moderate, classify as Low
    return "Low"

def train_ml_classifier(df_labeled: pd.DataFrame, progress_bar, status_text):
    """
    Trains a Logistic Regression classifier for higher education relevance.
    """
    if 'Article' not in df_labeled.columns or 'HigherEdRelevance' not in df_labeled.columns:
        status_text.error("Labeled data must contain 'Article' and 'HigherEdRelevance' columns.")
        return None, None

    # Filter out rows with missing text or labels
    df_labeled = df_labeled.dropna(subset=['Article', 'HigherEdRelevance'])
    if df_labeled.empty:
        status_text.warning("No valid data found in the uploaded file for training.")
        return None, None

    X = df_labeled['Article']
    y = df_labeled['HigherEdRelevance']

    if len(X) < 2: # Need at least 2 samples for train/test split
        status_text.warning("Not enough samples in labeled data to train. Need at least 2.")
        return None, None

    # Ensure all target labels are present in the training set
    unique_labels = y.unique()
    if not all(label in unique_labels for label in ["High", "Moderate", "Low"]):
        st.warning("Labeled data should ideally contain 'High', 'Moderate', and 'Low' labels for robust training.")

    status_text.text("Training model: Initializing TF-IDF Vectorizer...")
    progress_bar.progress(0.1)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') # Limit features for smaller model

    status_text.text("Training model: Fitting Vectorizer and transforming data...")
    progress_bar.progress(0.4)
    X_vectorized = vectorizer.fit_transform(X)

    status_text.text("Training model: Initializing Logistic Regression Classifier...")
    progress_bar.progress(0.7)
    classifier = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence

    status_text.text("Training model: Training Classifier...")
    progress_bar.progress(0.9)
    classifier.fit(X_vectorized, y)

    progress_bar.progress(1.0)
    status_text.success("ML Model trained successfully!")
    return vectorizer, classifier


# --- Streamlit Application ---

def main_app():
    """Main function to run the Streamlit News Scraper application."""
    st.set_page_config(layout="wide", page_title="News Scraper")
    st.title("📰 Malaysian News Scraper")

    st.markdown("""
        This application scrapes news articles from a predefined list of Malaysian news RSS feeds.
        Select the news sources you wish to crawl and click the "Start Scraping" button.
        The results will be displayed below, and you'll have an option to download them as a CSV file.
    """)

    # Initialize session state for scraped data and sentiment data if not already present
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = []
    if 'sentiment_df' not in st.session_state:
        st.session_state.sentiment_df = pd.DataFrame() # Initialize as empty DataFrame
    if 'classification_df' not in st.session_state:
        st.session_state.classification_df = pd.DataFrame() # Initialize as empty DataFrame
    if 'ml_vectorizer' not in st.session_state:
        st.session_state.ml_vectorizer = None
    if 'ml_classifier' not in st.session_state:
        st.session_state.ml_classifier = None

    # Download NLTK data once and get the analyzer
    sid_analyzer = download_nltk_data()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🌐 News Scraper", "📈 Sentiment Analysis", "📚 Higher Ed Classification"])

    with tab1:
        st.sidebar.header("Configuration")

        # 1. User can multi-choose which URL to crawl
        selected_feed_names = st.sidebar.multiselect(
            "Select News Sources to Scrape:",
            options=list(RSS_FEEDS.keys()),
            default=list(RSS_FEEDS.keys()), # Default to all selected
            help="Choose one or more news sources to fetch articles from."
        )

        # Convert selected names back to URLs
        selected_feeds = {name: RSS_FEEDS[name] for name in selected_feed_names}

        # Allow user to adjust sleep time per article
        sleep_time_per_article = st.sidebar.slider(
            "Delay between articles (seconds)",
            min_value=1, max_value=10, value=2, step=1,
            help="Increase this value to reduce the risk of being blocked by websites."
        )
        st.sidebar.info(f"Current delay per article: {sleep_time_per_article} seconds.")

        # 2. Provide a button for user to trigger the crawling process
        if st.button("🚀 Start Scraping"):
            if not selected_feeds:
                st.warning("Please select at least one news source to begin scraping.")
                # Clear existing data if no feeds are selected
                st.session_state.scraped_data = []
                st.session_state.sentiment_df = pd.DataFrame() # Also clear sentiment data
                st.session_state.classification_df = pd.DataFrame() # Also clear classification data
                st.session_state.ml_vectorizer = None # Clear ML model
                st.session_state.ml_classifier = None # Clear ML model
                return

            all_scraped_data = []
            st.subheader("Scraping Progress")

            # Create containers for better control over updates
            progress_container = st.empty()
            status_container = st.empty()
            article_status_container = st.empty() # This placeholder will now be passed directly

            total_feeds_to_scrape = len(selected_feeds)

            for i, (feed_name, feed_url) in enumerate(selected_feeds.items()):
                # Update overall status text
                status_container.text(f"Processing feed {i+1}/{total_feeds_to_scrape}: **{feed_name}** ({feed_url})")

                # Update overall progress bar
                overall_progress_value = (i + 1) / total_feeds_to_scrape
                progress_container.progress(overall_progress_value)

                feed_articles = scrape_feed(
                    feed_name, # Pass feed name for better display
                    feed_url,
                    article_status_container, # Pass the placeholder directly
                    article_sleep_time=sleep_time_per_article
                )
                all_scraped_data.extend(feed_articles)

                # Clear article status after each feed is done, before moving to the next
                article_status_container.empty()

                sleep(1) # Small delay between feeds for better UX on progress bar updates

            status_container.success("✅ Scraping complete!")
            progress_container.progress(1.0) # Ensure it fills to 100% at the end
            article_status_container.empty() # Final clear of any lingering article status

            # Store scraped data in session state for access by other tabs
            st.session_state.scraped_data = all_scraped_data
            st.session_state.sentiment_df = pd.DataFrame() # Reset sentiment data on new crawl
            st.session_state.classification_df = pd.DataFrame() # Reset classification data on new crawl
            # Do NOT reset ML model here, as it's independent of new scrapes unless explicitly re-trained

        # --- Display scraped data in Tab 1 (moved outside the button block) ---
        if st.session_state.scraped_data:
            st.subheader("📊 Scraped News Data")
            df = pd.DataFrame(st.session_state.scraped_data)

            # Format the 'Link' column as clickable hyperlinks
            df['Link'] = df['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

            # Display the DataFrame, allowing HTML in the 'Link' column
            st.write(df.to_html(escape=False), unsafe_allow_html=True)

            # Option to download data as CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="⬇️ Download Data as CSV",
                data=csv_buffer.getvalue().encode('utf-8-sig'),
                file_name="scraped_news_data.csv",
                mime="text/csv",
                help="Click to download the scraped data as a CSV file."
            )
        elif not st.session_state.scraped_data and 'last_scraped_attempted' not in st.session_state:
            # Only show warning if no data and no previous attempt was made (initial load)
            st.info("No data scraped yet. Select sources and click 'Start Scraping'.")
        elif not st.session_state.scraped_data and st.session_state.get('last_scraped_attempted', False):
            # Show warning if an attempt was made but no data was scraped
            st.warning("😔 No data was scraped. Please check the selected RSS feed URLs, your internet connection, or try increasing the delay between articles.")

        # Keep track if a scraping attempt was made
        # This button is already handled above, but its presence triggers re-runs
        # We need to ensure 'last_scraped_attempted' is set correctly
        if st.button("🚀 Start Scraping", key="start_scraping_button_bottom"): # Added key to avoid duplicate key warning
            st.session_state.last_scraped_attempted = True
        elif 'last_scraped_attempted' not in st.session_state:
            st.session_state.last_scraped_attempted = False


    with tab2:
        st.header("📈 Sentiment Analysis of News Articles")
        if st.session_state.scraped_data:
            st.info("Click the button below to perform sentiment analysis on the scraped articles.")
            st.warning("Sentiment analysis uses the NLTK VADER lexicon, which is a rule-based model. Its accuracy may vary depending on the text context and language nuances.")

            # 1. Button to trigger sentiment analysis
            if st.button("✨ Run Sentiment Analysis"):
                df_sentiment = pd.DataFrame(st.session_state.scraped_data)

                if 'Article' in df_sentiment.columns and not df_sentiment['Article'].empty:
                    total_articles = len(df_sentiment)
                    st.subheader("Analysis Progress")
                    analysis_progress_bar = st.progress(0)
                    analysis_status_text = st.empty()

                    sentiment_results = []
                    for idx, row in df_sentiment.iterrows():
                        article_text = row['Article']
                        sentiment = analyze_sentiment_vader(article_text, sid_analyzer)
                        sentiment_results.append(sentiment)

                        # 2. Progress bar for sentiment analysis
                        progress_percentage = (idx + 1) / total_articles
                        analysis_progress_bar.progress(progress_percentage)
                        analysis_status_text.text(f"Analyzing article {idx+1}/{total_articles}...")
                        sleep(0.01) # Small sleep to allow UI to update

                    df_sentiment['Sentiment'] = sentiment_results
                    st.session_state.sentiment_df = df_sentiment # Store the DataFrame with sentiment

                    analysis_status_text.success("✅ Sentiment analysis complete!")
                    analysis_progress_bar.progress(1.0) # Ensure it fills to 100%

                    st.subheader("Sentiment Distribution")
                    # Create a DataFrame for the pie chart
                    sentiment_counts = st.session_state.sentiment_df['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']

                    # Define custom colors for the pie chart
                    color_map = {
                        'Positive': 'green',
                        'Negative': 'red',
                        'Neutral': 'yellow'
                    }

                    # Create the pie chart
                    fig = px.pie(
                        sentiment_counts,
                        values='Count',
                        names='Sentiment',
                        title='Distribution of Article Sentiments',
                        color='Sentiment',
                        color_discrete_map=color_map
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Articles with Sentiment")
                    # Format the 'Link' column as clickable hyperlinks
                    df_sentiment['Link'] = df_sentiment['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
                    # Display the DataFrame with the new 'Sentiment' column
                    st.write(df_sentiment[['Source', 'Title', 'Sentiment', 'Link']].to_html(escape=False), unsafe_allow_html=True)

                    # Option to download data with sentiment as CSV
                    csv_buffer_sentiment = StringIO()
                    st.session_state.sentiment_df.to_csv(csv_buffer_sentiment, index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="⬇️ Download Data with Sentiment as CSV",
                        data=csv_buffer_sentiment.getvalue().encode('utf-8-sig'),
                        file_name="scraped_news_data_with_sentiment.csv",
                        mime="text/csv",
                        help="Click to download the scraped data including sentiment analysis as a CSV file."
                    )
                else:
                    st.warning("No 'Article' content found in the scraped data to perform sentiment analysis.")

            # Display previously analyzed data if available and not re-running
            elif not st.session_state.sentiment_df.empty:
                st.subheader("Previously Analyzed Sentiment Data")
                st.subheader("Sentiment Distribution")

                # Create a DataFrame for the pie chart from previously analyzed data
                sentiment_counts = st.session_state.sentiment_df['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                # Define custom colors for the pie chart
                color_map = {
                    'Positive': 'green',
                    'Negative': 'red',
                    'Neutral': 'yellow'
                }

                # Create the pie chart
                fig = px.pie(
                    sentiment_counts,
                    values='Count',
                    names='Sentiment',
                    title='Distribution of Article Sentiments',
                    color='Sentiment',
                    color_discrete_map=color_map
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Articles with Sentiment")
                # Format the 'Link' column as clickable hyperlinks
                df_display = st.session_state.sentiment_df.copy() # Create a copy to avoid modifying original session state DF
                df_display['Link'] = df_display['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
                st.write(df_display[['Source', 'Title', 'Sentiment', 'Link']].to_html(escape=False), unsafe_allow_html=True)

                csv_buffer_sentiment = StringIO()
                st.session_state.sentiment_df.to_csv(csv_buffer_sentiment, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Download Data with Sentiment as CSV",
                    data=csv_buffer_sentiment.getvalue().encode('utf-8-sig'),
                    file_name="scraped_news_data_with_sentiment.csv",
                    mime="text/csv",
                    help="Click to download the scraped data including sentiment analysis as a CSV file."
                )

        else:
            st.info("Please go to the 'News Scraper' tab and complete the crawling process first to enable sentiment analysis.")

    with tab3:
        st.header("📚 Higher Education News Classification")
        # Determine which DataFrame to use for classification
        data_for_classification = None
        if not st.session_state.sentiment_df.empty:
            data_for_classification = st.session_state.sentiment_df.copy()
            st.info("Using sentiment-analyzed data from Tab 2 for classification.")
        elif st.session_state.scraped_data:
            data_for_classification = pd.DataFrame(st.session_state.scraped_data).copy()
            st.info("Using raw scraped data from Tab 1 for classification (Sentiment Analysis not performed).")

        if data_for_classification is not None and not data_for_classification.empty:
            st.markdown("""
                You can classify articles by their relevance to the Higher Education sector.
                You can use the **default keyword-based classification** or
                **upload your own labeled data (CSV)** to train a custom ML model.
                The CSV should have 'Article' and 'HigherEdRelevance' columns (with labels 'High', 'Moderate', 'Low').
            """)

            uploaded_file = st.file_uploader("Upload Labeled Data (CSV) for ML Model Training", type="csv")

            if uploaded_file is not None:
                df_uploaded = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data (first 5 rows):")
                st.dataframe(df_uploaded.head())

                if st.button("⚙️ Train ML Model"):
                    training_progress_bar = st.progress(0)
                    training_status_text = st.empty()

                    vectorizer, classifier = train_ml_classifier(df_uploaded, training_progress_bar, training_status_text)

                    if vectorizer and classifier:
                        st.session_state.ml_vectorizer = vectorizer
                        st.session_state.ml_classifier = classifier
                        st.success("ML model successfully trained and ready for classification!")
                    else:
                        st.error("Failed to train ML model. Please check your labeled data format.")
                    training_progress_bar.empty() # Clear training progress bar
                    training_status_text.empty() # Clear training status text

            # This block handles running classification (ML or Keyword)
            # It's inside the 'if data_for_classification is not None and not data_for_classification.empty:'
            # to ensure classification is only attempted if there's data.
            if st.session_state.ml_classifier and st.session_state.ml_vectorizer:
                st.success("An ML model is currently loaded and will be used for classification.")
                if st.button("🔍 Run Higher Ed Classification (using ML Model)"):
                    df_classification = data_for_classification.copy() # Use the determined data source

                    if 'Article' in df_classification.columns and not df_classification['Article'].empty:
                        total_articles = len(df_classification)
                        st.subheader("Classification Progress")
                        classification_progress_bar = st.progress(0)
                        classification_status_text = st.empty()

                        relevance_results = []
                        for idx, row in df_classification.iterrows():
                            article_text = row['Article']
                            # Predict using the trained ML model
                            if pd.isna(article_text) or not isinstance(article_text, str):
                                relevance = "Low" # Handle missing/invalid text for prediction
                            else:
                                text_vectorized = st.session_state.ml_vectorizer.transform([article_text])
                                relevance = st.session_state.ml_classifier.predict(text_vectorized)[0]
                            relevance_results.append(relevance)

                            progress_percentage = (idx + 1) / total_articles
                            classification_progress_bar.progress(progress_percentage)
                            classification_status_text.text(f"Classifying article {idx+1}/{total_articles}...")
                            sleep(0.01) # Small sleep to allow UI to update

                        df_classification['HigherEdRelevance'] = relevance_results
                        st.session_state.classification_df = df_classification # Store for persistence

                        classification_status_text.success("✅ Classification complete (ML Model)!")
                        classification_progress_bar.progress(1.0)

                        st.subheader("Relevance Distribution")
                        relevance_counts = st.session_state.classification_df['HigherEdRelevance'].value_counts().reset_index()
                        relevance_counts.columns = ['Relevance', 'Count']

                        relevance_order = ["High", "Moderate", "Low"]
                        relevance_counts['Relevance'] = pd.Categorical(relevance_counts['Relevance'], categories=relevance_order, ordered=True)
                        relevance_counts = relevance_counts.sort_values('Relevance')

                        relevance_color_map = {
                            'High': 'darkgreen',
                            'Moderate': 'orange',
                            'Low': 'grey'
                        }

                        fig_relevance = px.bar(
                            relevance_counts,
                            x='Relevance',
                            y='Count',
                            title='Distribution of Higher Education Relevance (ML Model)',
                            color='Relevance',
                            color_discrete_map=relevance_color_map
                        )
                        st.plotly_chart(fig_relevance, use_container_width=True)

                        st.subheader("Articles with Higher Education Relevance (ML Model)")
                        # Ensure all original columns are present, plus the new 'HigherEdRelevance'
                        display_cols = [col for col in df_classification.columns if col != 'Article']
                        if 'Link' in display_cols:
                            df_classification['Link'] = df_classification['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
                        st.write(df_classification[display_cols].to_html(escape=False), unsafe_allow_html=True)

                        csv_buffer_classification = StringIO()
                        st.session_state.classification_df.to_csv(csv_buffer_classification, index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="⬇️ Download Data with Higher Ed Classification (ML) as CSV",
                            data=csv_buffer_classification.getvalue().encode('utf-8-sig'),
                            file_name="scraped_news_data_with_higher_ed_classification_ml.csv",
                            mime="text/csv",
                            help="Click to download the scraped data including higher education classification as a CSV file."
                        )
                    else:
                        st.warning("No 'Article' content found in the selected data to perform higher education classification with ML model.")
            else:
                # Fallback to keyword-based classification if no ML model is trained
                st.info("No ML model trained. Classification will use the keyword-based approach by default.")
                if st.button("🔍 Run Higher Ed Classification (using Keywords)"):
                    df_classification = data_for_classification.copy() # Use the determined data source

                    if 'Article' in df_classification.columns and not df_classification['Article'].empty:
                        total_articles = len(df_classification)
                        st.subheader("Classification Progress")
                        classification_progress_bar = st.progress(0)
                        classification_status_text = st.empty()

                        relevance_results = []
                        for idx, row in df_classification.iterrows():
                            article_text = row['Article']
                            relevance = classify_higher_ed_relevance_keyword(article_text) # Use keyword function
                            relevance_results.append(relevance)

                            progress_percentage = (idx + 1) / total_articles
                            classification_progress_bar.progress(progress_percentage)
                            classification_status_text.text(f"Classifying article {idx+1}/{total_articles}...")
                            sleep(0.01) # Small sleep to allow UI to update

                        df_classification['HigherEdRelevance'] = relevance_results
                        st.session_state.classification_df = df_classification # Store for persistence

                        classification_status_text.success("✅ Classification complete (Keyword-based)!")
                        classification_progress_bar.progress(1.0)

                        st.subheader("Relevance Distribution")
                        relevance_counts = st.session_state.classification_df['HigherEdRelevance'].value_counts().reset_index()
                        relevance_counts.columns = ['Relevance', 'Count']

                        relevance_order = ["High", "Moderate", "Low"]
                        relevance_counts['Relevance'] = pd.Categorical(relevance_counts['Relevance'], categories=relevance_order, ordered=True)
                        relevance_counts = relevance_counts.sort_values('Relevance')

                        relevance_color_map = {
                            'High': 'darkgreen',
                            'Moderate': 'orange',
                            'Low': 'grey'
                        }

                        fig_relevance = px.bar(
                            relevance_counts,
                            x='Relevance',
                            y='Count',
                            title='Distribution of Higher Education Relevance (Keyword-based)',
                            color='Relevance',
                            color_discrete_map=relevance_color_map
                        )
                        st.plotly_chart(fig_relevance, use_container_width=True)

                        st.subheader("Articles with Higher Education Relevance (Keyword-based)")
                        # Ensure all original columns are present, plus the new 'HigherEdRelevance'
                        display_cols = [col for col in df_classification.columns if col != 'Article']
                        if 'Link' in display_cols:
                            df_classification['Link'] = df_classification['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
                        st.write(df_classification[display_cols].to_html(escape=False), unsafe_allow_html=True)

                        csv_buffer_classification = StringIO()
                        st.session_state.classification_df.to_csv(csv_buffer_classification, index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="⬇️ Download Data with Higher Ed Classification (Keyword) as CSV",
                            data=csv_buffer_classification.getvalue().encode('utf-8-sig'),
                            file_name="scraped_news_data_with_higher_ed_classification_keyword.csv",
                            mime="text/csv",
                            help="Click to download the scraped data including higher education classification as a CSV file."
                        )
                    else:
                        st.warning("No 'Article' content found in the selected data to perform higher education classification with keyword model.")

            # Display previously classified data if available and not re-running
            # This block is now outside the "Run Higher Ed Classification" button conditions
            # but still within the `if data_for_classification is not None and not data_for_classification.empty:`
            if not st.session_state.classification_df.empty:
                st.subheader("Previously Classified Data")
                st.subheader("Relevance Distribution")

                relevance_counts = st.session_state.classification_df['HigherEdRelevance'].value_counts().reset_index()
                relevance_counts.columns = ['Relevance', 'Count']

                relevance_order = ["High", "Moderate", "Low"]
                relevance_counts['Relevance'] = pd.Categorical(relevance_counts['Relevance'], categories=relevance_order, ordered=True)
                relevance_counts = relevance_counts.sort_values('Relevance')

                relevance_color_map = {
                    'High': 'darkgreen',
                    'Moderate': 'orange',
                    'Low': 'grey'
                }

                fig_relevance = px.bar(
                    relevance_counts,
                    x='Relevance',
                    y='Count',
                    title='Distribution of Higher Education Relevance',
                    color='Relevance',
                    color_discrete_map=relevance_color_map
                )
                st.plotly_chart(fig_relevance, use_container_width=True)

                st.subheader("Articles with Higher Education Relevance")
                df_display_classification = st.session_state.classification_df.copy()
                df_display_classification['Link'] = df_display_classification['Link'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')

                # Ensure all original columns are present, plus the new 'HigherEdRelevance'
                display_cols = [col for col in df_display_classification.columns if col != 'Article']
                st.write(df_display_classification[display_cols].to_html(escape=False), unsafe_allow_html=True)

                csv_buffer_classification = StringIO()
                st.session_state.classification_df.to_csv(csv_buffer_classification, index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ Download Data with Higher Ed Classification as CSV",
                    data=csv_buffer_classification.getvalue().encode('utf-8-sig'),
                    file_name="scraped_news_data_with_higher_ed_classification.csv",
                    mime="text/csv",
                    help="Click to download the scraped data including higher education classification as a CSV file."
                )
        else:
            st.info("Please go to the 'News Scraper' tab and complete the crawling process first to enable higher education classification.")

if __name__ == "__main__":
    main_app()
