import streamlit as st
import pandas as pd
from transformers import pipeline # Using the standard Python transformers library
import io

# --- Configuration ---
# Define available models for the user to choose from
AVAILABLE_MODELS = {
    "DeBERTa-v3-xsmall (MoritzLaurer)": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    "XtremeDistil-l6-h256 (MoritzLaurer)": "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
    "DistilBERT-base-uncased-MNLI (Typeform)": "typeform/distilbert-base-uncased-mnli",
    "BART-large-MNLI (Facebook)": "facebook/bart-large-mnli"
}

MAX_ARTICLE_LENGTH = 512 # Max tokens for the model, adjust if needed. Long articles will be truncated.
# CONFIDENCE_THRESHOLD = 0.01 # This will now be set by user input in the UI

# --- Helper Functions ---

@st.cache_resource
def load_zsc_pipeline(model_id: str): # model_id is now an argument
    """
    Loads the Zero-Shot Classification pipeline from Hugging Face.
    This function is cached to avoid reloading the model on every rerun.
    """
    st.info(f"Loading Zero-Shot Classification model: {model_id}. This may take a moment on first run as the model downloads to the server.")
    try:
        # Initialize the zero-shot classification pipeline
        # The model will be downloaded and run server-side
        classifier = pipeline("zero-shot-classification", model=model_id)
        st.success("Model loaded successfully!")
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure you have the 'transformers' library installed and an internet connection on the server.")
        st.stop()

def classify_article(classifier, article_text, candidate_labels, threshold: float): # threshold is now passed as an argument
    """
    Performs zero-shot classification on a single article, returning all labels
    and scores sorted by confidence, and optionally filtered by a threshold.
    Truncates the article if it's too long for the model.
    """
    if not article_text or not isinstance(article_text, str):
        return "N/A", 0.0, []

    # Simple truncation for demonstration. For production, consider advanced chunking.
    if len(article_text.split()) > MAX_ARTICLE_LENGTH:
        article_text = " ".join(article_text.split()[:MAX_ARTICLE_LENGTH])
        # st.warning(f"Article truncated to {MAX_ARTICLE_LENGTH} words for classification.") # Removed for less clutter

    try:
        # Perform multi-label classification
        result = classifier(article_text, candidate_labels, multi_label=True)

        # The result['labels'] and result['scores'] are already sorted from highest to lowest score
        all_labels_with_scores = []
        for label, score in zip(result['labels'], result['scores']):
            if score >= threshold: # Only include labels above the confidence threshold
                all_labels_with_scores.append({"label": label, "score": score})

        # Format for display
        formatted_all_labels = ", ".join([f"{item['label']}: {item['score']:.2f}" for item in all_labels_with_scores])

        # Get the top predicted label and its score
        predicted_label = all_labels_with_scores[0]['label'] if all_labels_with_scores else 'N/A'
        confidence_score = all_labels_with_scores[0]['score'] if all_labels_with_scores else 0.0

        return predicted_label, confidence_score, formatted_all_labels
    except Exception as e:
        st.error(f"Error classifying article: {e}")
        return "Error", 0.0, "Error during classification"

# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="ZSC Risk Assessment (Server-Side)")

st.title("📰 Zero-Shot Classification for News Risk Assessment (Server-Side)")
st.markdown("""
This application uses a pre-trained Zero-Shot Classification (ZSC) model to categorize news articles into custom risk categories you define.
The model inference runs on the server where this Streamlit application is hosted.
It now supports **multi-label classification**, showing all relevant risk labels and their confidence scores.
""")

# --- Model Selection ---
st.header("0. Select ZSC Model")
selected_model_name = st.selectbox(
    "Choose a pre-trained Zero-Shot Classification model:",
    options=list(AVAILABLE_MODELS.keys()),
    index=0, # Default to the first model in the list
    help="Smaller models (e.g., xsmall, XtremeDistil) are faster but might be slightly less accurate. Larger models (e.g., BART-large) offer higher accuracy but are slower."
)
selected_model_id = AVAILABLE_MODELS[selected_model_name]

# --- Model Loading ---
classifier = load_zsc_pipeline(selected_model_id) # Pass the selected model ID

# --- Step 1: Upload News Articles ---
st.header("1. Upload News Articles (CSV or Excel)")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

articles_df = pd.DataFrame()
article_column = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            articles_df = pd.read_csv(uploaded_file)
        else: # .xlsx
            articles_df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(articles_df.head())

        # Allow user to select the article column
        default_article_col = 'Article' if 'Article' in articles_df.columns else articles_df.columns[0]
        article_column = st.selectbox("Select the column containing news articles:", articles_df.columns, index=articles_df.columns.get_loc(default_article_col) if default_article_col in articles_df.columns else 0)

    except Exception as e:
        st.error(f"Error reading file: {e}. Please ensure it's a valid CSV or Excel format.")
        st.stop()

# --- Step 2: Define Risk Descriptions ---
st.header("2. Define Risk Categories")
st.markdown("""
You can define your risk categories below, or **upload a custom CSV/Excel file** to overwrite them.
The file should contain columns: `candidate_label`, `candidate_description`, `keywords`.
""")

# Option to upload custom risk categories
custom_labels_file = st.file_uploader("Upload custom risk categories (CSV or Excel)", type=["csv", "xlsx"], key="custom_labels_uploader")

risk_categories_data = []
if custom_labels_file is not None:
    try:
        if custom_labels_file.name.endswith('.csv'):
            custom_df = pd.read_csv(custom_labels_file)
        else: # .xlsx
            custom_df = pd.read_excel(custom_labels_file)

        # Validate columns
        required_cols = ["candidate_label", "candidate_description", "keywords"]
        if not all(col in custom_df.columns for col in required_cols):
            st.error(f"Uploaded file must contain all required columns: {', '.join(required_cols)}")
            st.stop()

        risk_categories_data = custom_df.to_dict('records')
        st.success("Custom risk categories loaded successfully!")
        st.info("The table below is now populated with your uploaded categories.")

    except Exception as e:
        st.error(f"Error reading custom labels file: {e}. Please ensure it's a valid CSV or Excel format with the correct columns.")
        st.stop()
else:
    # Default risk categories if no file is uploaded
    default_risk_categories = [
        {"candidate_label": "Financial: Market Volatility", "candidate_description": "News indicating significant fluctuations in stock markets, currency exchange rates, or commodity prices.", "keywords": "stock, market, economy, recession, inflation, currency"},
        {"candidate_label": "Operational: Data Breach", "candidate_description": "Reports of cybersecurity incidents, data leaks, or system outages affecting company operations.", "keywords": "cybersecurity, data breach, hack, system, outage, IT"},
        {"candidate_label": "Reputational: Negative Media", "candidate_description": "Articles that could harm the company's public image, including scandals, poor product reviews, or ethical concerns.", "keywords": "scandal, lawsuit, protest, boycott, negative, criticism"},
        {"candidate_label": "Regulatory: Compliance Issue", "candidate_description": "News about new regulations, non-compliance fines, or investigations by regulatory bodies.", "keywords": "regulation, fine, compliance, investigation, lawsuit, government"},
        {"candidate_label": "Environmental: Climate Impact", "candidate_description": "Reports on environmental disasters, climate change impact, or company's environmental footprint.", "keywords": "climate, environment, pollution, carbon, sustainability, disaster"},
        {"candidate_label": "Geopolitical: Political Instability", "candidate_description": "News about political unrest, trade wars, or international conflicts impacting business.", "keywords": "geopolitical, war, conflict, trade, sanctions, election"},
        {"candidate_label": "Positive: Business Growth", "candidate_description": "News indicating positive business developments, new partnerships, or significant revenue growth.", "keywords": "growth, expansion, partnership, revenue, profit, innovation"},
    ]
    risk_categories_data = default_risk_categories


# Use st.data_editor for flexible input of risk categories
risk_categories_df = st.data_editor(
    pd.DataFrame(risk_categories_data),
    num_rows="dynamic",
    column_config={
        "candidate_label": st.column_config.TextColumn("Risk Label (for ZSC)", help="The label the ZSC model will use for classification."),
        "candidate_description": st.column_config.TextColumn("Description", help="A detailed description of this risk category."),
        "keywords": st.column_config.TextColumn("Keywords", help="Comma-separated keywords relevant to this risk (for your reference)."),
    },
    hide_index=True,
    key="risk_categories_editor_server"
)

candidate_labels = risk_categories_df["candidate_label"].tolist() if not risk_categories_df.empty else []

if not candidate_labels:
    st.warning("Please define at least one risk category label to proceed.")

# --- Step 3: Analyze Articles ---
st.header("3. Analyze Articles for Risk Triggers")

# New: Confidence Threshold Slider
user_confidence_threshold = st.slider(
    "Adjust Confidence Threshold for Multi-Label Output:",
    min_value=0.0,
    max_value=1.0,
    value=0.01, # Default value
    step=0.01,
    help="Only labels with a confidence score equal to or above this threshold will be included in the 'All Predicted Labels' column. Higher values mean fewer, but potentially more precise, labels."
)


if uploaded_file is not None and article_column is not None and candidate_labels:
    if st.button("Run Risk Analysis (Server-Side)"):
        if article_column not in articles_df.columns:
            st.error(f"The selected article column '{article_column}' does not exist in your uploaded data.")
            st.stop()

        st.info("Running classification on the server... This might take a while depending on the number of articles and your server's performance.")

        results = []
        progress_bar = st.progress(0)
        total_articles = len(articles_df)

        for i, row in articles_df.iterrows():
            article_text = str(row[article_column]) # Ensure text is string
            # Pass the user-defined threshold to the classification function
            predicted_label, confidence_score, all_labels_formatted = classify_article(
                classifier, article_text, candidate_labels, threshold=user_confidence_threshold
            )
            results.append({
                "Original Article": article_text,
                "Top Predicted Risk Label": predicted_label, # Renamed for clarity
                "Top Confidence Score": f"{confidence_score:.2f}", # Renamed for clarity
                "All Predicted Labels (Score)": all_labels_formatted # New column
            })
            progress_bar.progress((i + 1) / total_articles)

        results_df = pd.DataFrame(results)
        st.subheader("Analysis Results")
        st.dataframe(results_df)

        # Optional: Download results
        csv_output = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="risk_analysis_results_server_side_multi_label.csv", # Changed file name
            mime="text/csv",
        )

        st.subheader("Top Risk Label Distribution") # Updated title
        label_counts = results_df["Top Predicted Risk Label"].value_counts().reset_index()
        label_counts.columns = ["Risk Label", "Count"]
        st.bar_chart(label_counts.set_index("Risk Label"))

else:
    st.info("Please upload your articles and define risk categories to enable analysis.")

st.markdown("---")
st.markdown("""
**Note on Model Retraining:**
The `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` model is a pre-trained zero-shot classifier. It is *not* designed for on-the-fly retraining with a small input dataset in the traditional sense (like fine-tuning a classification head). Its strength lies in its ability to classify into *unseen* categories based on the semantic understanding it gained during its extensive pre-training on Natural Language Inference (NLI) tasks.

If you need to "retrain" or adapt it to very specific nuances of your data, you would typically:
* **Refine Candidate Labels/Descriptions:** Experiment with different phrasings for your `candidate_label` to see what works best.
* **Prompt Engineering:** For more advanced scenarios, you might try to craft more elaborate "prompts" or "hypotheses" that the ZSC model evaluates, though the `pipeline` abstracts much of this.
* **Few-Shot Learning (Advanced):** If you collect a small amount of labeled data, you could use a technique like SetFit, which leverages pre-trained sentence transformers and a small classification head. This would involve a separate training step.
""")
