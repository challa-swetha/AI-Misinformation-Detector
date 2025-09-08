# --- Final Application File for Hugging Face Deployment ---
# This script is a self-contained Gradio application. It's designed to be robust,
# handling potential file loading and API key errors gracefully.

# 1. Standard Library Imports
import os
import json
import warnings
from urllib.parse import urlparse

# 2. Third-Party Library Imports
import pandas as pd
import numpy as np
import torch
import gradio as gr
import google.generativeai as genai
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration & Initialization ---

# 3. Securely Load API Keys from Hugging Face Secrets
# This is the recommended way to handle sensitive keys.
# The user must set these in the "Settings" > "Secrets" section of their Space.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

# Configure the Google Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("WARNING: GOOGLE_API_KEY secret not found. The final analysis will fail.")

# 4. Define File Paths and Load Data
# On Hugging Face Spaces, all uploaded files are in the current directory ('.')
SAVE_PATH = "."
EVIDENCE_DF_PATH = os.path.join(SAVE_PATH, 'evidence_dataframe.csv')
EMBEDDINGS_PATH = os.path.join(SAVE_PATH, 'evidence_text_embeddings.npy')

# Global variable to hold the analyzer and track loading status
analyzer = None
LOADING_ERROR_MESSAGE = ""

# --- Core Application Logic ---

# 5. The Core Analysis Engine (Class and Functions)
class MisinformationAnalyzer:
    """A class to handle internal knowledge base lookups and source reputation checks."""
    def __init__(self, evidence_df, text_embeddings):
        self.device = "cpu"  # Use CPU for compatibility on free HF Spaces
        self.evidence_df = evidence_df
        self.evidence_texts = self.evidence_df['text'].tolist()
        self.evidence_text_embeddings = text_embeddings
        self.text_retrieval_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.source_reputation = {'reuters.com': 'High', 'apnews.com': 'High', 'bbc.com': 'High'}

    def check_source_reputation(self, url):
        if not url or not isinstance(url, str) or not url.startswith('http'):
            return "Not Applicable (No link provided)"
        try:
            domain = urlparse(url).netloc.replace('www.', '')
            return self.source_reputation.get(domain, "Neutral/Unknown")
        except:
            return "Invalid URL."

    def retrieve_text_evidence(self, claim, top_k=3):
        query_embedding = self.text_retrieval_model.encode(claim, device=self.device)
        hits = semantic_search(query_embedding, self.evidence_text_embeddings, top_k=top_k)
        if not hits or not hits[0]:
            return []
        return [{'text': self.evidence_texts[hit['corpus_id']], 'score': f"{hit['score']:.2f}"} for hit in hits[0]]

    def run_initial_analysis(self, text_claim, article_url=None):
        return {
            "input_claim": text_claim,
            "source_reputation": self.check_source_reputation(article_url),
            "text_evidence": self.retrieve_text_evidence(text_claim)
        }

# --- External API Functions ---
def fact_check_with_search(claim):
    """Uses SerpApi to perform a targeted search on official sources."""
    if not SERPAPI_API_KEY:
        return {"status": "Error", "details": "SERPAPI_API_KEY is not set."}
    try:
        params = {"api_key": SERPAPI_API_KEY, "q": f'"{claim}" official government announcement site:gov.in'}
        results = GoogleSearch(params).get_dict()
        if "organic_results" in results and results.get("organic_results"):
            return {"status": "Evidence Found", "details": f"Found: '{results['organic_results'][0]['title']}'"}
        return {"status": "No Evidence Found", "details": "Targeted search on official sites found no matching results."}
    except Exception as e:
        return {"status": "Error", "details": str(e)}

def verify_image_context(image_url):
    """Uses SerpApi to perform a reverse image search."""
    if not image_url or not image_url.startswith('http'):
        return {"status": "No Image Provided", "details": ""}
    if not SERPAPI_API_KEY:
        return {"status": "Error", "details": "SERPAPI_API_KEY is not set."}
    try:
        params = {"engine": "google_reverse_image", "image_url": image_url, "api_key": SERPAPI_API_KEY}
        results = GoogleSearch(params).get_dict()
        if "image_results" in results and results.get("image_results"):
            return {"status": "Context Found", "details": f"Image related to: '{results['image_results'][0].get('title', 'N/A')}'."}
        return {"status": "No Context Found", "details": "Could not find other instances of this image."}
    except Exception as e:
        return {"status": "Error", "details": str(e)}

def get_final_verdict(initial_report, fact_check_data, image_data):
    """Uses Google Gemini to synthesize the findings into a final verdict."""
    if not GOOGLE_API_KEY:
        return {"verdict": "Error", "credibility_score": 0, "explanation": "GOOGLE_API_KEY is not set.", "red_flags": []}

    llm = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""You are an AI misinformation analyst for users in India. Provide a clear verdict and educate the user based on the following evidence.
    **Fact-Check:** {fact_check_data['status']} - {fact_check_data['details']}
    **Image Verification:** {image_data['status']} - {image_data['details']}
    **Source Reputation:** {initial_report['source_reputation']}
    **Claim:** \"{initial_report['input_claim']}\"
    **Instructions:**
    1. **Verdict:** "False," "Misleading," or "Credible." Prioritize the Fact-Check. If it's "No Evidence Found," the claim is **False**.
    2. **Credibility Score:** 0-100. "False" must be below 10.
    3. **Explanation:** Simply explain WHY the claim is false/misleading, using the evidence.
    4. **Red Flags:** Choose relevant flags from ["Directly Contradicted by Fact-Check", "Image Used Out of Context", "Source Reputation Not Applicable", "No Corroborating Evidence"].
    5. Return a clean JSON object: {{"verdict": "...", "credibility_score": ..., "explanation": "...", "red_flags": [{{"flag": "...", "details": "..."}}]}}
    """
    try:
        response = llm.generate_content(prompt)
        cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)
    except Exception as e:
        return {"verdict": "Error", "credibility_score": 0, "explanation": f"LLM Error: {str(e)}", "red_flags": []}


# --- Main Application Logic ---

try:
    print("--- Loading knowledge base files... ---")
    if not os.path.exists(EVIDENCE_DF_PATH):
        raise FileNotFoundError(f"Required file not found: {EVIDENCE_DF_PATH}. Please upload it to the Space.")
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Required file not found: {EMBEDDINGS_PATH}. Please upload it to the Space.")

    # Load the data into memory
    evidence_df_data = pd.read_csv(EVIDENCE_DF_PATH)
    text_embeddings_data = np.load(EMBEDDINGS_PATH)

    # Initialize the analyzer ONCE
    analyzer = MisinformationAnalyzer(evidence_df=evidence_df_data, text_embeddings=text_embeddings_data)
    print("✅ Analyzer initialized successfully.")
except Exception as e:
    LOADING_ERROR_MESSAGE = str(e)
    print(f"❌ CRITICAL ERROR during initialization: {LOADING_ERROR_MESSAGE}")


def full_analysis_pipeline(claim_text, image_url, article_url):
    """This function runs the entire pipeline from input to final report."""
    # Check if the app failed to load critical files
    if LOADING_ERROR_MESSAGE:
        return f"## 🔴 Application Error\n**The application could not start correctly.**\n\n**Error details:** {LOADING_ERROR_MESSAGE}\n\nPlease check the application logs or contact the administrator."
    if not claim_text:
        return "## 🟡 Warning\nPlease enter a claim in the text box to start the analysis."

    print(f"\n--- Starting Analysis for: '{claim_text[:50]}...' ---")
    initial_report = analyzer.run_initial_analysis(claim_text, article_url)
    fact_check_data = fact_check_with_search(claim_text)
    image_data = verify_image_context(image_url)
    final_verdict = get_final_verdict(initial_report, fact_check_data, image_data)

    # Format the report for display
    report_markdown = f"## Verdict: {final_verdict.get('verdict', 'N/A')}\n"
    report_markdown += f"### Credibility Score: **{final_verdict.get('credibility_score', 'N/A')}/100**\n\n"
    report_markdown += f"**Explanation:** {final_verdict.get('explanation', '')}\n\n"
    report_markdown += "---\n\n### Red Flags Identified\n\n"
    
    red_flags = final_verdict.get('red_flags', [])
    if not red_flags:
        report_markdown += "- No specific red flags were automatically identified.\n"
    else:
        for flag in red_flags:
            flag_name = flag.get('flag', 'Unknown Flag')
            details = flag.get('details', 'No details provided.')
            report_markdown += f"#### 🚩 {flag_name}\n- **Details:** {details}\n\n"
    
    print("--- Analysis Complete ---")
    return report_markdown


# --- Gradio Web Interface Definition ---
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # 🇮🇳 AI-Powered Misinformation Detector
        This tool helps you verify information by analyzing text, checking official sources, and verifying images to provide a credibility score and verdict.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            claim_input = gr.Textbox(lines=4, label="Text Claim", placeholder="Paste the text from the social media post or WhatsApp message here...")
            image_url_input = gr.Textbox(label="Image URL (Optional)", placeholder="Paste the link to the image if there is one...")
            article_url_input = gr.Textbox(label="Article URL (Optional)", placeholder="Paste the link to the news article if there is one...")
            analyze_button = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=3):
            output_report = gr.Markdown(label="Analysis Report", value="Your report will appear here...")

    analyze_button.click(
        fn=full_analysis_pipeline,
        inputs=[claim_input, image_url_input, article_url_input],
        outputs=output_report
    )

# This line allows the script to be run directly and is required for Hugging Face Spaces
if __name__ == "__main__":
    iface.launch()

