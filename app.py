# immersive id="explainable_translator_flask_v2" type="code" title="app.py"
# app.py
# Flask application for an explainable translator (EN<>FR)
# with advanced rule-based linguistic explanation and a professional UI.
# Designed to be runnable locally or in Google Colab.

# --- Imports ---
print("Importing libraries...")
import os
import sys
import re
import time
import gc # Garbage Collector
import logging # For logging
from typing import List, Tuple, Dict, Optional, Any

# Set environment variable to avoid matplotlib warning in some environments
# Create a dummy config directory if it doesn't exist, useful in some minimal environments
config_dir = os.path.join(os.getcwd(), "configs_translator")
os.makedirs(config_dir, exist_ok=True)
os.environ['MPLCONFIGDIR'] = config_dir

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flag to track if essential libraries are successfully imported and models loaded
essential_libraries_loaded = True
spacy_models_installed = False # Track if spaCy models are successfully loaded


# Ensure necessary libraries are installed (for Colab/initial run)
# Note: In a production environment, you would rely on requirements.txt
# and a proper deployment setup rather than in-script installation checks.
try:
    import flask
    import torch
    import transformers
    import sentencepiece # Required by some transformers models
    import accelerate # Improves performance with transformers
    import spacy
    import pyngrok # For tunneling in Colab
    # Optional: Flask-Cors
    try:
        from flask_cors import CORS
        cors_available = True
    except ImportError:
        logger.warning("Flask-Cors not found. Install with 'pip install Flask-Cors' if needed for cross-origin requests.")
        cors_available = False


    # Check if spaCy models are installed, download if not
    try:
        # Attempt to load models to see if they are present
        spacy.load("en_core_web_sm")
        spacy.load("fr_core_news_sm")
        logger.info("spaCy models already installed.")
        spacy_models_installed = True
    except OSError:
        logger.warning("spaCy models not found. Downloading...")
        # Use subprocess or similar in a real app for better error handling
        os.system("python -m spacy download en_core_web_sm")
        os.system("python -m spacy download fr_core_news_sm")
        try:
             # Attempt to load again after download
             spacy.load("en_core_web_sm")
             spacy.load("fr_core_news_sm")
             logger.info("spaCy models downloaded and loaded.")
             spacy_models_installed = True
        except Exception as e:
             logger.error(f"Failed to load spaCy models even after attempting download: {e}")
             spacy_models_installed = False
             essential_libraries_loaded = False # SpaCy models are essential


    logger.info("Required core libraries imported successfully.")

except ImportError as e:
    logger.error(f"Error importing required libraries: {e}")
    logger.error("Please ensure all necessary libraries are installed by running: pip install -r requirements.txt")
    logger.error("Then run the spaCy model downloads: python -m spacy download en_core_web_sm and python -m spacy download fr_core_news_sm")
    essential_libraries_loaded = False # Set flag to False if imports fail
    spacy_models_installed = False # SpaCy won't work if spacy itself isn't imported
except Exception as e:
    logger.error(f"An unexpected error occurred during imports or initial checks: {e}")
    essential_libraries_loaded = False # Set flag to False on other errors
    spacy_models_installed = False


# Load spaCy models globally if imports and initial checks were successful
nlp_en = None
nlp_fr = None
if essential_libraries_loaded and spacy_models_installed:
    try:
        # Load models properly into global variables
        nlp_en = spacy.load("en_core_web_sm")
        nlp_fr = spacy.load("fr_core_news_sm")
        logger.info("spaCy models loaded into global variables.")
    except Exception as e:
        logger.error(f"Failed to load spaCy models into global variables: {e}")
        nlp_en = None
        nlp_fr = None
        essential_libraries_loaded = False # Update flag if spaCy loading fails here


# Flask setup
from flask import Flask, request, jsonify, render_template_string, render_template

app = Flask(__name__)
if cors_available: # Use the flag determined during imports
    CORS(app) # Enable CORS for all routes


logger.info("Flask app initialized.")


# --- Configuration & Constants ---

# Model identifiers from Hugging Face Hub
MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"

# --- Global Variables for Models & Tokenizers ---
# Use a dictionary to cache loaded models/tokenizers
loaded_models = {}
device_global = None

# --- Utility Functions ---
def clean_memory():
    """Force garbage collection and clear CUDA cache if available."""
    gc.collect()
    if 'torch' in sys.modules and torch.cuda.is_available():
        torch.cuda.empty_cache()
    # logger.debug("Memory cleaned.") # Optional debug message

def get_device():
    """Gets the appropriate torch device."""
    global device_global
    if device_global is None:
        # Only attempt to use CUDA/MPS if torch is available
        if 'torch' in sys.modules:
            if torch.cuda.is_available():
                device_global = torch.device("cuda")
            # Check for MPS (Apple Silicon) - uncomment if needed and PyTorch supports your MPS version
            # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            #     device_global = torch.device("mps")
            else:
                device_global = torch.device("cpu")
        else:
            device_global = torch.device("cpu") # Default to CPU if torch not available
        logger.info(f"Using device: {device_global}")
    return device_global

def load_model_and_tokenizer(model_name: str):
    """Loads a pre-trained model and tokenizer from Hugging Face, caching them."""
    global loaded_models
    # Only attempt to load models if transformers and torch are available
    if not ('transformers' in sys.modules and 'torch' in sys.modules):
        raise RuntimeError("Translation libraries (transformers/torch) not loaded. Cannot load translation model.")

    device = get_device()

    if model_name not in loaded_models:
        logger.info(f"Loading model and tokenizer for: {model_name}...")
        try:
            # Use from_pretrained with torch_dtype=torch.float16 for potential memory savings
            # This requires accelerate >= 0.20.0 and a supported GPU
            # Added error handling for model loading
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.eval() # Set to evaluation mode
            loaded_models[model_name] = {"model": model, "tokenizer": tokenizer}
            logger.info(f"{model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Re-raise the error to be caught by the calling function
            raise RuntimeError(f"Failed to load model: {model_name}. Check model name, internet connection, and available resources.") from e
    else:
        # Ensure model is on the correct device if device changed (unlikely in a simple script)
        loaded_models[model_name]["model"].to(device)
        logger.info(f"Using cached model and tokenizer for: {model_name}")

    return loaded_models[model_name]["model"], loaded_models[model_name]["tokenizer"]

# --- Linguistic Analysis and Explanation Logic (Highly Advanced Rule-Based) ---

def analyze_sentence(text: str, lang: str) -> Optional[spacy.tokens.Doc]:
    """Analyzes a sentence using spaCy."""
    # Only attempt analysis if spaCy models are loaded
    if not (spacy_models_installed and nlp_en and nlp_fr):
         logger.warning("spaCy models not available for analysis.")
         return None

    try:
        if lang == "en" and nlp_en:
            return nlp_en(text)
        elif lang == "fr" and nlp_fr:
            return nlp_fr(text)
        else:
            logger.warning(f"spaCy model for language '{lang}' not loaded or spaCy not available.")
            return None
    except Exception as e:
        logger.error(f"Error during spaCy analysis for language '{lang}': {e}")
        return None


def get_morph_features(token: spacy.tokens.Token) -> Dict[str, str]:
    """Converts spaCy token morphology to a readable dictionary."""
    return {key: str(value) for key, value in token.morph.to_dict().items()}

def describe_verb_form_rulebased(token: spacy.tokens.Token, lang: str) -> str:
    """Describes a verb's form in user-friendly terms based on morphology."""
    morph = get_morph_features(token)
    description_parts = []

    # Tense
    tense = morph.get('Tense')
    if tense and tense != 'N/A':
        description_parts.append(f"in the **{tense}** tense")

    # Mood
    mood = morph.get('Mood')
    if mood and mood != 'N/A':
         description_parts.append(f"in the **{mood}** mood")

    # Person and Number (often go together for agreement)
    person = morph.get('Person')
    number = morph.get('Number')
    if person and number and person != 'N/A' and number != 'N/A':
        if lang == 'fr':
            description_parts.append(f"conjugated for the **{person} person {number}**")
        else:
            description_parts.append(f"is in the **{person} person {number}** form")
    elif person and person != 'N/A':
         description_parts.append(f"is conjugated for the **{person} person**")
    elif number and number != 'N/A':
         description_parts.append(f"is in the **{number}** number")

    # Verb Form (e.g., infinitive, participle, finite)
    verb_form = morph.get('VerbForm')
    if verb_form and verb_form != 'N/A':
         description_parts.append(f"({verb_form} form)")

    # Aspect
    aspect = morph.get('Aspect')
    if aspect and aspect != 'N/A':
         description_parts.append(f"with '{aspect}' aspect")

    # Voice (e.g., Passive)
    voice = morph.get('Voice')
    if voice and voice != 'N/A':
        description_parts.append(f"in the **{voice}** voice")


    if not description_parts:
        return "in an unspecified form"

    return ", ".join(description_parts)

def describe_noun_features_rulebased(token: spacy.tokens.Token, lang: str) -> List[str]:
    """Describes noun features in user-friendly terms, especially for French."""
    descriptions = []
    morph = get_morph_features(token)

    if lang == 'fr':
        # French Noun Gender
        gender = morph.get("Gender")
        if gender and isinstance(gender, str):
             descriptions.append(f"This French noun is typically **{gender.lower()}**.")

        # French Noun Number
        number = morph.get("Number")
        if number and isinstance(number, str):
             descriptions.append(f"It is in the **{number}** number.")

    elif lang == 'en':
         # English Noun Number
         number = morph.get("Number")
         if number and isinstance(number, str):
             descriptions.append(f"It is in the **{number}** number.")

    if not descriptions:
         descriptions.append(f"Standard noun form in {lang}.")

    return descriptions

def describe_adjective_features_rulebased(token: spacy.tokens.Token, lang: str) -> List[str]:
    """Describes adjective features and position in user-friendly terms."""
    descriptions = []
    morph = get_morph_features(token)

    if lang == 'fr':
        # French Adjective Agreement (Gender and Number)
        gender = morph.get("Gender")
        number = morph.get("Number")
        if (gender and isinstance(gender, str)) or (number and isinstance(number, str)):
             agreement_info = []
             if gender and isinstance(gender, str): agreement_info.append(gender.lower())
             if number and isinstance(number, str): agreement_info.append(number.lower())
             if agreement_info:
                descriptions.append(f"This adjective is in the **{', '.join(agreement_info)}** form.")
                descriptions.append("In French, adjectives must agree in gender and number with the noun they modify.")

    if not descriptions:
         descriptions.append(f"Standard adjective form in {lang}.")

    return descriptions

def describe_determiner_features_rulebased(token: spacy.tokens.Token, lang: str) -> List[str]:
    """Describes determiner (article) usage and features in user-friendly terms."""
    descriptions = []
    morph = get_morph_features(token)

    if lang == 'fr':
        # French Determiner Agreement (Gender and Number)
        gender = morph.get("Gender")
        number = morph.get("Number")
        if (gender and isinstance(gender, str)) or (number and isinstance(number, str)):
             agreement_info = []
             if gender and isinstance(gender, str): agreement_info.append(gender.lower())
             if number and isinstance(number, str): agreement_info.append(number.lower())
             if agreement_info:
                descriptions.append(f"This article is in the **{', '.join(agreement_info)}** form.")
                descriptions.append("French articles agree in gender and number with the noun they precede.")

    if not descriptions:
         descriptions.append(f"Standard article usage in {lang}.")

    return descriptions

def generate_rulebased_linguistic_explanation(source_doc: spacy.tokens.Doc, target_doc: spacy.tokens.Doc, direction: str) -> str:
    """
    Generates a detailed linguistic explanation based on rule-based analysis of spaCy Docs.
    Focuses on explaining grammatical choices based on observed features and dependency relations.
    This is a highly complex rule-based system.
    """
    explanations = []

    src_lang = "English" if direction == "EN->FR" else "French"
    tgt_lang = "French" if direction == "EN->FR" else "English"
    tgt_lang_code = "fr" if direction == "EN->FR" else "en"
    src_lang_code = "en" if direction == "EN->FR" else "fr"


    explanations.append(f"**Understanding Your Translation: {src_lang} → {tgt_lang}**\n")
    explanations.append(f"Here's a detailed look at the linguistic transformations in the translation of \"{source_doc.text}\" to \"{target_doc.text}\":\n")

    # --- Verb Analysis and Conjugation ---
    tgt_verbs = [token for token in target_doc if token.pos_ == "VERB"]
    if tgt_verbs:
        explanations.append("### Verb Conjugation and Form")
        for tgt_verb in tgt_verbs:
            explanation_lines = [f"- The verb '**{tgt_verb.text}**' (root form: '{tgt_verb.lemma_}')"]

            # Describe the verb form and why
            verb_form_description = describe_verb_form_rulebased(tgt_verb, tgt_lang_code)
            explanation_lines.append(f"  * It is {verb_form_description}.")

            # Explain agreement based on dependency parse subject
            subject_token = None
            # Look for a nominal subject (nsubj) or clausal subject (csubj) among children or tokens with verb as head
            subject_candidates = [token for token in target_doc if token.head == tgt_verb and token.dep_ in ["nsubj", "csubj"]]
            if subject_candidates:
                 subject_token = subject_candidates[0] # Take the first subject found

            if subject_token:
                 subject_morph = get_morph_features(subject_token)
                 subject_person = subject_morph.get('Person', 'N/A')
                 subject_number = subject_morph.get('Number', 'N/A')
                 verb_morph = get_morph_features(tgt_verb)
                 verb_person = verb_morph.get('Person', 'N/A')
                 verb_number = verb_morph.get('Number', 'N/A')

                 agreement_notes = []
                 if subject_person != 'N/A' and verb_person != 'N/A' and subject_person == verb_person:
                      agreement_notes.append(f"agrees in person ({subject_person})")
                 elif subject_person != 'N/A' and verb_person != 'N/A':
                      agreement_notes.append(f"person changed ({subject_person} → {verb_person})")

                 if subject_number != 'N/A' and verb_number != 'N/A' and subject_number == verb_number:
                      agreement_notes.append(f"agrees in number ({subject_number})")
                 elif subject_number != 'N/A' and verb_number != 'N/A':
                      agreement_notes.append(f"number changed ({subject_number} → {verb_number})")

                 if agreement_notes:
                     explanation_lines.append(f"    * In {tgt_lang}, this verb's form {', and '.join(agreement_notes)} with its subject, '**{subject_token.text}**'.")
                 elif tgt_lang_code == 'fr' and ('Person' in verb_morph or 'Number' in verb_morph):
                      explanation_lines.append(f"    * In French, this verb form typically requires agreement in person and number with its subject, which appears to be '**{subject_token.text}**'.")

            elif tgt_lang_code == 'fr' and ('Person' in get_morph_features(tgt_verb) or 'Number' in get_morph_features(tgt_verb)):
                 # If French verb is conjugated but subject not found, note agreement is required
                 explanation_lines.append(f"    * In French, this verb form requires agreement in person and number with its subject (subject not explicitly identified by analysis).")


            # Check for common French compound tenses (e.g., Passé Composé, Futur Proche)
            if tgt_lang_code == 'fr':
                 # Check for Passé Composé (avoir/être + past participle)
                 if tgt_verb.lemma_ in ['avoir', 'être'] and tgt_verb.i + 1 < len(target_doc) and target_doc[tgt_verb.i + 1].pos_ == 'VERB' and 'VerbForm=Part' in get_morph_features(target_doc[tgt_verb.i + 1]).values():
                      explanation_lines.append(f"  * **Compound Tense:** Together with the following past participle ('{target_doc[tgt_verb.i + 1].text}'), it forms a compound tense (like the *Passé Composé*), typically used to describe past actions.")
                 # Check for Futur Proche (aller + infinitive)
                 elif tgt_verb.lemma_ == 'aller' and tgt_verb.i + 1 < len(target_doc) and target_doc[tgt_verb.i + 1].pos_ == 'VERB' and 'VerbForm=Inf' in get_morph_features(target_doc[tgt_verb.i + 1]).values():
                      explanation_lines.append(f"  * **Compound Tense:** Together with the following infinitive ('{target_doc[tgt_verb.i + 1].text}'), it forms the *Futur Proche* (near future) tense, indicating an action that will happen soon.")

            explanations.append("".join(explanation_lines))

    # --- Nouns, Adjectives, and Articles (Focus on Agreement and Usage) ---
    if (src_lang == "English" and tgt_lang == "French") or (src_lang == "French" and tgt_lang == "English"):
        tgt_nouns = [token for token in target_doc if token.pos_ in ["NOUN", "PROPN"]]
        tgt_adjs = [token for token in target_doc if token.pos_ == "ADJ"]
        tgt_dets = [token for token in target_doc if token.pos_ == "DET"]

        if tgt_nouns or tgt_adjs or tgt_dets:
             explanations.append("### Noun, Adjective, and Article Agreement and Usage (in French)")

             # Explain Nouns and their modifiers
             explained_adjs = set()
             explained_dets = set()

             for tgt_noun in tgt_nouns:
                 explanation_lines = [f"- The noun '**{tgt_noun.text}**'"]
                 noun_features = describe_noun_features_rulebased(tgt_noun, tgt_lang)
                 if noun_features:
                     explanation_lines.append("  * " + " ".join(noun_features))

                 # Find and explain modifying adjectives (using dependency parse and proximity)
                 modifying_adjs = [adj for adj in tgt_adjs if adj.head == tgt_noun or (abs(adj.i - tgt_noun.i) <= 3 and adj.dep_ in ['amod', 'acl', 'relcl'])] # Check dependency or proximity+deps
                 if modifying_adjs:
                     explanation_lines.append(f"  * It is modified by the adjective(s) **{', '.join([a.text for a in modifying_adjs])}**.")
                     if tgt_lang == "French":
                          explanation_lines.append(f"    * In French, these adjectives must agree in **gender** and **number** with the noun '**{tgt_noun.text}**'.")
                          # Add specific adjective agreement details if possible
                          for adj in modifying_adjs:
                               adj_morph = get_morph_features(adj)
                               adj_gender = adj_morph.get('Gender', 'N/A')
                               adj_number = adj_morph.get('Number', 'N/A')
                               noun_morph = get_morph_features(tgt_noun)
                               noun_gender = noun_morph.get('Gender', 'N/A')
                               noun_number = noun_morph.get('Number', 'N/A')
                               agreement_status = []
                               if adj_gender != 'N/A' and noun_gender != 'N/A' and adj_gender == noun_gender: agreement_status.append('gender agreement')
                               elif adj_gender != 'N/A' and noun_gender != 'N/A': agreement_status.append(f'gender mismatch ({adj_gender} vs {noun_gender})')
                               if adj_number != 'N/A' and noun_number != 'N/A' and adj_number == noun_number: agreement_status.append('number agreement')
                               elif adj_number != 'N/A' and noun_number != 'N/A': agreement_status.append(f'number mismatch ({adj_number} vs {noun_number})')
                               if agreement_status:
                                    explanation_lines.append(f"      - Adjective '**{adj.text}**': {', '.join(agreement_status)}.")
                               else:
                                    explanation_lines.append(f"      - Adjective '**{adj.text}**': Agreement details not fully determined by analysis.")

                          # Discuss adjective position heuristics in French
                          preceding_adjs = [adj for adj in modifying_adjs if adj.i < tgt_noun.i]
                          following_adjs = [adj for adj in modifying_adjs if adj.i > tgt_noun.i]
                          if preceding_adjs and following_adjs:
                              explanation_lines.append(f"    * Note the position: some adjectives ('{', '.join([a.text for a in preceding_adjs])}') precede the noun, while others ('{', '.join([a.text for a in following_adjs])}') follow it. Adjective position in French depends on the adjective type.")
                          elif preceding_adjs:
                               explanation_lines.append(f"    * Note the position: adjectives ('{', '.join([a.text for a in preceding_adjs])}') precede the noun. This is common for certain types of adjectives (e.g., BAGS adjectives - Beauty, Age, Goodness, Size).")
                          elif following_adjs:
                               explanation_lines.append(f"    * Note the position: adjectives ('{', '.join([a.text for a in following_adjs])}') follow the noun. This is the more common position for many descriptive adjectives in French.")


                          for adj in modifying_adjs: explained_adjs.add(adj)


                 # Find and explain modifying determiners (articles) (using dependency parse and proximity)
                 modifying_dets = [det for det in tgt_dets if det.head == tgt_noun or (abs(det.i - tgt_noun.i) <= 2 and det.dep_ in ['det'])] # Check dependency or proximity+det
                 if modifying_dets:
                     explanation_lines.append(f"  * It uses the article(s) '**{', '.join([d.text for d in modifying_dets])}**'.")
                     if tgt_lang == "French":
                          explanation_lines.append(f"    * French articles must also agree in **gender** and **number** with the noun '**{tgt_noun.text}**'.")
                          for det in modifying_dets:
                               det_morph = get_morph_features(det)
                               det_gender = det_morph.get('Gender', 'N/A')
                               det_number = det_morph.get('Number', 'N/A')
                               noun_morph = get_morph_features(tgt_noun)
                               noun_gender = noun_morph.get('Gender', 'N/A')
                               noun_number = noun_morph.get('Number', 'N/A')
                               agreement_status = []
                               if det_gender != 'N/A' and noun_gender != 'N/A' and det_gender == noun_gender: agreement_status.append('gender agreement')
                               elif det_gender != 'N/A' and noun_gender != 'N/A': agreement_status.append(f'gender mismatch ({det_gender} vs {noun_gender})')
                               if det_number != 'N/A' and noun_number != 'N/A' and det_number == noun_number: agreement_status.append('number agreement')
                               elif det_number != 'N/A' and noun_number != 'N/A': agreement_status.append(f'number mismatch ({det_number} vs {noun_number})')
                               if agreement_status:
                                    explanation_lines.append(f"      - Article '**{det.text}**': {', '.join(agreement_status)}.")
                               else:
                                    explanation_lines.append(f"      - Article '**{det.text}**': Agreement details not fully determined by analysis.")

                          # Discuss article usage (definite, indefinite, partitive, contractions)
                          for det in modifying_dets:
                               det_text_lower = det.text.lower()
                               if det_text_lower in ['le', 'la', 'l\'', 'les']:
                                    explanation_lines.append(f"    * '**{det.text}**' is a definite article, used for specific or known nouns.")
                               elif det_text_lower in ['un', 'une', 'des']:
                                    explanation_lines.append(f"    * '**{det.text}**' is an indefinite article, used for non-specific or unknown nouns.")
                               elif det_text_lower in ['du', 'de la', 'de l\'', 'des']:
                                    explanation_lines.append(f"    * '**{det.text}**' is a partitive article, used for uncountable nouns or a portion of a countable noun.")
                               elif det_text_lower in ['au', 'aux']:
                                    explanation_lines.append(f"    * '**{det.text}**' is a contraction of 'à le' or 'à les', used with the preposition 'à'.")
                               elif det_text_lower in ['du', 'des']:
                                     # Handle du/des as contractions of de + le/les if appropriate
                                     if det.head and det.head.lemma_ == 'de': # Simple check if head is 'de'
                                          explanation_lines.append(f"    * '**{det.text}**' can also be a contraction of 'de le' or 'de les', used with the preposition 'de'.")


                          for det in modifying_dets: explained_dets.add(det)

                 explanations.append("".join(explanation_lines))

             # Add explanations for adjectives/determiners not linked to a noun above
             unlinked_adjs = [adj for adj in tgt_adjs if adj not in explained_adjs]
             if unlinked_adjs:
                  explanations.append(f"- The translation also includes the {tgt_lang} adjective(s) **{', '.join([a.text for a in unlinked_adjs])}**.")
                  if tgt_lang == "French":
                       explanations.append("  * These adjectives also need to agree with the noun they modify (even if not explicitly linked by the analysis tool). Adjective position relative to nouns can differ between languages.")

             unlinked_dets = [det for det in tgt_dets if det not in explained_dets]
             if unlinked_dets:
                  explanations.append(f"- The translation uses {tgt_lang} articles like **{', '.join([d.text for d in unlinked_dets])}**.")
                  if tgt_lang == "French":
                       explanations.append("  * French articles agree in gender and number with the noun they precede. Article usage varies between languages.")


    # --- Word Order and Structural Differences ---
    explanations.append("### Word Order and Sentence Structure")
    explanations.append(f"- The arrangement of words in the sentence is adjusted to follow typical {tgt_lang} sentence structure. This can be different from {src_lang} word order.")

    # Add more specific word order comments if possible (requires more complex rule-based comparison)
    # For example, check subject-verb-object order if identifiable in both docs.
    # This is highly dependent on spaCy's parse accuracy and alignment logic.
    # A simpler addition: comment on common differences like negation placement in French.
    if tgt_lang_code == 'fr':
        negation_found = False
        for token in target_doc:
            # Simple check for common French negation particles
            if token.text.lower() in ['ne', 'n\''] and token.i + 1 < len(target_doc) and target_doc[token.i + 1].text.lower() in ['pas', 'plus', 'jamais', 'rien', 'personne']:
                 explanations.append(f"- **Negation:** French typically uses a two-part negation structure (e.g., 'ne... pas') around the verb. The translation reflects this structure.")
                 negation_found = True
                 break # Assume one negation explanation is enough per sentence

    # --- Other Notable Transformations (Prepositions, Adverbs, etc.) ---
    explanations.append("### Other Transformations")
    # Identify prepositions and their heads/children to see how phrases are constructed
    tgt_preps = [token for token in target_doc if token.pos_ == "ADP"]
    if tgt_preps:
         explanation_lines = [f"- The translation uses prepositions like **{', '.join([p.text for p in tgt_preps])}**."]
         if tgt_lang == "French":
              explanation_lines.append(" In French, prepositions are crucial for indicating relationships between words and phrases (e.g., location, time, possession).")
              # Could add examples of prepositional phrases if identifiable
         explanations.append("".join(explanation_lines))
    else:
         explanations.append("- Prepositions were used to build phrases in the translation.")


    tgt_advs = [token for token in target_doc if token.pos_ == "ADV"]
    if tgt_advs:
         explanation_lines = [f"- Adverbs like **{', '.join([a.text for a in tgt_advs])}** are used to modify verbs, adjectives, or other adverbs."]
         # Could comment on their position if it's notably different from English
         explanations.append("".join(explanation_lines))

    # Add comments on pronoun placement, especially French object pronouns
    if tgt_lang_code == 'fr':
        # Simple check for common French object pronouns preceding the verb
        pronoun_before_verb = False
        for token in target_doc:
            if token.pos_ == 'PRON' and token.i + 1 < len(target_doc) and target_doc[token.i + 1].pos_ == 'VERB' and token.dep_ in ['obj', 'iobj', 'dobj']: # Check if pronoun is object and precedes verb
                 explanations.append(f"- **Pronoun Placement:** French often places object pronouns (like '**{token.text}**') before the verb ('{target_doc[token.i + 1].text}'), which differs from English word order.")
                 pronoun_before_verb = True
                 break # Assume one pronoun placement explanation is enough

    # --- General Note ---
    explanations.append("### How This Translation Was Generated")
    explanations.append(f"- This translation was produced by a neural machine translation model. The model learned to make these linguistic transformations by analyzing millions of example sentences. It doesn't apply explicit grammatical rules like a human, but rather recognizes patterns to generate grammatically correct and natural-sounding translations in {tgt_lang}.")


    return "\n".join(explanations)


# --- Flask Routes ---

# HTML Template is now in templates/index.html

@app.route('/')
def index():
    """Renders the main HTML page from the template."""
    # Check if essential libraries are loaded before rendering
    if not essential_libraries_loaded:
         error_html = """
         <!DOCTYPE html>
         <html lang="en">
         <head>
             <meta charset="UTF-8">
             <meta name="viewport" content="width=device-width, initial-scale=1.0">
             <title>Setup Error</title>
             <script src="https://cdn.tailwindcss.com"></script>
              <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
              <style> body { font-family: 'Inter', sans-serif; } </style>
         </head>
         <body class="bg-gray-100 text-gray-900 flex items-center justify-center min-h-screen p-4">
             <div class="container mx-auto bg-white shadow-xl rounded-xl p-8 max-w-2xl w-full text-center">
                 <h1 class="text-2xl font-bold text-red-600 mb-4">Setup Error</h1>
                 <p class="text-gray-700 mb-4">The application could not start because essential libraries or spaCy models failed to load.</p>
                 <p class="text-gray-700 mb-6">Please check the output in your Colab notebook or terminal for specific error messages and installation instructions.</p>
                 <p class="text-sm text-gray-500 text-left">Ensure you have run: <br>
                 <code>!pip install Flask torch transformers sentencepiece accelerate spacy pyngrok Flask-Cors</code><br>
                 and<br>
                 <code>!python -m spacy download en_core_web_sm</code><br>
                 <code>!python -m spacy download fr_core_news_sm</code></p>
                 <p class="text-sm text-gray-500 text-left mt-4">If using ngrok for public access, ensure you have set your authtoken:<br>
                 <code>import os</code><br>
                 <code>os.environ["NGROK_AUTH_TOKEN"] = "YOUR_AUTH_TOKEN"</code></p>
             </div>
         </body>
         </html>
         """
         logger.error("Rendering setup error page due to missing essential libraries.")
         return render_template_string(error_html), 500 # Return 500 status code for server error


    # Render the HTML template from the templates folder
    return render_template('index.html')

@app.route('/translate_explain', methods=['POST'])
def translate_and_explain_route():
    """Handles translation and explanation requests from the frontend."""
    # Check if essential libraries were loaded before processing
    if not essential_libraries_loaded:
        logger.error("Received request but essential libraries are not loaded.")
        return jsonify({
            "translation": "",
            "explanation": "**Initialization Error:** Required libraries or spaCy models failed to load. Please check the server logs for installation instructions.",
            "status": "error",
            "error": "Initialization failed."
        }), 500 # Return a 500 Internal Server Error status

    data = request.get_json()
    source_text = data.get('source_text', '')
    direction = data.get('direction', 'EN->FR') # Default direction

    # Basic input sanitization (limit length)
    if len(source_text) > 1000: # Arbitrary limit
         logger.warning(f"Input text exceeds maximum length: {len(source_text)}")
         return jsonify({
              "translation": "",
              "explanation": "**Input Error:** Input text is too long. Please limit it to 1000 characters.",
              "status": "error",
              "error": "Input text too long."
         }), 400 # Bad Request

    # Call the backend logic function
    result = get_translation_and_explanation(source_text, direction)

    # Return the result as JSON
    return jsonify(result)


# --- Backend Logic Function (Called by Flask Route) ---
def get_translation_and_explanation(source_text: str, direction: str) -> Dict[str, Any]:
    """
    Performs translation and generates a rule-based linguistic explanation.
    Returns a dictionary suitable for a JSON response.
    """
    logger.info(f"Processing request: Direction={direction}, Input='{source_text[:50]}...'") # Log first 50 chars

    # Initialize outputs
    translation_output = ""
    textual_explanation = "Processing..."
    status = "success"
    error_message = None

    # Input validation (basic checks already done in route, but double-check)
    if not source_text.strip():
        logger.warning("Received empty input text.")
        status = "error"
        error_message = "Please enter some text to translate and analyze."
        textual_explanation = error_message
        return {"translation": translation_output, "explanation": textual_explanation, "status": status, "error": error_message}

    if direction not in ["EN->FR", "FR->EN"]:
        logger.warning(f"Received invalid direction: {direction}")
        status = "error"
        error_message = "Invalid translation direction selected."
        textual_explanation = error_message
        return {"translation": translation_output, "explanation": textual_explanation, "status": status, "error": error_message}

    source_lang_code = "en" if direction == "EN->FR" else "fr"
    target_lang_code = "fr" if direction == "EN->FR" else "en" # Corrected target lang code logic
    source_lang_name = "English" if direction == "EN->FR" else "French"
    target_lang_name = "French" if direction == "EN->FR" else "English"


    start_time = time.time()
    translation_successful = False
    current_translation_output = ""

    try:
        # 1. Load appropriate model and tokenizer
        # Check if transformers and torch are available before loading model
        if 'transformers' not in sys.modules or 'torch' not in sys.modules:
             logger.error("Translation libraries (transformers/torch) not loaded.")
             raise RuntimeError("Translation libraries (transformers/torch) not loaded.")

        model_name = MODEL_EN_FR if direction == "EN->FR" else MODEL_FR_EN
        model, tokenizer = load_model_and_tokenizer(model_name)
        device = get_device() # Ensure device is known

        # 2. Prepare input
        # Add language code prefix for Helsinki-NLP models
        # See https://huggingface.co/Helsinki-NLP/opus-mt-en-fr#translation-example
        if direction == "FR->EN":
             # For FR->EN, the source text needs the >>en<< prefix
             input_text_processed = f">>en<< {source_text}"
        else: # EN->FR
             # For EN->FR, the source text needs the >>fr<< prefix
             input_text_processed = f">>fr<< {source_text}"

        inputs = tokenizer(input_text_processed, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 3. Generate translation
        logger.info("Generating translation...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=4, # Use beam search (common default for generation)
                early_stopping=True,
                output_scores=False,
                return_dict_in_generate=True
            )

        # 4. Decode generated sequence
        generated_ids = outputs.sequences[0]
        current_translation_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Translation generated: '{current_translation_output[:50]}...'")
        translation_successful = True

    except RuntimeError as e:
         logger.error(f"Runtime Error during translation: {e}")
         status = "error"
         error_message = f"Error during translation: {e}"
         current_translation_output = "" # Clear output on error
         textual_explanation = error_message
    except Exception as e:
        logger.error(f"An unexpected error occurred during translation: {e}")
        status = "error"
        error_message = f"An unexpected error occurred during translation: {e}"
        current_translation_output = "" # Clear output on error
        textual_explanation = error_message


    # 5. Generate Detailed Linguistic Explanation (only if translation was successful or partially successful)
    if translation_successful:
        logger.info("Performing linguistic analysis and generating explanation...")
        # Check if spaCy models are available before analyzing
        if not (spacy_models_installed and nlp_en and nlp_fr):
             status = "warning" # Analysis failed, but translation might be OK
             error_message = "Linguistic analysis libraries (spaCy) or models not loaded. Explanation will be unavailable."
             textual_explanation = error_message # Provide a message in the explanation field
             logger.warning("Skipping linguistic analysis due to missing spaCy models.")
        else:
            source_doc = analyze_sentence(source_text, source_lang_code)
            target_doc = analyze_sentence(current_translation_output, target_lang_code)

            if source_doc is None or target_doc is None:
                 status = "warning" # Analysis failed, but translation might be OK
                 error_message = "Could not perform detailed linguistic analysis (spaCy analysis failed). Explanation will be unavailable."
                 textual_explanation = error_message # Provide a message in the explanation field
                 logger.warning("Skipping linguistic analysis due to spaCy analysis failure.")
            else:
                # Generate explanation using rule-based logic
                textual_explanation = generate_rulebased_linguistic_explanation(
                    source_doc,
                    target_doc,
                    direction
                )
                logger.info("Rule-based linguistic explanation generated.")
    else:
         # If translation failed, provide a simple error explanation
         textual_explanation = f"Translation failed: {error_message}"


    end_time = time.time()
    logger.info(f"Request processed in {end_time - start_time:.2f} seconds. Status: {status}")
    clean_memory() # Clean up memory after processing

    # Return results as a dictionary
    return {
        "translation": current_translation_output,
        "explanation": textual_explanation,
        "status": status,
        "error": error_message
    }


# --- Running the Flask App (for Colab/Local) ---
if __name__ == '__main__':
    # Only attempt to run the app if essential libraries were loaded
    if not essential_libraries_loaded:
        logger.error("\nFlask app cannot start due to missing essential libraries.")
        logger.error("Please install them and try again.")
        # Render the error page directly if Flask is available but other libraries are not
        if 'flask' in sys.modules:
             app.run(host='0.0.0.0', port=5000, debug=True) # Still run to show error page
    else:
        try:
            # Use pyngrok to expose the port
            from pyngrok import ngrok
            import os # Ensure os is imported here for environment variable access

            # Terminate any existing ngrok tunnels
            ngrok.kill()
            logger.info("Terminated existing ngrok tunnels.")

            # Get ngrok authtoken from environment variable
            NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")

            if NGROK_AUTH_TOKEN:
                # Set the authtoken
                ngrok.set_auth_token(NGROK_AUTH_TOKEN)
                logger.info("ngrok authtoken set.")
                # Start a new ngrok HTTP tunnel on port 5000
                # Using a try-except block specifically for ngrok connection issues
                try:
                     public_url = ngrok.connect(5000)
                     logger.info(f"\n🎉 Flask app is running! Access it at: {public_url}\n")
                except Exception as ngrok_e:
                     logger.error(f"\nError starting ngrok tunnel: {ngrok_e}")
                     logger.error("Please ensure ngrok is installed and your authtoken is valid.")
                     public_url = None # Set public_url to None if tunnel fails


                logger.info("Starting Flask app...")
                # Set debug=True for development, False for production
                # host='0.0.0.0' makes the server accessible externally (needed for ngrok/colab_tunnel)
                # port=5000 is a common default Flask port
                # Use threaded=True to allow ngrok thread to run alongside Flask development server
                # Use use_reloader=False when running with pyngrok to avoid issues
                app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
                logger.info("Flask app stopped.")
            else:
                logger.warning("\nNGROK_AUTH_TOKEN environment variable not set.")
                logger.warning("Please get your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
                logger.warning("And set it in Colab using: import os; os.environ['NGROK_AUTH_TOKEN'] = 'YOUR_AUTH_TOKEN'")
                logger.warning("Flask app will run locally but won't be publicly accessible without ngrok.")
                # Run locally if ngrok authtoken is not set
                app.run(host='0.0.0.0', port=5000, debug=True)


        except ImportError:
             logger.error("\nCould not import pyngrok. Please install it using: pip install pyngrok")
             logger.error("Flask app will run locally but won't be publicly accessible without a tunnel.")
             # Fallback to running without tunnel if pyngrok is not available
             app.run(host='0.0.0.0', port=5000, debug=True)
        except Exception as e:
             logger.error(f"\nAn unexpected error occurred while starting the Flask app: {e}")
             logger.error("Please check the error message and ensure the port (5000) is not already in use.")


