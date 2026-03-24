import csv
import io
import re
import sys
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import random
import os

# ================== IMPORTS FOR ML PIPELINE ==================
from collections import Counter
from sklearn.pipeline import make_pipeline , Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity , linear_kernel

# ================== NEW IMPORTS FOR MACRO-F1 OPTIMIZATION ==================
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold , ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModel , AutoTokenizer
from safetensors.torch import load_file
from sklearn.utils import resample
from joblib import Parallel, delayed

class HybridLegalModel(nn.Module):
    def __init__(self, n_classes=3, n_meta_features=2):
        super().__init__()

        # Base encoder (NO loading of weights here)
        self.bert = AutoModel.from_pretrained(
            "nlpaueb/legal-bert-base-uncased",
            use_safetensors=True,        # 🔐 avoid .bin loading
            trust_remote_code=False
        )

        self.dropout = nn.Dropout(0.3)

        self.meta_layer = nn.Sequential(
            nn.Linear(n_meta_features, 16),
            nn.ReLU()
        )

        self.classifier = nn.Linear(768 + 16, n_classes)

    def forward(self, input_ids, attention_mask, meta_data):
        pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output

        pooled = self.dropout(pooled)
        meta_out = self.meta_layer(meta_data)

        combined = torch.cat([pooled, meta_out], dim=1)
        return self.classifier(combined)
    
def save_model_safely(model, path="best_hybrid_model.safetensors"):
    """
    Secure, CVE-safe model saving
    """
    model.eval()
    save_file(model.state_dict(), path)
    print(f"✅ Model safely saved to {path}")

def load_model_safely(
    model_path="best_hybrid_model.safetensors",
    n_classes=3,
    n_meta_features=2,
    device="cpu"
):
    """
    CVE-safe, production-ready loader
    """

    # ---- Sanity check ----
    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    if torch_version < (2, 0):
        raise RuntimeError("PyTorch >= 2.0 required")

    # ---- Tokenizer (safe) ----
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpaueb/legal-bert-base-uncased",
        use_safetensors=True,
        trust_remote_code=False
    )

    # ---- Model ----
    model = HybridLegalModel(
        n_classes=n_classes,
        n_meta_features=n_meta_features
    )

    # ---- Safe weight loading ----
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    return tokenizer, model

@torch.no_grad()
def predict_verdict(tokenizer, model, text, device="cpu"):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    meta_data = torch.zeros((1, 2), device=device)

    logits = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        meta_data=meta_data
    )

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    labels = [
        "Against Accused (Prosecution Wins)",
        "In Favour of Accused",
        "Other / Procedural"
    ]

    return {
        "verdict": labels[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "distribution": dict(zip(labels, probs))
    }

# ================== SESSION STATE INITIALIZATION ==================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ================== MAIN GRID ==================
row1_left, row1_right = st.columns(2, gap="large")
row2_left, row2_right = st.columns(2, gap="large")
row3 = st.container()

def display_per_class_performance(eval_data, y_test_distribution):
    """
    Displays per-class Recall, Precision, F1, Support
    BELOW Phase-1 KPI cards (main panel).
    """
    if not eval_data:
        return

    per_class = eval_data.get("per_class_metrics", {})
    if not per_class:
        st.info("Per-class metrics unavailable.")
        return

    st.markdown("### 📌 Per-Class Performance")

    for cls, metrics in per_class.items():
        recall = metrics.get("recall", 0) * 100
        f1 = metrics.get("f1", 0) * 100
        support = y_test_distribution.get(cls, 0)

        with st.container():
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #e5e7eb;
                    border-radius:10px;
                    padding:15px;
                    margin-bottom:12px;
                ">
                    <h4 style="margin-bottom:10px;">{cls}</h4>
                    <div style="display:flex; gap:30px; font-size:0.95rem;">
                        <div><b>Recall</b><br>{recall:.1f}%</div>
                        <div><b>F1</b><br>{f1:.1f}%</div>
                        <div><b>Support</b><br>{support}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

def get_tfidf_from_pipeline(pipeline):
    """
    Safely extract TF-IDF vectorizer from a sklearn pipeline,
    regardless of whether it was created via make_pipeline or Pipeline.
    """
    for name, step in pipeline.named_steps.items():
        if isinstance(step, TfidfVectorizer):
            return step
    raise KeyError("TF-IDF vectorizer not found in pipeline.")

def phase1_kpi_cards(
    test_acc,
    test_macro_f1,
    cv_macro_f1,
    substantive_pct
):
    st.markdown(
        """
        <style>
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 25px;
        }
        .kpi-card {
            background: linear-gradient(135deg, #6b5cff, #8b5cf6);
            color: white;
            padding: 18px 20px;
            border-radius: 14px;
            box-shadow: 0 8px 20px rgba(99,102,241,0.35);
            min-height: 120px;
        }
        .kpi-title {
            font-size: 0.85rem;
            font-weight: 600;
            opacity: 0.95;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 0.7rem;
            opacity: 0.85;
            margin-top: 6px;
        }
        @media (max-width: 1200px) {
            .kpi-container {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-title">Test Accuracy</div>
                <div class="kpi-value">{test_acc:.1f}%</div>
                <div class="kpi-sub">Traditional metric (can be misleading)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Test Macro-F1</div>
                <div class="kpi-value">{test_macro_f1:.1f}%</div>
                <div class="kpi-sub">Primary fairness metric</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">CV Macro-F1 Avg</div>
                <div class="kpi-value">{cv_macro_f1:.1f}%</div>
                <div class="kpi-sub">5-fold cross-validation</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Substantive Cases</div>
                <div class="kpi-value">{substantive_pct:.1f}%</div>
                <div class="kpi-sub">Minority class percentage</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def has_minimum_classes(labels, min_classes=2):
    return len(set(labels)) >= min_classes

def normalize_verdict(v):
    v = str(v).lower().strip()
    if any(x in v for x in ["convicted", "guilty", "liable"]):
        return "Against Accused (Prosecution Wins)"
    if any(x in v for x in ["acquitted", "not guilty", "allowed", "quashed"]):
        return "In Favour of Accused"
    return "Other / Procedural"

def is_procedural_case(order_text: str = "", acts_text: str = "") -> bool:
    """
    Determines whether a case is procedural (non-substantive),
    meaning no final adjudication on guilt/innocence.
    """
    text = f"{order_text} {acts_text}".lower()
    procedural_markers = [
        "notice issued",
        "rule issued",
        "interim order",
        "adjourned",
        "listed for hearing",
        "disposed of",
        "dismissed as withdrawn",
        "procedural",
        "directions issued",
        "status report",
        "compliance report",
        "transfer petition",
        "review petition",
        "leave granted",
        "sine die",
        "costs imposed",
        "liberty granted",
        "registry directed",
        "pending",
        "next date",
    ]

    # If any strong conviction/acquittal markers exist → NOT procedural
    substantive_markers = [
        "convicted",
        "acquitted",
        "not guilty",
        "sentence",
        "sentenced to",
        "charges quashed",
        "appeal allowed",
        "appeal dismissed on merits",
        "set aside conviction",
        "ipc 302",
        "ipc 376",
        "ipc 307",
        "ipc 420",
    ]
    if any(x in text for x in substantive_markers):
        return False
    return any(x in text for x in procedural_markers)

# ================== NEW UNIFIED LABELING FUNCTION ==================
# def get_macro_label(verdict_raw, order_text="", acts_text=""):
#     """
#     Unified hierarchical labeling strategy.
#     Returns one of three classes with priority: Substantive > Procedural.
#     """
#     if pd.isna(verdict_raw) or str(verdict_raw).strip() == "":
#         return "Other / Procedural"
    
#     combined = f"{str(verdict_raw).lower()} {str(order_text).lower()} {str(acts_text).lower()}"
    
#     # 1. Check for substantive markers first (AGAINST ACCUSED)
#     against_keywords = [
#         "convicted", "guilty", "sentence imposed", "sentenced to",
#         "conviction upheld", "appeal dismissed on merits",
#         "liable", "decreed", "injunction granted", "refund ordered",
#         "ipc 302", "ipc 376", "ipc 307", "ipc 420", "ipc 304",
#         "murder", "rape", "cheating", "theft"
#     ]
    
#     for kw in against_keywords:
#         if kw in combined:
#             return "Against Accused (Prosecution Wins)"
    
#     # 2. Check for substantive markers (IN FAVOUR)
#     favor_keywords = [
#         "acquitted", "not guilty", "charges quashed", "fir quashed",
#         "proceedings quashed", "appeal allowed", "set aside conviction",
#         "benefit of doubt", "relief granted", "allowed", "quashed",
#         "acquittal", "exonerated"
#     ]
    
#     for kw in favor_keywords:
#         if kw in combined:
#             return "In Favour of Accused"
    
#     # 3. Check procedural markers
#     procedural_keywords = [
#         "notice issued", "rule issued", "interim order", "adjourned",
#         "listed for hearing", "disposed of", "dismissed as withdrawn",
#         "directions issued", "status report", "compliance report",
#         "transfer petition", "review petition", "leave granted",
#         "sine die", "costs imposed", "liberty granted", "registry directed",
#         "pending", "next date", "hearing", "order reserved"
#     ]
    
#     for kw in procedural_keywords:
#         if kw in combined:
#             return "Other / Procedural"
    
#     # 4. Default to procedural
#     return "Other / Procedural"

# ================== NEW: ADVANCED RESAMPLING FOR MACRO-F1 ==================
def advanced_resampling_strategy(train_rows, random_state=42):
    """
    Advanced resampling strategy that:
    1. Uses square root weighting for class importance
    2. Doesn't aim for perfect equality (reduces overfitting)
    3. Handles extreme imbalance better
    """
    # Separate by class
    procedural = [r for r in train_rows if r["label"] == "Other / Procedural"]
    against = [r for r in train_rows if r["label"] == "Against Accused (Prosecution Wins)"]
    favor = [r for r in train_rows if r["label"] == "In Favour of Accused"]
    
    if not procedural:
        return train_rows
    
    # Calculate target sizes using square root weighting (less aggressive)
    majority_size = len(procedural)
    
    # Square root weighting: minority classes get sqrt(imbalance_ratio) * current_size
    imbalance_ratio = majority_size / (len(against) + 1e-6)
    target_against = int(len(against) * np.sqrt(min(imbalance_ratio, 10)))  # Cap at 10x
    
    imbalance_ratio = majority_size / (len(favor) + 1e-6)
    target_favor = int(len(favor) * np.sqrt(min(imbalance_ratio, 10)))  # Cap at 10x
    
    # But don't exceed reasonable limits
    max_minority_size = int(majority_size * 0.4)  # Minorities at most 40% of majority
    target_against = min(target_against, max_minority_size)
    target_favor = min(target_favor, max_minority_size)
    
    # Ensure minimum samples for training
    target_against = max(target_against, 5)
    target_favor = max(target_favor, 5)
    
    # Upsample minority classes
    if against and len(against) < target_against:
        against = resample(
            against,
            replace=True,
            n_samples=target_against,
            random_state=random_state
        )
    
    if favor and len(favor) < target_favor:
        favor = resample(
            favor,
            replace=True,
            n_samples=target_favor,
            random_state=random_state
        )
    
    # Under-sample majority if needed (but keep it majority)
    target_procedural = int(majority_size * 0.7)  # Keep 70% of original
    if len(procedural) > target_procedural:
        procedural = resample(
            procedural,
            replace=False,
            n_samples=target_procedural,
            random_state=random_state
        )
    
    # Combine
    balanced_data = procedural + against + favor
    random.shuffle(balanced_data)
    
    return balanced_data

# ================== NEW: MACRO-F1 OPTIMIZED MODEL TRAINING ==================
# def train_macro_f1_optimized_model(X_train, y_train):
#     """
#     Train model with explicit macro-F1 optimization using GridSearchCV
#     """
#     # Calculate class weights for macro-F1
#     class_counts = Counter(y_train)
#     total = sum(class_counts.values())
    
#     # Square root weighting (less aggressive than pure inverse)
#     class_weights = {}
#     for cls, count in class_counts.items():
#         if count > 0:
#             weight = np.sqrt(total / (len(class_counts) * count))
#         else:
#             weight = 1.0
#         class_weights[cls] = weight
    
#     # Create pipeline
#     from sklearn.pipeline import Pipeline
    
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer(
#             ngram_range=(1, 3),
#             min_df=2,
#             max_features=10000,
#             sublinear_tf=True
#         )),
#         ('clf', LogisticRegression(
#             max_iter=1000,
#             random_state=42
#         ))
#     ])
    
#     # Parameter grid for macro-F1 optimization
#     param_grid = {
#         'clf__C': [0.01, 0.1, 1.0],
#         'clf__class_weight': [class_weights, 'balanced', None],
#         'clf__solver': ['liblinear', 'saga'],
#         'clf__penalty': ['l1', 'l2'],
#         'tfidf__max_features': [5000, 10000, 15000]
#     }
    
#     # Use StratifiedKFold for cross-validation with macro-F1 scoring
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
#     # Grid search with macro-F1 scoring
#     grid_search = GridSearchCV(
#         pipeline,
#         param_grid,
#         cv=cv,
#         scoring='f1_macro',
#         n_jobs=-1,
#         verbose=0
#     )
    
#     # Fit the model
#     grid_search.fit(X_train, y_train)
    
#     return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
def train_macro_f1_optimized_model(X_text, y_train):
    """
    Fully parallelized macro-F1 optimization
    WITHOUT repeated TF-IDF computation
    """

    # -------- 1. TF-IDF ONCE --------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )
    X_vec = vectorizer.fit_transform(X_text)

    # -------- 2. Class weights --------
    class_counts = Counter(y_train)
    total = sum(class_counts.values())
    class_weights = {
        cls: np.sqrt(total / (len(class_counts) * cnt))
        for cls, cnt in class_counts.items()
    }

    # -------- 3. Parameter grid --------
    param_grid = list(ParameterGrid({
        "C": [0.01, 0.1, 1.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],   # <-- PARALLEL FRIENDLY
        "class_weight": [class_weights, "balanced"]
    }))

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # -------- 4. Parallel model evaluation --------
    def eval_one(params):
        clf = LogisticRegression(
            max_iter=1000,
            n_jobs=1,      # IMPORTANT: avoid nested parallelism
            **params
        )

        scores = []
        for tr_idx, val_idx in skf.split(X_vec, y_train):
            clf.fit(X_vec[tr_idx], np.array(y_train)[tr_idx])
            preds = clf.predict(X_vec[val_idx])
            scores.append(
                f1_score(np.array(y_train)[val_idx], preds, average="macro")
            )

        return np.mean(scores), params, clf

    results = Parallel(
        n_jobs=os.cpu_count(),
        backend="loky",   # TRUE parallelism
        verbose=10
    )(delayed(eval_one)(p) for p in param_grid)

    # -------- 5. Select best --------
    best_score, best_params, _ = max(results, key=lambda x: x[0])

    # -------- 6. Train final model --------
    final_clf = LogisticRegression(
        max_iter=1000,
        n_jobs=os.cpu_count(),
        **best_params
    )
    final_clf.fit(X_vec, y_train)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", final_clf)
    ])

    return pipeline, best_params, best_score

# ================== NEW: HONEST CV WITH MACRO-F1 FOCUS ==================
def perform_macro_f1_cv(train_rows_raw, n_splits=5):
    """
    Phase-1 FAST diagnostic CV
    - Fixed parameters
    - No GridSearch
    - No resampling
    - Macro-F1 focused
    """

    X = np.array([r["text"] for r in train_rows_raw])
    y = np.array([r["label"] for r in train_rows_raw])

    # Fixed, defensible Phase-1 baseline
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=8000,
        sublinear_tf=True
    )

    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(
        C=0.01,
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        clf,
        X_vec,
        y,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1
    )

    return scores.tolist(), [{"fold": i+1, "cv_macro_f1": s} for i, s in enumerate(scores)]

# Now you can define your class
class HybridLegalModel(torch.nn.Module):
    def __init__(self, n_classes=3, n_meta_features=2): 
        super(HybridLegalModel, self).__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.drop = torch.nn.Dropout(p=0.3)
        self.meta_layer = torch.nn.Linear(n_meta_features, 16)
        self.meta_relu = torch.nn.ReLU()
        # 768 is the standard hidden size for BERT-base
        self.out = torch.nn.Linear(768 + 16, n_classes) 
    def forward(self, input_ids, attention_mask, meta_data):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output_text = self.drop(pooled_output)
        output_meta = self.meta_layer(meta_data)
        output_meta = self.meta_relu(output_meta)
        combined = torch.cat((output_text, output_meta), dim=1)
        return self.out(combined)

# ================== CONFIGURATION ==================
# PASSWORD = "judicia123"
# def password_gate():
#     pw = st.text_input("Enter password:", type="password")
#     if pw != PASSWORD:
#         st.stop()
# password_gate()
# Place this near your imports in app.py
    
def safe_strip(val):
    """
    Safely converts any value (NaN, float, int, None) to a clean string.
    """
    if val is None:
        return ""
    if isinstance(val, float):
        if np.isnan(val):
            return ""
    return str(val).strip()

# ================== PHASE-2 BIAS LEXICON ==================
BIAS_KEYWORDS = {
    "caste": [
        "caste", "dalit", "tribe", "tribal",
        "backward", "backward class", "scheduled caste", "scheduled tribe",
        "sc", "st", "obc", "other backward class"
    ],
    "religion": [
        "hindu", "muslim", "christian", "sikh", "parsi", "jain",
        "religion", "faith", "minority", "community"
    ],
    "gender": [
        "male", "female", "man", "woman", "boy", "girl",
        "gender", "sex", "transgender", "husband", "wife"
    ],
    "economic": [
        "rich", "poor", "elite", "uneducated", "underprivileged",
        "economically", "weaker", "section"
    ],
}

# ================== PHASE-2 TRANSFORMER LOADING + IG ATTRIBUTION ==================
# Note: Ensure torch, transformers, and captum are installed
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from captum.attr import LayerIntegratedGradients
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    st.error("Missing dependencies: torch, transformers, or captum. Phase-2 features will be disabled.")

TRANSFORMER_MODEL_PATH = "/Users/daysh/Desktop/College/SHAP/legal_verdict_transformer"

@st.cache_resource
def load_transformer_model():
    if not TRANSFORMER_AVAILABLE:
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpaueb/legal-bert-base-uncased"
        )

        model = HybridLegalModel(n_classes=3, n_meta_features=2)

        # 🔐 SAFE LOADING — NO torch.load ANYWHERE
        state_dict = load_file("best_hybrid_model.safetensors")
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return tokenizer, model

    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

def transformer_predict(tokenizer, model, text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # REQUIRED FOR HYBRID MODEL
    meta_data = torch.zeros((1, 2))
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            meta_data=meta_data
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # while Stage-2 performs fine-grained verdict classification.
    labels = [
        "Against Accused (Prosecution Wins)",
        "In Favour of Accused",
        "Other / Procedural"
    ]
    return labels[int(probs.argmax())], probs, inputs

def ig_token_attributions(tokenizer, model, text, target_class_id):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    embedding_layer = model.bert.embeddings.word_embeddings
    meta_data = torch.zeros((1, 2))
    def forward(inputs_embeds, attention_mask):
        pooled = model.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).pooler_output
        pooled = model.drop(pooled)
        meta_out = model.meta_relu(model.meta_layer(meta_data))
        combined = torch.cat((pooled, meta_out), dim=1)
        logits = model.out(combined)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, target_class_id]
    lig = LayerIntegratedGradients(forward, embedding_layer)
    baseline_ids = torch.zeros_like(input_ids)
    baseline_emb = embedding_layer(baseline_ids)
    input_emb = embedding_layer(input_ids)
    attributions = lig.attribute(
        input_emb,
        baselines=baseline_emb,
        additional_forward_args=(attention_mask,),
        n_steps=24
    )
    token_attr = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
    token_attr = abs(token_attr) / (abs(token_attr).sum() + 1e-9)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    return tokens, token_attr

def categorize_bias_word(word: str):
    w = word.lower()
    for cat, kws in BIAS_KEYWORDS.items():
        if w in kws:
            return cat
    return None

def compute_bai(tokens, norm_attr):
    total = norm_attr.sum()
    if total == 0:
        return 0.0, {}
    bias_sum = 0
    per_cat = {}
    for tok, a in zip(tokens, norm_attr):
        word = tok.lstrip("#")
        cat = categorize_bias_word(word)
        if cat:
            bias_sum += a
            per_cat[cat] = per_cat.get(cat, 0) + a
    bai = float(bias_sum)
    for k in per_cat:
        per_cat[k] = round(per_cat[k], 4)
    return round(bai, 4), per_cat

# Fix for CSV field limit on Windows (OverflowError protection)
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # Set to a standard large integer for Windows 64-bit
    csv.field_size_limit(2147483647)

# Try importing LIME for explainability
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Legal Verdict Predictor - JUDICIA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ MOCK DATA ------------------
MOCK_CSV_DATA = """facts_summary,issues_raised,acts_summary,ips_sections_used,order_given,final_verdict,case_id,case_title
"Defendant was found with stolen jewelry matching the description. CCTV confirms presence.","Theft committed?","Theft in dwelling","380","Convicted.","Guilty","CRM-2024-001","State vs Doe"
"Accused had a verified alibi (flight tickets) for the time of the crime.","Alibi validity","None","None","Acquitted.","Not Guilty","CRM-2024-002","State vs Smith"
"Company failed to deliver 500 units of steel despite advance payment.","Breach of contract","Contract Act","73","Refund ordered.","Liable","CIV-2024-101","BuildCo vs SteelInc"
"Landlord refused to return security deposit citing wear and tear.","Deposit refund","Rent Control","None","Return ordered.","Liable","CIV-2024-102","Tenant vs Owner"
"Police recovered murder weapon from accused's home matching ballistics.","Possession of weapon","Arms Act","25","Conviction upheld.","Guilty","CRM-2024-003","State vs X"
"Accused acted in self-defense when attacked by armed group.","Right to defense","IPC","96, 97","Acquittal.","Not Guilty","CRM-2024-004","State vs Y"
"Driver speeding at 120kmph caused collision injuring pedestrians.","Rash driving","Motor Vehicles","279","Guilty of negligence.","Guilty","CRM-2024-005","State vs Driver"
"Employee leaked trade secrets to competitor violating NDA.","NDA Breach","Contract Act","27","Injunction granted.","Liable","CIV-2024-103","TechCorp vs Emp"
"Husband provided proof of adultery (hotel receipts).","Divorce grounds","Hindu Marriage Act","13","Divorce granted.","Decreed","FAM-2024-201","H vs W"
"Signature on will proven to be forged by expert.","Will validity","Succession Act","None","Will void.","Invalid","CIV-2024-104","Heirs vs Claimant"
"""

def map_verdict_to_macro_label(verdict_raw: str, order_text: str = "", acts_text: str = "") -> str:
    v = f"{verdict_raw} {order_text} {acts_text}".lower()
    # -----------------------------
    # AGAINST ACCUSED (CONVICTION)
    # -----------------------------
    if any(x in v for x in [
        "convicted",
        "guilty",
        "sentence imposed",
        "sentenced to",
        "conviction upheld",
        "appeal dismissed on merits",
        "petition dismissed on merits",
        "ipc 302",
        "ipc 376",
        "ipc 307",
        "ipc 420"
    ]):
        return "Against Accused (Prosecution Wins)"
    # -----------------------------
    # IN FAVOUR OF ACCUSED
    # -----------------------------
    if any(x in v for x in [
        "acquitted",
        "not guilty",
        "charges quashed",
        "fir quashed",
        "proceedings quashed",
        "appeal allowed",
        "relief granted",
        "set aside conviction",
        "benefit of doubt"
    ]):
        return "In Favour of Accused"
    # -----------------------------
    # PROCEDURAL / NEUTRAL
    # -----------------------------
    return "Other / Procedural"

def load_and_clean_data(reader_iterable):
    """Reads raw rows, cleans text, and maps labels."""
    data_rows = []
    
    for row in reader_iterable:
        facts = safe_strip(row.get("facts_summary"))
        issues = safe_strip(row.get("issues_raised"))
        acts = safe_strip(row.get("acts_summary"))
        ips = safe_strip(row.get("ips_sections_used"))
        order = safe_strip(row.get("order_given"))
        verdict_raw = safe_strip(row.get("final_verdict", ""))
        verdict_norm = normalize_verdict(verdict_raw)

        if verdict_raw == "":
            continue   # drop unlabeled cases
        input_text = " ".join([facts, issues, acts, ips, order]).strip()
        
        # Validation: Ensure text has enough content
        if len(re.findall(r"[a-zA-Z]{3,}", input_text)) < 5:
            continue
        
        combined_acts = f"{acts} {ips}".strip()
        macro_label = map_verdict_to_macro_label(verdict_raw, order, combined_acts)
        
        data_rows.append({
            "text": input_text,
            "label": macro_label,
            "case_id": row.get("case_id", "Unknown"),
            "case_title": row.get("case_title", "Untitled"),
            "original_verdict": verdict_raw,
            "raw_facts": facts,
            "raw_issues": issues,
            "raw_acts": combined_acts,
            "raw_precedents": order,
            "original_verdict": verdict_norm
        })
    return data_rows

# ================== ORIGINAL FUNCTION KEPT ==================
def upsample_training_data(rows):
    """
    Upsamples minority classes to match the MAJORITY size.
    """
    proc = [r for r in rows if r["label"] == "Other / Procedural"]
    fav = [r for r in rows if r["label"] == "In Favour of Accused"]
    ag  = [r for r in rows if r["label"] == "Against Accused (Prosecution Wins)"]
    if not proc: return rows 
    # Target: Match majority class size
    target_size = len(proc)
    # Upsample minority classes
    if fav:
        fav = resample(fav, replace=True, n_samples=target_size, random_state=42)
    if ag:
        ag = resample(ag, replace=True, n_samples=target_size, random_state=42)
    combined = proc + fav + ag
    random.shuffle(combined)
    return combined

# ================== ORIGINAL FUNCTION KEPT ==================
def perform_honest_cv(train_rows_raw, n_splits=5):
    """
    Runs 5-Fold CV with Trigrams + C=0.01
    """
    X_raw = np.array([r["text"] for r in train_rows_raw])
    y_raw = np.array([r["label"] for r in train_rows_raw])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold_train_idx, fold_val_idx in skf.split(X_raw, y_raw):
        # 1. Split Raw Data
        X_fold_tr_raw = X_raw[fold_train_idx]
        y_fold_tr_raw = y_raw[fold_train_idx]
        X_fold_val = X_raw[fold_val_idx]
        y_fold_val = y_raw[fold_val_idx]
        
        if len(set(y_fold_tr_raw)) < 2: continue
        # 2. Upsample ONLY the Training portion
        fold_rows_reconstructed = [{"text": t, "label": l} for t, l in zip(X_fold_tr_raw, y_fold_tr_raw)]
        fold_rows_bal = upsample_training_data(fold_rows_reconstructed)
        
        X_fold_tr_bal = [r["text"] for r in fold_rows_bal]
        y_fold_tr_bal = [r["label"] for r in fold_rows_bal]
        
        # 3. Train (Trigrams + C=0.01)
        # We increase n-grams to (1, 3) to capture "dismissed as withdrawn"
        vec_fold = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
        X_tr_vec = vec_fold.fit_transform(X_fold_tr_bal)
        clf_fold = LogisticRegression(max_iter=1000, C=0.01, solver="liblinear", class_weight=None)
        clf_fold.fit(X_tr_vec, y_fold_tr_bal)
        
        # 4. Validate
        X_val_vec = vec_fold.transform(X_fold_val)
        pred_val = clf_fold.predict(X_val_vec)
        
        cv_scores.append({
            "accuracy": accuracy_score(y_fold_val, pred_val),
            "macro_f1": f1_score(y_fold_val, pred_val, average="macro", zero_division=0)
        })
        
    return cv_scores

# =====================================================
# 3. MAIN ORCHESTRATOR - UPDATED WITH MACRO-F1 OPTIMIZATION
# =====================================================
def _process_rows_and_train(reader_iterable):
    """
    Main Orchestrator: Trigrams (1-3) + C=0.01
    NOW WITH MACRO-F1 OPTIMIZATION
    """
    print("\n--- RUNNING ENHANCED PIPELINE WITH MACRO-F1 OPTIMIZATION ---\n")
    # 1. Load Data
    data_rows = load_and_clean_data(reader_iterable)
    if len(data_rows) < 10: return None 
    original_counts = Counter([r["label"] for r in data_rows])
    # 2. Split 80/20
    all_labels = [r["label"] for r in data_rows]
    train_rows_raw, test_rows_raw = train_test_split(
        data_rows,
        test_size=0.20,
        stratify=all_labels,
        random_state=42
    )
    # ================== NEW: MACRO-F1 CV ==================
    with st.spinner("Running 5-fold cross-validation with macro-F1 optimization..."):
        cv_scores_macro, cv_details = perform_macro_f1_cv(train_rows_raw)
        cv_macro_f1_mean = float(np.mean(cv_scores_macro)) if len(cv_scores_macro) > 0 else 0.0
    # ================== NEW: MACRO-F1 OPTIMIZED FINAL TRAINING ==================
    with st.spinner("Training final model with macro-F1 optimization..."):
        # Use advanced resampling for final training
        train_rows_bal = advanced_resampling_strategy(train_rows_raw)
        X_train_bal = [r["text"] for r in train_rows_bal]
        y_train_bal = [r["label"] for r in train_rows_bal]
        
        # Train with macro-F1 optimization
        macro_f1_model, best_params, best_score = train_macro_f1_optimized_model(X_train_bal, y_train_bal)
        
        # Train ORIGINAL model for comparison
        pipeline_stage2_original = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=8000, sublinear_tf=True),
            LogisticRegression(max_iter=1000, C=0.01, solver="liblinear", class_weight=None)
        )
        pipeline_stage2_original.fit(X_train_bal, y_train_bal)
    # 5. Final Evaluation on UNTOUCHED test set
    X_test = [r["text"] for r in test_rows_raw]
    y_test = [r["label"] for r in test_rows_raw]
    # Predict with both models for comparison
    y_pred_macro = macro_f1_model.predict(X_test)
    y_pred_original = pipeline_stage2_original.predict(X_test)
    # Calculate metrics for both models
    def calculate_comprehensive_metrics(y_true, y_pred, model_name):
        return {
            "model": model_name,
            "test_accuracy": accuracy_score(y_true, y_pred),
            "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "test_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "test_macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "per_class_metrics": {}
        }
    eval_results_macro = calculate_comprehensive_metrics(y_test, y_pred_macro, "Macro-F1 Optimized")
    eval_results_original = calculate_comprehensive_metrics(y_test, y_pred_original, "Original")
    # Calculate per-class metrics for both models
    classes = macro_f1_model.named_steps['clf'].classes_
    
    y_test_np = np.array(y_test)

    for cls in classes:
        cls_mask = (y_test_np == cls)

        if cls_mask.any():
            eval_results_macro["per_class_metrics"][cls] = {
                "recall": recall_score(
                    y_test_np, y_pred_macro, labels=[cls],
                    average=None, zero_division=0
                )[0],
                "f1": f1_score(
                    y_test_np, y_pred_macro, labels=[cls],
                    average=None, zero_division=0
                )[0],
            }

            eval_results_original["per_class_metrics"][cls] = {
                "recall": recall_score(
                    y_test_np, y_pred_original, labels=[cls],
                    average=None, zero_division=0
                )[0],
                "f1": f1_score(
                    y_test_np, y_pred_original, labels=[cls],
                    average=None, zero_division=0
                )[0],
            }

    # 6. Stage-1 Filter (keep original)
    X_full = [r["text"] for r in data_rows]

    # Stage-1 performs binary procedural vs substantive filtering,
    y_full = ["Procedural" if r["label"] == "Other / Procedural" else "Substantive" for r in data_rows]
    pipeline_stage1 = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3,max_features=8000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    pipeline_stage1.fit(X_full, y_full)
    # 7. LIME
    explainer = None
    if LimeTextExplainer is not None:
        try:
            explainer = LimeTextExplainer(
                class_names=list(macro_f1_model.named_steps["clf"].classes_)
            )
        except Exception:
            explainer = None
    # 8. Return Artifacts
    tfidf_macro = get_tfidf_from_pipeline(macro_f1_model)
    tfidf_original = get_tfidf_from_pipeline(pipeline_stage2_original)
    tfidf_stage1 = get_tfidf_from_pipeline(pipeline_stage1)

    return {
        # Models
        "pipeline_stage1": pipeline_stage1,
        "pipeline_stage2_original": pipeline_stage2_original,
        "pipeline_stage2_macro": macro_f1_model,
        
        # Data
        "rows": train_rows_bal,
        "total_samples": len(data_rows),
        "class_counts": dict(original_counts),
        
        # Vectorizers and matrices
        "tfidf_macro": tfidf_macro,
        "X_macro_matrix": tfidf_macro.transform(X_train_bal),
        "tfidf_original": tfidf_original,
        "X_original_matrix": tfidf_original.transform(X_train_bal),
        "tfidf_stage1": tfidf_stage1,
        "X_stage1_matrix": tfidf_stage1.transform(X_full),
        
        # Labels
        "stage1_labels": y_full,
        "stage2_labels": y_train_bal,
        "classes": classes,
        
        # Explainability
        "explainer": explainer,
        
        # Evaluation results
        "evaluation_macro": eval_results_macro,
        "evaluation_original": eval_results_original,
        "cv_macro_f1_mean": cv_macro_f1_mean,
        "cv_macro_f1_scores": cv_scores_macro,
        "cv_details": cv_details,
        "best_params": best_params,
        "best_cv_score": best_score,
        "test_distribution": dict(Counter(y_test))
    }

@st.cache_resource
def train_model_from_csv(csv_file, file_id):
    """
    Chunk-safe CSV trainer.
    Does NOT load full file into memory.
    """
    required_cols = [
        "facts_summary",
        "issues_raised",
        "acts_summary",
        "ips_sections_used",
        "order_given",
        "final_verdict",
    ]
    data_rows = []
    CHUNK_SIZE = 50_000
    for chunk in pd.read_csv(
        csv_file,
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
        on_bad_lines="skip"
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        missing = [c for c in required_cols if c not in chunk.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        for _, row in chunk.iterrows():
            facts = str(row.get("facts_summary", "")).strip()
            issues = str(row.get("issues_raised", "")).strip()
            acts = str(row.get("acts_summary", "")).strip()
            ips = str(row.get("ips_sections_used", "")).strip()
            order = str(row.get("order_given", "")).strip()
            verdict_raw = safe_strip(row.get("final_verdict", ""))
            verdict_norm = normalize_verdict(verdict_raw)

            if verdict_raw == "" or verdict_raw.lower() in ["nan", "null"]:
                continue
            input_text = " ".join([facts, issues, acts, ips, order]).strip()
            if len(re.findall(r"[a-zA-Z0-9]{2,}", input_text)) < 3:
                continue
            data_rows.append({
                "text": input_text,
                "label": map_verdict_to_macro_label(verdict_raw, order, f"{acts} {ips}"),
                "case_id": row.get("case_id", "Unknown ID"),
                "case_title": row.get("case_title", "Untitled Case"),
                "original_verdict": verdict_raw,
                "raw_facts": facts,
                "raw_issues": issues,
                "raw_acts": f"{acts} {ips}".strip(),
                "raw_precedents": order,
            })
    if not data_rows:
        raise ValueError("No valid data found in CSV.")
    return _process_rows_and_train(data_rows)

@st.cache_resource
def train_model_from_mock():
    """Trains the model from the built-in mock string."""
    f = io.StringIO(MOCK_CSV_DATA)
    reader = csv.DictReader(f)
    return _process_rows_and_train(reader)

def find_similar_cases(input_text, train_data, top_k=3):
    """Simple A-RAG style retrieval using cosine similarity over TF-IDF."""
    # Use the macro-F1 model's vectorizer for consistency
    tfidf = train_data["tfidf_macro"]
    X_matrix = train_data["X_macro_matrix"]
    rows = train_data["rows"]
    # Transform the current user input
    try:
        input_vec = tfidf.transform([input_text])
    except Exception as e:
        # If the vectorizer itself fails (rare), return empty
        return []
    # --- CRITICAL FIX: CHECK DIMENSIONS ---
    # The number of features (columns) in the input must match the training matrix
    if input_vec.shape[1] != X_matrix.shape[1]:
        # This silently handles the crash by returning no results
        # prompting the user to reload rather than showing a code error
        st.toast("⚠️ Cache mismatch detected. Please clear cache (Press 'C') to fix similarity search.", icon="⚠️")
        return []
    # --------------------------------------
    if input_vec.nnz == 0:
        return []
    
    # Proceed with cosine similarity if dimensions match
    # Calculate Similarity
    cosine_sims = linear_kernel(input_vec, X_matrix).flatten()
    top_idx = np.argpartition(cosine_sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(cosine_sims[top_idx])[::-1]]

    # Sort by highest score
    related_docs_indices = top_idx
    results = []
    for idx in related_docs_indices:
        score = cosine_sims[idx]
        if score > 0.01:
            row = rows[idx]
            results.append(row | {"score": score})  # Merge score into row dict
    return results

# ================== NEW: COEFFICIENT EXPLANATION FUNCTION ==================
def explain_with_coefficients(model, input_text):
    """Extract feature coefficients for explanation"""
    try:
        # Get the vectorizer and classifier
        vectorizer = get_tfidf_from_pipeline(model)
        classifier = model.named_steps['clf']
        
        # Transform the input
        input_vec = vectorizer.transform([input_text])
        
        # Get feature names and coefficients
        feature_names = vectorizer.get_feature_names_out()
        
        # Get non-zero features
        nonzero_indices = input_vec.nonzero()[1]
        
        impacts = []
        for idx in nonzero_indices:
            feature = feature_names[idx]
            # Sum coefficients across all classes (absolute value for importance)
            importance = np.abs(classifier.coef_[:, idx]).sum()
            if importance > 0.01:  # Threshold
                impacts.append((feature, importance))
        
        # Sort by importance
        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts[:10]  # Top 10 features
        
    except Exception as e:
        print(f"Explanation error: {e}")
        return []

# ------------------ STYLES ------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'Source Serif Pro', serif;
        color: #1e3a8a;
    }
    .stCard {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    /* Probability Bars */
    .prob-container {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .prob-label {
        width: 180px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #334155;
    }
    .prob-bar-bg {
        flex-grow: 1;
        background-color: #f1f5f9;
        height: 12px;
        border-radius: 6px;
        margin-right: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background-color: #3b82f6;
        border-radius: 6px;
    }
    .prob-val {
        width: 50px;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    /* RAG Cards */
    .rag-card {
        background-color: #f8fafc;
        border-left: 4px solid #0ea5e9;
        padding: 12px;
        margin-top: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .rag-title {
        font-weight: 700;
        color: #0f172a;
        display: flex;
        justify-content: space-between;
    }
    .rag-verdict {
        color: #d97706;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .rag-score {
        background: #e0f2fe;
        color: #0284c7;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    /* Impact Words */
    .word-impact {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .impact-pos {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .impact-neg {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    /* Footer Warning */
    .disclaimer-box {
        margin-top: 40px;
        padding: 15px;
        background: #fff1f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        color: #991b1b;
        font-size: 0.85rem;
        text-align: center;
        font-weight: 500;
    }
    
    /* New styles for macro-F1 comparison */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin-bottom: 10px;
    }
    .comparison-box {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .phase-box {
        background-color: #e7f3ff;
        border: 2px solid #007bff;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ SIDEBAR ------------------
st.sidebar.header("📂 Data & Model")

upload_mode = st.sidebar.radio(
    "Select CSV upload mode",
    options=["Single CSV file", "Two CSV files (merge)"],
    index=0
)

uploaded_csv_1 = None
uploaded_csv_2 = None

if upload_mode == "Single CSV file":
    uploaded_csv_1 = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        key="single_csv"
    )

elif upload_mode == "Two CSV files (merge)":
    uploaded_csv_1 = st.sidebar.file_uploader(
        "Upload CSV File – Part 1",
        type=["csv"],
        key="csv_part_1"
    )
    uploaded_csv_2 = st.sidebar.file_uploader(
        "Upload CSV File – Part 2",
        type=["csv"],
        key="csv_part_2"
    )

st.sidebar.info("🔁 Re-training model after CSV merge. Similarity scores may be conservative initially.")
def load_multiple_csvs(csv_files, chunk_size=50_000):
    """
    Reads one or more CSV files safely using chunking
    and returns a list of row dictionaries.
    """
    all_rows = []

    for csv_file in csv_files:
        for chunk in pd.read_csv(
            csv_file,
            chunksize=chunk_size,
            encoding="utf-8-sig",
            on_bad_lines="skip"
        ):
            chunk.columns = [c.strip() for c in chunk.columns]
            all_rows.extend(chunk.to_dict("records"))

    return all_rows

# ================== NEW: MODEL COMPARISON DISPLAY ==================
def display_model_comparison(train_data):
    """Display comparison between original and macro-F1 optimized models"""
    if not train_data:
        return
    
    st.markdown("### 📊 Model Comparison: Original vs Macro-F1 Optimized")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Model")
        eval_orig = train_data.get("evaluation_original", {})
        if eval_orig:
            st.metric("Accuracy", f"{eval_orig.get('test_accuracy', 0)*100:.1f}%")
            st.metric("Macro-F1", f"{eval_orig.get('test_macro_f1', 0)*100:.1f}%")
            st.metric("Macro-Recall", f"{eval_orig.get('test_macro_recall', 0)*100:.1f}%")
    
    with col2:
        st.markdown("#### Macro-F1 Optimized Model")
        eval_macro = train_data.get("evaluation_macro", {})
        if eval_macro:
            st.metric("Accuracy", f"{eval_macro.get('test_accuracy', 0)*100:.1f}%")
            st.metric("Macro-F1", f"{eval_macro.get('test_macro_f1', 0)*100:.1f}%")
            st.metric("Macro-Recall", f"{eval_macro.get('test_macro_recall', 0)*100:.1f}%")
            st.metric("CV Macro-F1", f"{train_data.get('cv_macro_f1_mean', 0)*100:.1f}%")
    
    # Improvement analysis
    if eval_orig and eval_macro:
        f1_improvement = eval_macro.get('test_macro_f1', 0) - eval_orig.get('test_macro_f1', 0)
        recall_improvement = eval_macro.get('test_macro_recall', 0) - eval_orig.get('test_macro_recall', 0)
        
        st.markdown(f"#### 🚀 Improvement Analysis")
        st.markdown(f"- **Macro-F1 Improvement**: `+{f1_improvement*100:.1f}%`")
        st.markdown(f"- **Macro-Recall Improvement**: `+{recall_improvement*100:.1f}%`")
        
        # Show per-class improvements
        st.markdown("#### Per-Class Recall Improvement")
        per_class_orig = eval_orig.get('per_class_metrics', {})
        per_class_macro = eval_macro.get('per_class_metrics', {})
        
        for cls in per_class_macro.keys():
            if cls in per_class_orig:
                orig_recall = per_class_orig[cls].get('recall', 0)
                macro_recall = per_class_macro[cls].get('recall', 0)
                improvement = macro_recall - orig_recall
                
                col_a, col_b, col_c = st.columns(3)
                col_a.write(f"**{cls}**")
                col_b.metric("Original Recall", f"{orig_recall*100:.1f}%")
                col_c.metric("Macro-F1 Recall", f"{macro_recall*100:.1f}%", 
                           delta=f"{improvement*100:+.1f}%")
    
    # Best parameters
    st.markdown("#### ⚙️ Best Parameters (Macro-F1 Model)")
    st.json(train_data.get('best_params', {}))
    
    # CV Details
    with st.expander("📈 Cross-Validation Details"):
        cv_details = train_data.get('cv_details', [])
        if cv_details:
            for fold in cv_details:
                st.write(f"**Fold {fold['fold']}**: CV Macro-F1 = {fold['cv_macro_f1']:.3f}, "
                        f"Val Macro-F1 = {fold['val_macro_f1']:.3f}")

train_data = None
source_label = ""

try:
    # -------- CASE 1: SINGLE CSV --------
    if upload_mode == "Single CSV file" and uploaded_csv_1 is not None:
        with st.spinner("Loading CSV and training model..."):
            all_rows = load_multiple_csvs([uploaded_csv_1])

        train_data = _process_rows_and_train(all_rows)
        source_label = "User CSV (Single File)"

        st.sidebar.success(f"Model trained on {len(all_rows)} rows")

    # -------- CASE 2: TWO CSVs --------
    elif (
        upload_mode == "Two CSV files (merge)"
        and uploaded_csv_1 is not None
        and uploaded_csv_2 is not None
    ):
        with st.spinner("Loading and merging CSV files..."):
            all_rows = load_multiple_csvs([uploaded_csv_1, uploaded_csv_2])

        train_data = _process_rows_and_train(all_rows)
        source_label = "User CSV (Merged Files)"

        st.sidebar.success(
        f"Dataset size: {len(all_rows)} rows\n"
        f"Training: {int(len(all_rows)*0.8)} (80%) | "
        f"Testing: {int(len(all_rows)*0.2)} (20%)"
    )

    # -------- FALLBACK: MOCK DATA --------
    else:
        train_data = train_model_from_mock()
        source_label = "Built-in Mock Data"

        st.sidebar.info("Using built-in mock data for demonstration.")

        st.sidebar.success(
            f"Model trained using mock dataset ({train_data['total_samples']} rows)"
            )

        # Display model comparison
        display_model_comparison(train_data)
        
        # ---- Model evaluation display ----
        eval_data_macro = train_data.get("evaluation_macro", None)
        eval_data_original = train_data.get("evaluation_original", None)

        if eval_data_macro:
            st.sidebar.markdown("### 📈 Model Evaluation (Macro-F1)")
            st.sidebar.write(f"Test Macro-F1: **{eval_data_macro.get('test_macro_f1', 0)*100:.2f}%**")
            st.sidebar.write(f"Test Accuracy: **{eval_data_macro.get('test_accuracy', 0)*100:.2f}%**")
            
            cv_mean = train_data.get("cv_macro_f1_mean")
            if cv_mean is not None:
                st.sidebar.write(f"5-Fold CV Macro-F1: **{cv_mean*100:.2f}%**")
            
            if "classification_report" in eval_data_macro:
                st.sidebar.markdown("#### 📄 Classification Report")
                st.sidebar.text(eval_data_macro["classification_report"])

            if "confusion_matrix" in eval_data_macro:
                st.sidebar.markdown("#### 🔢 Confusion Matrix")
                st.sidebar.text(eval_data_macro["confusion_matrix"])
        
        # Display model comparison for mock data too
        display_model_comparison(train_data)

    # Data Bias Warning in Sidebar
    # Data Bias Warning in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Distribution")
    
    # This now pulls the REAL (original) counts, not the synthetic training counts
    counts = train_data.get("class_counts", {})
    total = train_data.get("total_samples", 0)
    
    if total > 0:
        for cls, cnt in counts.items():
            pct = (cnt / total) * 100
            st.sidebar.text(f"{cls}: {pct:.1f}%")
    else:
        st.sidebar.text("Distribution unavailable")

    if train_data.get("explainer") is None:
        st.sidebar.warning(
            "Note: 'lime' library not found. Using coefficient-based explainability instead."
        )
    
    # ================== SIDEBAR: CLASSIFICATION REPORT ==================
    if train_data and "evaluation_macro" in train_data:
        eval_macro = train_data["evaluation_macro"]

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📄 Classification Report")

        report = eval_macro.get("classification_report", None)
        if report:
            st.sidebar.text(report)

        st.sidebar.markdown("### 🔢 Confusion Matrix")
        cm = eval_macro.get("confusion_matrix", None)
        if cm is not None:
            st.sidebar.text(cm)

except Exception as e:
    st.sidebar.error(f"Error: {e}")

# ------------------ MAIN UI ------------------
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 2rem;'>⚖️ JUDICIA: AI Legal Assistant</h1>",
    unsafe_allow_html=True,
)

col_input, col_output = st.columns([1, 1], gap="large")

# ================== MAIN GRID ==================
row1_left, row1_right = st.columns(2, gap="large")
row2_left, row2_right = st.columns(2, gap="large")
row3 = st.container()

with row1_left:
    st.markdown("### 📝 Case Details")

    # Initialize session state for inputs if not present
    if "input_facts" not in st.session_state:
        st.session_state.input_facts = ""
    if "input_issues" not in st.session_state:
        st.session_state.input_issues = ""
    if "input_acts" not in st.session_state:
        st.session_state.input_acts = ""
    if "input_precedents" not in st.session_state:
        st.session_state.input_precedents = ""

    # Case metadata inputs
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        case_id = st.text_input("Case ID", "CASE-2025-001")
        case_date = st.text_input("Date of Filing / Judgment", "2025-03-24")
    with meta_col2:
        court_name = st.text_input("Court Name", "Supreme Court of India")
        judge_name = st.text_input("Presiding Judge(s)", "")

    case_title = st.text_input("Case Title", "State vs Anderson")
    case_type = st.selectbox(
        "Type of Case",
        [
            "Criminal",
            "Civil",
            "Writ Petition",
            "Constitutional",
            "Family / Matrimonial",
            "Property / Land Dispute",
            "Labour & Employment",
            "Service Law",
            "Taxation",
            "Consumer Dispute",
            "Corporate / Commercial",
            "Environmental",
            "Intellectual Property",
            "Other",
        ],
        index=0,
    )

    st.info(
        "Enter case facts below. The AI will validate if the input makes sense before processing."
    )

    # Inputs with keys for programmatic updates
    facts = st.text_area(
        "Facts of the Case",
        height=150,
        placeholder="Enter the summary of facts...",
        key="input_facts",
    )
    issues = st.text_area(
        "Issues Raised",
        height=80,
        placeholder="Legal issues involved...",
        key="input_issues",
    )
    acts = st.text_input(
        "Acts/Sections Involved",
        placeholder="e.g. IPC 302, CrPC 482",
        key="input_acts",
    )
    precedents = st.text_area(
        "Arguments / Precedents Cited", height=80, key="input_precedents"
    )

    predict_btn = st.button(
        "Analyze Verdict & Find Precedents", type="primary", use_container_width=True
    )

with row1_right:
    # ----------------------------------------------------
    # NEW: DYNAMIC KPI CARDS DISPLAY (ALWAYS SHOW IF TRAINED)
    # ----------------------------------------------------
    if train_data:
        # Extract metrics from the trained model data
        eval_macro = train_data.get("evaluation_macro", {})
        
        # 1. Test Accuracy
        real_test_acc = eval_macro.get("test_accuracy", 0) * 100
        
        # 2. Test Macro F1
        real_test_f1 = eval_macro.get("test_macro_f1", 0) * 100
        
        # 3. CV Macro F1
        real_cv_f1 = train_data.get("cv_macro_f1_mean", 0) * 100
        
        # 4. Substantive Pct
        counts = train_data.get("class_counts", {})
        total_samples = train_data.get("total_samples", 0)
        proc_count = counts.get("Other / Procedural", 0)
        
        if total_samples > 0:
            real_sub_pct = ((total_samples - proc_count) / total_samples) * 100
        else:
            real_sub_pct = 0.0
        
        # Display the cards using the real data
        st.markdown("## 📊 Phase 1 Results: Actual Model Performance")
        phase1_kpi_cards(
            test_acc=real_test_acc,
            test_macro_f1=real_test_f1,
            cv_macro_f1=real_cv_f1,
            substantive_pct=real_sub_pct
        )

        # ---------- NEW: PER-CLASS PERFORMANCE BELOW KPI CARDS ----------
        eval_macro = train_data.get("evaluation_macro", {})
        test_dist = train_data.get("test_distribution", {})

        display_per_class_performance(eval_macro, test_dist)

    if predict_btn or (facts and len(facts) > 10):  # Trigger if data exists
        input_text = " ".join([facts, issues, acts, precedents]).strip()

        # 1. Sensitivity & validity check
        valid_words = re.findall(r"[a-zA-Z0-9]{2,}", input_text)

        if len(valid_words) < 3:
            st.warning("Waiting for sufficient data to analyze...")

        elif train_data is None:
            st.error("⚠️ Model failed to load.")

        else:
            # Use macro-F1 model's vectorizer for consistency
            tfidf = train_data["tfidf_macro"]
            input_vec = tfidf.transform([input_text])

            if input_vec.nnz == 0:
                st.error(
                    "Nonsensical / Unrecognized Input:** The text provided does not"
                    "contain legal terminology present in the training database."
                    "Please check for typos or use standard legal phrasing."
                )
            else:
                # ================== NEW: USE MACRO-F1 MODEL ==================
                model = train_data["pipeline_stage2_macro"]
                
                # Get predictions from both models for comparison
                probas_macro = model.predict_proba([input_text])[0]
                probas_original = train_data["pipeline_stage2_original"].predict_proba([input_text])[0]
                    
                classes = model.named_steps['clf'].classes_
                sorted_indices = np.argsort(probas_macro)[::-1]
                top_class = classes[sorted_indices[0]]
                top_conf = probas_macro[sorted_indices[0]]
                    
                # Get prediction from original model too
                pred_original = train_data["pipeline_stage2_original"].predict([input_text])[0]
                conf_original = max(probas_original)

                # Bias signal
                counts = train_data.get("class_counts", {})
                total = sum(counts.values())
                class_prevalence = counts.get(top_class, 0) / total if total > 0 else 0

                if class_prevalence > 0.65:
                    bias_line = (
                        "Bias status: this prediction is **potentially biased** "
                        "because this verdict type dominates the training data."
                    )
                else:
                    bias_line = (
                        "Bias status: this prediction appears **relatively unbiased** "
                        "with respect to class distribution in the training data."
                    )

with row2_left:
                    st.markdown("### 🔍 Analysis Results")
                    st.markdown(
                        f"""
                        <div style="padding: 15px; background: #eff6ff; border-radius: 8px;
                                    border: 1px solid #bfdbfe; margin-bottom: 20px;">
                            <h3 style="margin:0; color: #1e3a8a;">Predicted Verdict: {top_class}</h3>
                            <p style="margin:5px 0 0 0; color: #64748b;">
                                Confidence: <b>{top_conf * 100:.1f}%</b> (Macro-F1 Optimized Model)
                            </p>
                            <p style="margin:5px 0 0 0; color: #94a3b8;">
                                Original Model: {pred_original} ({conf_original*100:.1f}% confidence)
                            </p>
                            <p style="margin:8px 0 0 0; font-size:0.85rem; color:#b45309;">
                                {bias_line}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Case Snapshot
                    st.markdown("#### 🗂 Case Snapshot")
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc; border-radius:8px;
                                    border:1px solid #e2e8f0; padding:10px 12px;
                                    font-size:0.9rem; margin-bottom:1rem;">
                            <div style="display:flex; justify-content:space-between;
                                        border-bottom:1px dashed #e2e8f0;
                                        padding-bottom:4px; margin-bottom:4px;">
                                <b>Case ID</b><span>{case_id}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;
                                        border-bottom:1px dashed #e2e8f0;
                                        padding-bottom:4px; margin-bottom:4px;">
                                <b>Title</b><span>{case_title}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;
                                        border-bottom:1px dashed #e2e8f0;
                                        padding-bottom:4px; margin-bottom:4px;">
                                <b>Court</b><span>{court_name}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;
                                        border-bottom:1px dashed #e2e8f0;
                                        padding-bottom:4px; margin-bottom:4px;">
                                <b>Date</b><span>{case_date}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;
                                        border-bottom:1px dashed #e2e8f0;
                                        padding-bottom:4px; margin-bottom:4px;">
                                <b>Case Type</b><span>{case_type}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <b>Judge(s)</b><span>{judge_name or "—"}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Additional explicit bias warning text
                    if class_prevalence > 0.65:
                        st.warning(
                            f"⚠️ **Potential Bias Warning:** The verdict '{top_class}' appears in "
                            f"{class_prevalence * 100:.1f}% of the historical data. "
                            "The model might be favoring this outcome due to class imbalance."
                        )

                    # Confidence breakdown for both models
                    st.markdown("#### 📊 Confidence Breakdown (Macro-F1 Model)")
                    for i in sorted_indices:
                        cls = classes[i]
                        pct = probas_macro[i] * 100
                        st.markdown(
                            f"""
                            <div class="prob-container">
                                <div class="prob-label">{cls}</div>
                                <div class="prob-bar-bg">
                                    <div class="prob-bar-fill" style="width: {pct}%;"></div>
                                </div>
                                <div class="prob-val">{pct:.1f}%</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    
                    # ================== NEW: MODEL PERFORMANCE CONTEXT ==================
                    with st.expander("📈 Model Performance Context"):
                        eval_macro = train_data.get("evaluation_macro", {})
                        if eval_macro:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Model Macro-F1", f"{eval_macro.get('test_macro_f1', 0)*100:.1f}%")
                            with col_b:
                                st.metric("Model Macro-Recall", f"{eval_macro.get('test_macro_recall', 0)*100:.1f}%")
                            
                            # Show per-class performance
                            st.markdown("**Per-Class Recall:**")
                            per_class = eval_macro.get("per_class_metrics", {})
                            for cls, metrics in per_class.items():
                                st.write(f"- {cls}: {metrics.get('recall', 0)*100:.1f}%")

                    # 4. Explainability (LIME or fallback)
                    st.markdown("#### 💡 Explanation (Key Influencing Terms)")
                    st.caption("Which words in your input influenced this decision the most?")

                    if train_data["explainer"] is not None:
                        with st.spinner("Generating LIME explanation..."):
                            exp = train_data["explainer"].explain_instance(
                                input_text,
                                model.predict_proba,
                                num_features=6,
                            )
                            components.html(exp.as_html(), height=300, scrolling=True)
                    else:
                        st.caption(
                            "Using coefficient analysis (LIME not available). "
                            "Green = Supports verdict, Red = Opposes."
                        )
                        impacts = explain_with_coefficients(model, input_text)
                        if impacts:
                            for word, weight in impacts:
                                style_cls = "impact-pos" if weight > 0 else "impact-neg"
                                arrow = "↑" if weight > 0 else "↓"
                                st.markdown(
                                    f'<span class="word-impact {style_cls}">'
                                    f"{word} ({arrow} {abs(weight):.3f})</span>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No strong features detected.")

with row2_right:
                    # 5. A-RAG (similar precedents)
                    # ------------------ PHASE-2: TRANSFORMER RESEARCH MODEL ------------------
                    st.markdown("### 🧠 Phase-2 Research: Transformer Verdict + Bias Attribution")

                    with st.expander("🧠 Phase-2 Transformer Analysis"):
                        tokenizer_t, model_t = load_transformer_model()

                    if tokenizer_t and model_t:
                        pred_label_t, probs_t, inputs_t = transformer_predict(tokenizer_t, model_t, input_text)
                        conf_t = float(probs_t.max())

                        st.success(f"Transformer Verdict: {pred_label_t} ({conf_t*100:.1f}% confidence)")
                        # IG attribution
                        try:
                            tokens_ig, attribs_ig = ig_token_attributions(tokenizer_t, model_t, input_text, probs_t.argmax())
                            bai, cat_contrib = compute_bai(tokens_ig, attribs_ig)

                            st.markdown("#### Token Influence (Integrated Gradients)")
                            heatmap = []
                            for tok, a in zip(tokens_ig, attribs_ig):
                                if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                                    continue

                                # === CORRECTION MADE HERE: Indentation fixed ===
                                word = tok.lstrip("#")
                                color = f"rgba(59,130,246,{a*0.8+0.1})"
                                border = "2px solid red" if categorize_bias_word(word) else "1px solid #ddd"
                                heatmap.append(
                                    f"<span style='padding:2px 4px; margin:2px; border:{border}; background:{color};'>{word}</span>"
                                )
                                # ================================================

                            st.markdown("<div style='line-height:1.8'>" + " ".join(heatmap) + "</div>", unsafe_allow_html=True)

                            st.markdown("#### Bias Attribution Index (BAI)")
                            st.write(f"**BAI:** {bai}")
                            if cat_contrib:
                                st.write("Breakdown by category:")
                                for k, v in cat_contrib.items():
                                    st.write(f"- {k}: {v}")
                            else:
                                st.write("No sensitive terms influenced the prediction.")
                        except Exception as e:
                            st.warning(f"Transformer explanation error: {e}")
                    else:
                        st.info("Transformer model not loaded (Check 'DAYSH/legal-verdict-transformer' availability or installs).")

                    st.markdown("#### 📚 Similar Case Precedents (A-RAG)")
                    st.caption("Cases with similar fact patterns from the database:")
                    similar_cases = find_similar_cases(input_text, train_data)

                    if similar_cases:
                        for sim in similar_cases:
                            with st.container():
                                st.markdown(
                                    f"""
                                    <div class="rag-card">
                                        <div class="rag-title">
                                            <span>{sim['case_title']} <small>({sim['case_id']})</small></span>
                                            <span class="rag-score">Similarity: {sim['score'] * 100:.0f}%</span>
                                        </div>
                                        <div style="margin-top:5px; font-size:0.9rem;">
                                            Verdict: <span style="color:#d97706; font-weight:bold;">{sim['original_verdict']}</span>.
                                        </div>
                                        <div style="font-style: italic; color: #64748b; font-size: 0.8rem;
                                                    margin-top:5px; border-top:1px dashed #cbd5e1; padding-top:4px;">
                                            "{sim['text'][:200]}..."
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                # Load Button
                                if st.button(
                                    f"📥 Load Data from {sim['case_id']}",
                                    key=f"btn_{sim['case_id']}",
                                ):
                                    st.session_state.input_facts = sim["raw_facts"]
                                    st.session_state.input_issues = sim["raw_issues"]
                                    st.session_state.input_acts = sim["raw_acts"]
                                    st.session_state.input_precedents = sim[
                                        "raw_precedents"
                                    ]
                                    st.rerun()
                    else:
                        st.info("No highly similar historical cases found in the database.")
                    
                    # Show diagnostic insights
                    eval_macro = train_data.get("evaluation_macro", {})
                    if eval_macro and eval_macro.get('test_macro_f1', 0) < 0.6:
                        st.markdown("""
                        <div class="phase-box">
                        <h4>⚠️ Diagnostic Insight from Phase 1</h4>
                        
                        **Problem**: Text-only models achieve **{:.1f}% macro-F1** despite **{:.1f}% accuracy**.
                        
                        **Root Causes**:
                        1. **Text patterns ≠ legal reasoning**: Model exploits spurious correlations
                        2. **No element-based analysis**: Cannot track prima facie requirements
                        3. **Bias amplification**: Historical imbalances are perpetuated
                        4. **Black-box decisions**: No attribution-aware validation
                        
                        **Solution (Phase 2)**: Structured legal elements + attribution-aware validation
                        </div>
                        """.format(
                            eval_macro.get('test_macro_f1', 0)*100,
                            eval_macro.get('test_accuracy', 0)*100
                        ), unsafe_allow_html=True)

with row3:
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚖️ <b>DISCLAIMER:</b> This verdict is predicted by AI based on historical patterns.
            <b>All final judgments are made by the Court.</b><br>
            AI can make mistakes. Please consult a qualified lawyer or judge for professional legal advice.
        </div>
        """,
        unsafe_allow_html=True,
        )

    if st.session_state.run_analysis and train_data:
        try:
            similar_cases = find_similar_cases(input_text, train_data)

            if similar_cases:
                for sim in similar_cases:
                    st.markdown(f"- **{sim['case_title']}** ({sim['score']*100:.1f}%)")
            else:
                st.info("No highly similar historical cases found.")

        except Exception as e:
            st.error(f"A-RAG Error: {e}")
