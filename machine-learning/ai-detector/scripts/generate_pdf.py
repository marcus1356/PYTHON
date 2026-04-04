"""
Gera o documento técnico do AI Detector em PDF.
Uso: python scripts/generate_pdf.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF, XPos, YPos

PDF_OUT = ROOT / "AI_Detector_Technical_Document.pdf"
METRICS_PATH = ROOT / "app" / "models" / "ml" / "v3_metrics.json"


# ---------------------------------------------------------------------------
# Paleta de cores
# ---------------------------------------------------------------------------
C_DARK   = (15,  23,  42)    # slate-900
C_ACCENT = (99,  102, 241)   # indigo-500
C_GREEN  = (16,  185, 129)   # emerald-500
C_AMBER  = (245, 158,  11)   # amber-500
C_LIGHT  = (241, 245, 249)   # slate-100
C_WHITE  = (255, 255, 255)
C_GRAY   = (100, 116, 139)   # slate-500
C_RED    = (239,  68,  68)   # red-500


class PDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.current_chapter = ""

    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_WHITE)
        self.set_y(3)
        self.cell(0, 6, "AI DETECTOR  |  Technical Documentation", align="C")
        self.set_text_color(*C_GRAY)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, "AI Detector - Confidential Technical Documentation - 2025", align="C")

    def cover_page(self):
        self.add_page()
        # Background
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 297, "F")

        # Accent bar
        self.set_fill_color(*C_ACCENT)
        self.rect(0, 120, 210, 4, "F")

        # Title
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 42)
        self.set_y(60)
        self.cell(0, 20, "AI DETECTOR", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "", 18)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 12, "Technical Documentation", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_y(135)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*C_LIGHT)
        self.cell(0, 8, "Statistical Machine Learning for AI Content Detection", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.cell(0, 8, "Without Depending on LLMs", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Tags
        self.set_y(165)
        tags = ["v3.0", "93.4% Accuracy", "12 Features", "Real-time", "Continuous Learning"]
        total_w = len(tags) * 36 + (len(tags) - 1) * 4
        x_start = (210 - total_w) / 2
        for i, tag in enumerate(tags):
            self.set_fill_color(*C_ACCENT)
            x = x_start + i * 40
            self.set_xy(x, 165)
            self.cell(36, 8, tag, align="C", fill=True)

        # Stats block
        self.set_y(195)
        stats = [
            ("93.4%", "CV Accuracy"),
            ("16,000", "Training Samples"),
            ("12", "Features"),
            ("<50ms", "Latency"),
        ]
        col_w = 210 / len(stats)
        for i, (val, lbl) in enumerate(stats):
            self.set_fill_color(*C_DARK)
            self.set_draw_color(*C_ACCENT)
            self.set_line_width(0.5)
            self.set_xy(i * col_w + 5, 195)
            self.cell(col_w - 10, 20, "", border=1)
            self.set_xy(i * col_w + 5, 198)
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*C_GREEN)
            self.cell(col_w - 10, 8, val, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_xy(i * col_w + 5, 207)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*C_LIGHT)
            self.cell(col_w - 10, 6, lbl, align="C")

        # Date
        self.set_y(250)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*C_GRAY)
        self.cell(0, 8, "April 2025", align="C")

    def section_title(self, number: str, title: str):
        self.ln(6)
        self.set_fill_color(*C_ACCENT)
        self.rect(15, self.get_y(), 3, 10, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*C_DARK)
        self.set_x(22)
        self.cell(0, 10, f"{number}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.3)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(4)

    def subsection(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_DARK)
        self.set_x(15)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.set_x(15)
        self.multi_cell(180, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def bullet(self, items: list[tuple[str, str]]):
        """items = [(bold_label, text), ...]"""
        for label, text in items:
            self.set_x(20)
            self.set_fill_color(*C_ACCENT)
            self.circle(19, self.get_y() + 2.5, 1.2, "F")
            if label:
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*C_DARK)
                self.cell(35, 5.5, label + ":", new_x=XPos.RIGHT, new_y=YPos.TOP)
                self.set_font("Helvetica", "", 10)
                self.set_text_color(50, 50, 50)
                self.multi_cell(145, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                self.set_font("Helvetica", "", 10)
                self.set_text_color(50, 50, 50)
                self.multi_cell(170, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def info_box(self, title: str, text: str, color=None):
        if color is None:
            color = C_LIGHT
        y = self.get_y()
        self.set_fill_color(*color)
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.3)
        self.set_x(15)
        self.rect(15, y, 180, 4, "F")  # header
        self.set_fill_color(248, 250, 252)
        self.rect(15, y + 4, 180, 22, "F")
        self.set_xy(18, y + 0.5)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_WHITE if color == C_ACCENT else C_DARK)
        self.cell(0, 3.5, title)
        self.set_xy(18, y + 6)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(174, 4.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def feature_table(self, features: list[dict]):
        headers = ["#", "Feature", "Sinal de IA", "Importancia", "Descricao"]
        widths  = [8, 42, 28, 22, 80]

        # Header
        self.set_fill_color(*C_DARK)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 8)
        self.set_x(15)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        for i, f in enumerate(features):
            fill = i % 2 == 0
            self.set_fill_color(248, 250, 252) if fill else self.set_fill_color(*C_WHITE)
            self.set_text_color(*C_DARK)
            self.set_font("Helvetica", "", 8)
            self.set_x(15)
            row_y = self.get_y()
            self.cell(widths[0], 6, str(i + 1), border=1, fill=fill, align="C")
            self.set_font("Helvetica", "B", 8)
            self.cell(widths[1], 6, f["name"], border=1, fill=fill)
            self.set_font("Helvetica", "", 8)

            # Signal color
            sig = f["signal"]
            if "Alta" in sig:
                self.set_text_color(*C_RED)
            else:
                self.set_text_color(*C_GREEN)
            self.cell(widths[2], 6, sig, border=1, fill=fill, align="C")
            self.set_text_color(*C_DARK)

            # Importance bar
            imp = f.get("importance", 0)
            self.set_x(15 + sum(widths[:3]))
            self.cell(widths[3], 6, f"{imp:.1%}", border=1, fill=fill, align="C")

            self.cell(widths[4], 6, f["desc"], border=1, fill=fill)
            self.ln()

        self.ln(3)

    def accuracy_bar(self, label: str, value: float, color):
        bar_w = int(value * 130)
        y = self.get_y()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_DARK)
        self.set_x(15)
        self.cell(45, 6, label)
        self.set_fill_color(*color)
        self.rect(60, y + 1, bar_w, 4, "F")
        self.set_fill_color(220, 220, 220)
        self.rect(60 + bar_w, y + 1, 130 - bar_w, 4, "F")
        self.set_xy(195, y)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*color)
        self.cell(0, 6, f"{value:.1%}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def roadmap_item(self, phase: str, title: str, items: list[str], status: str, color):
        y = self.get_y()
        self.set_fill_color(*color)
        self.rect(15, y, 180, 8, "F")
        self.set_xy(18, y + 1.5)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_WHITE)
        self.cell(30, 5, phase)
        self.cell(120, 5, title)
        self.set_font("Helvetica", "B", 8)
        self.cell(0, 5, f"[{status}]", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        for item in items:
            self.set_x(22)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(60, 60, 60)
            self.cell(0, 5, f"  - {item}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


# ---------------------------------------------------------------------------
# Conteúdo
# ---------------------------------------------------------------------------

def build_pdf():
    # Load metrics if available
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    cv_acc = metrics.get("cv_accuracy_mean", 0.934)
    n_samples = metrics.get("n_train_samples", 16000)
    feat_imp = metrics.get("feature_importances", {})

    pdf = PDF()
    pdf.set_margins(15, 18, 15)

    # ---- COVER ----
    pdf.cover_page()

    # ---- PAGE 2: CONTENTS ----
    pdf.add_page()
    pdf.set_y(25)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*C_DARK)
    pdf.cell(0, 10, "Table of Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*C_ACCENT)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    toc = [
        ("1", "Executive Summary", 3),
        ("2", "The Problem: Detecting AI-Generated Content", 3),
        ("3", "Why Not Just Use an LLM?", 4),
        ("4", "Our Approach: Statistical Detection", 4),
        ("5", "Feature Engineering - 12 Discriminators", 5),
        ("6", "Model Architecture: The Cascade", 6),
        ("7", "Performance Metrics", 7),
        ("8", "Continuous Learning System", 8),
        ("9", "Competitive Differentiation", 8),
        ("10", "Technical Stack", 9),
        ("11", "Roadmap", 9),
    ]
    for num, title, page in toc:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*C_DARK)
        pdf.set_x(15)
        pdf.cell(15, 7, num + ".", align="R")
        pdf.cell(150, 7, title)
        pdf.set_text_color(*C_ACCENT)
        pdf.cell(0, 7, str(page), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(220, 220, 220)
        pdf.set_line_width(0.1)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())

    # ---- PAGE 3: EXECUTIVE SUMMARY ----
    pdf.add_page()
    pdf.set_y(25)

    pdf.section_title("1", "Executive Summary")
    pdf.body(
        "AI Detector is a machine learning platform designed to identify AI-generated content "
        "in text and images. Unlike competing solutions that rely exclusively on Large Language "
        "Models (LLMs) for classification, AI Detector uses a cascade architecture where the "
        "statistical ML engine is the primary decision-maker - fast, deterministic, and "
        "privacy-preserving - while Claude serves only as a tiebreaker in uncertain cases."
    )
    pdf.body(
        f"The current model (v3) achieves {cv_acc:.1%} cross-validated accuracy on {n_samples:,} "
        "real-world examples from public research datasets (HuggingFace). It processes text in "
        "under 50ms on commodity hardware with no external API calls required."
    )

    pdf.info_box(
        "Key Metrics",
        f"Cross-validated Accuracy: {cv_acc:.1%}   |   "
        f"Training Samples: {n_samples:,}   |   "
        "Feature Dimensions: 12   |   Inference Latency: <50ms   |   "
        "External API Required: No (optional)",
        C_ACCENT,
    )

    # ---- SECTION 2 ----
    pdf.section_title("2", "The Problem: Detecting AI-Generated Content")
    pdf.body(
        "The proliferation of Large Language Models (GPT-4, Claude, Gemini, Llama) has made "
        "it trivially easy to generate human-like text at scale. This creates serious challenges "
        "across multiple domains:"
    )
    pdf.bullet([
        ("Education", "Students submitting AI-written essays and assignments."),
        ("Journalism", "AI-generated disinformation and synthetic news articles."),
        ("Research", "Fabricated academic abstracts polluting scientific literature."),
        ("Content Platforms", "Spam, fake reviews, and low-quality AI content flooding feeds."),
        ("Hiring", "AI-written cover letters and resumes misrepresenting candidates."),
        ("Legal", "AI-generated contracts and documents passed as human-authored."),
    ])
    pdf.body(
        "The core challenge is that LLMs produce text that is grammatically correct, semantically "
        "coherent, and superficially indistinguishable from human writing. However, at the "
        "statistical level, LLM-generated text exhibits systematic, measurable patterns that "
        "humans do not replicate - and this is the foundation of our detection approach."
    )

    # ---- SECTION 3 ----
    pdf.section_title("3", "Why Not Just Use an LLM?")
    pdf.body(
        "The naive approach to AI detection is to ask another LLM: 'Is this text AI-generated?' "
        "This approach is fundamentally flawed for several reasons:"
    )
    pdf.bullet([
        ("Hallucination", "LLMs produce confident but incorrect classifications. They can 'decide' "
            "a text is human-written based on stylistic preferences, not ground truth."),
        ("Cost", "Every text analysis requires an API call (typically $0.001-0.01 per query). "
            "At scale, this becomes prohibitive."),
        ("Latency", "Network roundtrips add 500ms-3s of latency - unacceptable for real-time use cases."),
        ("Jailbreaking", "LLMs can be prompted to always answer 'human' or manipulated via "
            "adversarial prefixes in the content being analyzed."),
        ("Black Box", "No explanation is provided. You cannot audit why a text was flagged."),
        ("Privacy", "Sending user content to third-party LLM APIs creates data sovereignty issues."),
        ("Drift", "LLM behavior changes with model updates - your classifier may degrade without notice."),
    ])
    pdf.info_box(
        "Our Philosophy",
        "Claude (or any LLM) is used ONLY as a tiebreaker when the statistical model confidence "
        "falls in the uncertain zone [0.35, 0.65]. The ML engine is the core. LLM is the "
        "optional safety net. This keeps costs low, latency fast, and the system auditable.",
        C_ACCENT,
    )

    # ---- SECTION 4 ----
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("4", "Our Approach: Statistical Detection")
    pdf.body(
        "AI-generated text is statistically distinguishable from human writing at multiple "
        "linguistic levels. The key insight is rooted in information theory and psycholinguistics:"
    )
    pdf.bullet([
        ("Burstiness", "Human language is 'bursty' (Goh & Barabasi, 2008) - sentence lengths "
            "vary dramatically. LLMs produce eerily uniform sentence rhythms."),
        ("Zipf's Law", "Human texts follow Zipf's Law with a long tail of rare words. LLMs "
            "converge to a more restricted vocabulary distribution."),
        ("Perplexity", "LLMs generate text that other LLMs find highly probable (low perplexity). "
            "Human writing is more 'surprising' at the n-gram level."),
        ("Hedging", "Humans express uncertainty ('maybe', 'I think', 'seems'). LLMs assert "
            "with uniform confidence - they rarely hedge."),
        ("First Person", "Humans write from lived experience. LLMs adopt an impersonal, "
            "objective register, avoiding 'I', 'my', 'we'."),
        ("Transition Abuse", "LLMs overuse formal transition phrases ('Furthermore', 'Moreover', "
            "'It is worth noting') as structural scaffolding."),
    ])
    pdf.body(
        "These patterns are not individual silver bullets - each feature alone is insufficient. "
        "The power comes from combining all 12 features in a Random Forest ensemble that learns "
        "their joint discriminative boundaries from 16,000 real-world examples."
    )

    # ---- SECTION 5: FEATURES ----
    pdf.section_title("5", "Feature Engineering - 12 Discriminators")
    pdf.body(
        "The feature vector has 12 dimensions, computed purely from the raw text using "
        "regular expressions and statistical functions - no external APIs, no pretrained "
        "embeddings, no neural networks."
    )

    features_data = [
        {
            "name": "avg_sentence_length",
            "signal": "Alta = IA",
            "importance": feat_imp.get("avg_sentence_length", 0.054),
            "desc": "Mean words per sentence. AI: uniform 18-26 words.",
        },
        {
            "name": "vocabulary_richness",
            "signal": "Media = IA",
            "importance": feat_imp.get("vocabulary_richness", 0.093),
            "desc": "Type-Token Ratio. AI: constrained 0.35-0.55.",
        },
        {
            "name": "burstiness",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("burstiness", 0.301),
            "desc": "CV of sentence lengths. Humans vary wildly.",
        },
        {
            "name": "punctuation_density",
            "signal": "Media = IA",
            "importance": feat_imp.get("punctuation_density", 0.086),
            "desc": "Punctuation chars / total chars. AI: 0.03-0.07.",
        },
        {
            "name": "avg_word_length",
            "signal": "Alta = IA",
            "importance": feat_imp.get("avg_word_length", 0.162),
            "desc": "Mean chars per word. AI uses formal vocabulary.",
        },
        {
            "name": "transition_word_density",
            "signal": "Alta = IA",
            "importance": feat_imp.get("transition_word_density", 0.075),
            "desc": "Rate of 'Furthermore', 'Moreover', etc.",
        },
        {
            "name": "first_person_ratio",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("first_person_ratio", 0.057),
            "desc": "I/my/me/we per token. Humans write personally.",
        },
        {
            "name": "hedge_word_ratio",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("hedge_word_ratio", 0.010),
            "desc": "'Maybe', 'perhaps', 'I think'. AI asserts confidently.",
        },
        {
            "name": "question_density",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("question_density", 0.003),
            "desc": "Questions per sentence. Humans question more.",
        },
        {
            "name": "bigram_repetition",
            "signal": "Alta = IA",
            "importance": feat_imp.get("bigram_repetition_score", 0.032),
            "desc": "Repeated 2-word sequences. AI repeats phrases.",
        },
        {
            "name": "lexical_entropy",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("lexical_diversity_entropy", 0.031),
            "desc": "Shannon entropy of word freq. AI is predictable.",
        },
        {
            "name": "hapax_legomena",
            "signal": "Baixa = IA",
            "importance": feat_imp.get("hapax_legomena_ratio", 0.097),
            "desc": "Words appearing exactly once. Zipf tail measure.",
        },
    ]
    pdf.feature_table(features_data)

    # ---- SECTION 6: ARCHITECTURE ----
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("6", "Model Architecture: The Cascade")

    pdf.body(
        "Detection follows a three-layer cascade. Each layer only triggers if the previous "
        "layer lacks sufficient confidence. This minimizes cost and latency while maximizing accuracy."
    )

    # Cascade diagram (text-based)
    cascade_text = (
        "INPUT (Text or Image)\n"
        "       |\n"
        "  [Layer 1] Heuristics - 12 statistical features, rule-based thresholds\n"
        "       | score in [0.40, 0.60]?\n"
        "  [Layer 2] Random Forest - 200 trees, trained on 16,000 real examples\n"
        "       | blended score in [0.35, 0.65]?\n"
        "  [Layer 3] Claude API - Haiku for text, Sonnet Vision for images\n"
        "       |\n"
        "   VERDICT: ai | human | uncertain + probability score + explanation"
    )
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(*C_LIGHT)
    pdf.set_x(15)
    y0 = pdf.get_y()
    pdf.multi_cell(180, 5, cascade_text, fill=True, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.subsection("Layer 1: Statistical Heuristics")
    pdf.body(
        "The 12 features are computed from raw text using pure Python - no network calls, "
        "no model loading, no GPU. This runs in microseconds. The score is the sum of "
        "weighted rule matches. Interpretable output: you know exactly which features "
        "triggered the AI classification."
    )

    pdf.subsection("Layer 2: Random Forest")
    pdf.body(
        "A Random Forest with 200 decision trees learns the joint feature boundaries from "
        "labeled data. It combines the 12 features into a probability score with better "
        "precision than the rule-based system. The model is stored as a joblib file and "
        "loaded into memory once at startup."
    )

    pdf.subsection("Layer 3: Claude API (Optional)")
    pdf.body(
        "Only triggered when Layer 2 score falls in [0.35, 0.65] - the genuine uncertain "
        "zone. Uses claude-haiku-4-5 for text (fast, cheap) and claude-sonnet-4-6 Vision "
        "for images. The final score is a weighted blend: 40% ML + 60% Claude. "
        "Disabled if ANTHROPIC_API_KEY is not set."
    )

    # ---- SECTION 7: PERFORMANCE ----
    pdf.section_title("7", "Performance Metrics")

    pdf.body(
        f"Training dataset: {n_samples:,} texts from two public HuggingFace datasets.\n"
        "Evaluation: 5-fold Stratified K-Fold cross-validation."
    )

    pdf.subsection("Cross-validation Results (5-fold)")
    phases = [
        ("Fase 1 - Heuristic v1 (5 features, manual)", 0.71, C_AMBER),
        ("Fase 2.5 - RF on HC3+dmitva (5 features, 6k)", 0.853, C_AMBER),
        ("Fase 3 - RF v3 (12 features, 16k samples)", cv_acc, C_GREEN),
    ]
    pdf.ln(2)
    for label, acc, color in phases:
        pdf.accuracy_bar(label, acc, color)
    pdf.ln(4)

    pdf.body(
        f"The v3 model represents a {((cv_acc - 0.853) * 100):.1f} percentage point improvement "
        "over v2, achieved through feature engineering (7 new features) and a larger, more "
        "diverse training set (16k vs 6k examples)."
    )

    pdf.info_box(
        "Most Discriminative Feature",
        f"Burstiness (sentence length CV) accounts for {feat_imp.get('burstiness', 0.301):.1%} "
        "of model decisions. This aligns with the linguistic research: human writing has dramatic "
        "sentence rhythm variation; LLM output is rhythmically monotonous.",
        C_ACCENT,
    )

    # ---- SECTION 8: CONTINUOUS LEARNING ----
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("8", "Continuous Learning System")

    pdf.body(
        "The platform implements two complementary learning mechanisms that allow the model "
        "to improve over time from real user traffic - without retraining from scratch."
    )

    pdf.bullet([
        ("SGDClassifier", "Online learning via partial_fit(). Each user correction via "
            "POST /feedback immediately updates the SGD model. Zero latency, happens in "
            "the same request. Persisted to disk via joblib."),
        ("RF Retrain", "When RETRAIN_THRESHOLD (default: 50) confirmed examples accumulate, "
            "the Random Forest is retrained in a background FastAPI task. The old model "
            "continues serving until the retrain completes."),
        ("Auto-labeling", "Each confident detection (verdict != 'uncertain') is automatically "
            "saved to the training buffer. User corrections override the auto-label."),
        ("Buffer Persistence", "All training examples are stored in the SQLite TrainingExample "
            "table - audit trail, reproducibility, manual review possible."),
    ])

    pdf.body(
        "This creates a virtuous cycle: the more the system is used, the more labeled examples "
        "it accumulates, the better the model becomes - without any manual annotation work."
    )

    # ---- SECTION 9: COMPETITIVE ADVANTAGE ----
    pdf.section_title("9", "Competitive Differentiation")

    headers = ["Capability", "AI Detector", "LLM-only Detector", "GPTZero / Turnitin"]
    col_w = [55, 32, 38, 45]
    rows = [
        ("Hallucination-free", "YES", "NO", "Partial"),
        ("Cost per query", "$0 (ML only)", "$0.001-0.01", "$0.02-0.10"),
        ("Inference latency", "<50ms", "500-3000ms", "200-800ms"),
        ("Explainable", "YES (features)", "NO", "Limited"),
        ("Privacy (on-premise)", "YES", "NO (API)", "NO (API)"),
        ("Continuous learning", "YES", "NO", "NO"),
        ("Image detection", "YES (Vision)", "Limited", "NO"),
        ("Jailbreak resistant", "YES", "NO", "Partial"),
        ("Open architecture", "YES", "NO", "NO"),
    ]

    pdf.set_fill_color(*C_DARK)
    pdf.set_text_color(*C_WHITE)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(15)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    for i, row in enumerate(rows):
        fill = i % 2 == 0
        self_fill = (248, 250, 252) if fill else C_WHITE
        pdf.set_fill_color(*self_fill)
        pdf.set_x(15)
        for j, (val, w) in enumerate(zip(row, col_w)):
            if j == 1:
                color = C_GREEN if val == "YES" or val.startswith("$0") or "<50" in val else C_RED
                pdf.set_text_color(*color)
                pdf.set_font("Helvetica", "B", 9)
            elif j > 1:
                is_neg = val in ("NO", "NO (API)") or "1-0.01" in val or "3000" in val
                pdf.set_text_color(*C_RED if is_neg else C_DARK)
                pdf.set_font("Helvetica", "", 9)
            else:
                pdf.set_text_color(*C_DARK)
                pdf.set_font("Helvetica", "", 9)
            pdf.cell(w, 6, val, border=1, fill=fill, align="C" if j > 0 else "L")
        pdf.ln()
    pdf.ln(4)

    # ---- SECTION 10: TECH STACK ----
    pdf.section_title("10", "Technical Stack")
    pdf.bullet([
        ("Backend", "FastAPI (Python 3.12) - async, OpenAPI-documented."),
        ("Database", "SQLite + SQLAlchemy async + aiosqlite. Production: swap to PostgreSQL."),
        ("ML Runtime", "scikit-learn RandomForestClassifier + SGDClassifier, joblib serialization."),
        ("Claude Integration", "anthropic SDK - claude-haiku-4-5 (text), claude-sonnet-4-6 (Vision)."),
        ("API", "POST /api/v1/detect (text + image), POST /api/v1/detect/feedback."),
        ("Lab", "Jupyter Notebook (detector_evolution.ipynb) - phased documentation."),
        ("Deployment", "uvicorn, port 8002. Container-ready."),
    ])

    # ---- SECTION 11: ROADMAP ----
    pdf.add_page()
    pdf.set_y(25)
    pdf.section_title("11", "Roadmap")

    pdf.roadmap_item(
        "Phase 1", "Heuristic Engine",
        ["5 statistical features", "Rule-based scoring", "REST API (FastAPI)"],
        "DONE", C_GREEN,
    )
    pdf.roadmap_item(
        "Phase 2", "Machine Learning - Random Forest",
        ["6 real-world examples -> RF trained", "85.3% accuracy", "Joblib model persistence"],
        "DONE", C_GREEN,
    )
    pdf.roadmap_item(
        "Phase 2.5", "Real Dataset Training",
        ["HC3 + dmitva datasets (6,000 samples)", "Stratified K-Fold CV", "HuggingFace integration"],
        "DONE", C_GREEN,
    )
    pdf.roadmap_item(
        "Phase 3", "Cascade Architecture + Continuous Learning",
        [
            "12-feature vector (7 new features)",
            "16,000 training samples -> 93.4% accuracy",
            "Claude API integration (text + Vision)",
            "SGDClassifier online learning",
            "RF background retrain",
            "POST /detect + POST /feedback endpoints",
        ],
        "DONE", C_GREEN,
    )
    pdf.roadmap_item(
        "Phase 4", "TF-IDF + Embedding Features",
        [
            "Add n-gram TF-IDF features as additional dimensions",
            "Sentence-level embedding similarity (cosine repetition)",
            "Local embedding model (sentence-transformers, no API)",
            "Expected accuracy improvement: 93% -> 95-96%",
        ],
        "NEXT", C_AMBER,
    )
    pdf.roadmap_item(
        "Phase 5", "Perplexity & Token Probability",
        [
            "Local LM perplexity scoring (GPT-2 small, runs locally)",
            "DetectGPT perturbation approach (Mitchell et al., 2023)",
            "Hybrid: perplexity + our 12 features + RF",
            "Target: 96-98% accuracy on academic text",
        ],
        "PLANNED", C_AMBER,
    )
    pdf.roadmap_item(
        "Phase 6", "Production & Scale",
        [
            "PostgreSQL migration (Alembic)",
            "Redis caching for repeated texts",
            "Batch API endpoint",
            "Webhook for async analysis on large documents",
            "Dashboard with analytics",
        ],
        "PLANNED", (130, 130, 180),
    )

    # ---- SAVE ----
    pdf.output(str(PDF_OUT))
    print(f"PDF gerado: {PDF_OUT}")


if __name__ == "__main__":
    build_pdf()
