
"""
HBB Variant Pathogenicity Predictor
Feature extraction, model training, and SHAP explanations.

IMPORTANT:
Training labels are SYNTHETIC (rule-based + noise).
This is an educational prototype, not a clinical diagnostic tool.
Replace build_training_data() with validated ClinVar data for
real research and independently validate the resulting model.
"""

import base64
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqUtils import seq3

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Reference HBB coding sequence
# ------------------------------------------------------------

REF_CDS = (
    "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGT"
    "GAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCT"
    "GATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCAC"
    "CTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTC"
    "AGGCTCCTGGGCAACGTGCTGGTCTGCGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCC"
    "TATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAA"
)

REF_PROTEIN = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAH"
    "LDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH*"
)

assert str(Seq(REF_CDS).translate()) == REF_PROTEIN, (
    "REF_CDS does not encode HBB protein"
)

BLOSUM = substitution_matrices.load("BLOSUM62")

FEATURES = [
    "Amino acid change",
    "A-T content",
    "Conservation score",
    "k-mer (CTG)",
    "Nucleotide position",
]

WINDOW = 30

# Placeholder conservation values.
# Replace with experimentally supported conservation data
# before using the model for research.
KNOWN_CONSERVED = {
    6: 0.98,
    58: 0.95,
    63: 1.0,
    92: 1.0,
    102: 0.9,
    106: 0.9,
}


# ------------------------------------------------------------
# Conservation score
# ------------------------------------------------------------

def conservation(aa_pos: int) -> float:
    if aa_pos in KNOWN_CONSERVED:
        return KNOWN_CONSERVED[aa_pos]

    return float(
        np.random.default_rng(aa_pos).uniform(0.3, 0.9)
    )


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

def featurize(
    seq: str,
    nt_idx: int,
    ref_aa: str,
    alt_aa: str
) -> list:

    aa_pos = nt_idx // 3

    if ref_aa == alt_aa:
        severity = 0.0

    elif alt_aa == "*" or aa_pos == 0:
        severity = 1.0

    else:
        severity = float(
            np.clip(
                (4 - BLOSUM[ref_aa][alt_aa]) / 8,
                0,
                1
            )
        )

    lo = max(0, nt_idx - WINDOW)
    hi = min(len(seq), nt_idx + WINDOW + 1)

    win = seq[lo:hi]

    at_content = (
        win.count("A") + win.count("T")
    ) / len(win)

    return [
        severity,
        at_content,
        conservation(aa_pos),
        win.count("CTG"),
        nt_idx / len(REF_CDS),
    ]


# ------------------------------------------------------------
# Codon analysis
# ------------------------------------------------------------

def codon_change(seq: str, nt_idx: int):

    codon_start = (nt_idx // 3) * 3

    ref_codon = REF_CDS[codon_start:codon_start + 3]
    alt_codon = seq[codon_start:codon_start + 3]

    ref_aa = str(Seq(ref_codon).translate())
    alt_aa = str(Seq(alt_codon).translate())

    return ref_codon, alt_codon, ref_aa, alt_aa


# ------------------------------------------------------------
# Synthetic training data
# ------------------------------------------------------------

def build_training_data(n=3000, seed=0):

    rng = np.random.default_rng(seed)

    X = []
    y = []

    for _ in range(n):

        idx = int(
            rng.integers(0, len(REF_CDS) - 3)
        )

        alt_base = rng.choice(
            [b for b in "ACGT" if b != REF_CDS[idx]]
        )

        mut = (
            REF_CDS[:idx]
            + alt_base
            + REF_CDS[idx + 1:]
        )

        _, _, ref_aa, alt_aa = codon_change(mut, idx)

        features = featurize(
            mut, idx, ref_aa, alt_aa
        )

        # Synthetic labels for demonstration only.
        logit = 6 * (
            0.6 * features[0]
            + 0.4 * features[2]
            - 0.5
        ) + 0.8 * (features[1] - 0.5)

        probability = 1 / (1 + np.exp(-logit))

        label = int(rng.random() < probability)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


# ------------------------------------------------------------
# Train the machine-learning model
# ------------------------------------------------------------

X, y = build_training_data()

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

MODEL = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    random_state=42
)

MODEL.fit(X_tr, y_tr)


# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------

METRICS = {
    "accuracy": round(
        accuracy_score(y_te, MODEL.predict(X_te)), 3
    ),
    "roc_auc": round(
        roc_auc_score(
            y_te,
            MODEL.predict_proba(X_te)[:, 1]
        ),
        3
    ),
    "n_train": len(X_tr),
    "n_test": len(X_te),
    "note": (
        "Trained on synthetic labels. "
        "Metrics do not represent clinical performance."
    ),
}


# ------------------------------------------------------------
# SHAP explanations
# ------------------------------------------------------------

EXPLAINER = shap.TreeExplainer(MODEL)

_bg = X_te[:300]
_bg_sv = EXPLAINER.shap_values(_bg)

GLOBAL_IMPORTANCE = dict(
    zip(
        FEATURES,
        np.abs(_bg_sv).mean(axis=0).round(3).tolist()
    )
)


def _summary_png() -> bytes:

    plt.figure(figsize=(6, 3.6))

    shap.summary_plot(
        _bg_sv,
        _bg,
        feature_names=FEATURES,
        show=False
    )

    plt.tight_layout()

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        dpi=130,
        transparent=True
    )

    plt.close()

    return buf.getvalue()


SUMMARY_PNG = _summary_png()


# ------------------------------------------------------------
# Mutation helper and example variants
# ------------------------------------------------------------

def mutate(idx: int, base: str) -> str:

    return (
        REF_CDS[:idx]
        + base
        + REF_CDS[idx + 1:]
    )


EXAMPLES = {
    "HbS (E6V) - Pathogenic": mutate(19, "T"),
    "HbC (E6K) - Pathogenic": mutate(18, "A"),
    "Synonymous (E6E) - Likely benign": mutate(20, "A"),
    "Reference HBB (no variant)": REF_CDS,
}


# ------------------------------------------------------------
# FASTA parsing
# ------------------------------------------------------------

def parse_fasta(text: str) -> str:

    body = "".join(
        line.strip()
        for line in text.splitlines()
        if not line.startswith(">")
    )

    return re.sub(
        r"[^ACGT]",
        "",
        body.upper().replace("U", "T")
    )


# ------------------------------------------------------------
# Analyze DNA and calculate the displayed score
# ------------------------------------------------------------

def analyze(text: str) -> dict:

    seq = parse_fasta(text)

    if len(seq) < 30:
        raise ValueError(
            "Sequence too short. Paste at least 30 nt "
            "of the HBB coding sequence starting at ATG."
        )

    n = min(len(seq), len(REF_CDS))

    diffs = [
        i
        for i in range(n)
        if seq[i] != REF_CDS[i]
        and (i // 3 * 3 + 3) <= len(seq)
    ]

    base = {
        "gene": "HBB (Hemoglobin subunit beta)",
        "length": len(seq),
        "sequence": seq,
    }

    # Case 1: Exact reference sequence.
    # This is a reference-match indicator, NOT model confidence.
    
    if seq == REF_CDS:
        # Extract features for the reference sequence.
        # These describe a reference baseline, not a real variant.
        feats = featurize(REF_CDS, 0, "M", "M")
        arr = np.array([feats])

        probabilities = MODEL.predict_proba(arr)[0]
        p = float(probabilities[1])
        confidence = max(p, 1.0 - p)

        # Calculate SHAP contributions for the baseline features.
        sv = EXPLAINER.shap_values(arr)[0]

        shap_data = sorted(
            [
                {
                    "feature": feature,
                    "value": round(float(value), 3)
                }
                for feature, value in zip(FEATURES, sv)
            ],
            key=lambda item: -abs(item["value"]),
        )

        return {
            **base,
            "variant": None,
            "label": "Benign",
            "confidence": 1.0,
            "confidence_type": "reference_match",
            "message": (
                "Exact match to the HBB reference sequence. "
                "No variant detected. The 100% score represents "
                "a reference match, not clinical confidence."
            ),
            "shap": shap_data,
        }


    # Case 2: No differences in the compared region.
    # Do not assign 100% when only a partial or extended sequence
    # matches the reference.
    if not diffs:

        return {
            **base,
            "variant": None,
            "label": "Uncertain",
            "confidence": None,
            "confidence_type": None,
            "message": (
                "No differences detected in the compared region. "
                "The sequence is not an exact full reference match, "
                "so a complete variant assessment is not available."
            ),
            "shap": [],
        }

    # Case 3: One or more differences detected.
    # This prototype analyzes the first detected nucleotide change.
    idx = diffs[0]

    ref_codon, alt_codon, ref_aa, alt_aa = codon_change(
        seq, idx
    )

    feats = featurize(
        seq, idx, ref_aa, alt_aa
    )

    arr = np.array([feats])

    probabilities = MODEL.predict_proba(arr)[0]

    # Probability of the model's positive class.
    p = float(probabilities[1])

    pathogenic = p >= 0.5

    # Probability associated with the predicted class.
    model_score = max(p, 1.0 - p)

    shap_values = EXPLAINER.shap_values(arr)[0]

    aa_pos = idx // 3

    return {
        **base,
        "label": "Pathogenic" if pathogenic else "Benign",
        "confidence": round(float(model_score), 3),
        "confidence_type": "synthetic_model_score",
        "message": (
            f"DNA variation detected at nucleotide position {idx + 1}: "
            f"{REF_CDS[idx]} → {seq[idx]}. "
            f"The corresponding amino acid change is "
            f"{ref_aa}{aa_pos}{alt_aa}. "
            f"The model predicts this variant as "
            f"{'Pathogenic' if pathogenic else 'Benign'} "
            f"with a score of {model_score:.2f} "
            f"({model_score * 100:.0f}%). "
            "This is an experimental prediction based on synthetic "
            "training labels and is not clinically validated."  
        ),
        "variant": {
            "position": aa_pos,
            "nt_position": idx + 1,
            "codon_start": idx // 3 * 3,
            "ref_codon": (
                f"{ref_codon} ({seq3(ref_aa)})"
            ),
            "alt_codon": (
                f"{alt_codon} ({seq3(alt_aa)})"
            ),
            "aa_change": (
                f"{ref_aa}{aa_pos}{alt_aa}"
            ),
            "conservation": feats[2],
            "extra_variants": len(diffs) - 1,
        },
        "shap": sorted(
            [
                {
                    "feature": feature,
                    "value": round(float(value), 3),
                }
                for feature, value in zip(
                    FEATURES, shap_values
                )
            ],
            key=lambda item: -abs(item["value"]),
        ),
    }


# ------------------------------------------------------------
# Return the global SHAP summary image as Base64
# ------------------------------------------------------------

def summary_png_b64() -> str:

    return base64.b64encode(
        SUMMARY_PNG
    ).decode()
