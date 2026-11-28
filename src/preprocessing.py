import re
from pathlib import Path
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def _ensure_nltk_resources() -> None:
    # NTLK check
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))

DEFAULT_TEXT_COLS = [
    "job_title", "role_family", "seniority", "industry",
    "city", "province", "work_mode", "skills", "tools"
]

OUTPUT_COLS = [
    "job_id", "job_title", "city", "work_mode",
    "salary_mid_idr", "document_text", "clean_text", "token_count"
]

def preprocess_text(text: str) -> list[str]:
    #stopwords removal(except + and #), lowercasing, and tokenization for BM25 indexing
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower() #lowercasing
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text) #remove special characters except + and #
    tokens = word_tokenize(text) #tokenization
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens

def preprocess_to_string(text: str) -> str:
    # spacing returning tokens to string for BM25 indexing
    return " ".join(preprocess_text(text))

def build_corpus_dataframe(df: pd.DataFrame, text_cols: list[str] | None = None) -> pd.DataFrame:
    #merge columns to document_text (for dense) and clean_text (for BM25)
    cols = text_cols if text_cols is not None else DEFAULT_TEXT_COLS
    df_out = df.copy()

    for col in cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna("").astype(str)

    available_cols = [c for c in cols if c in df_out.columns]
    df_out["document_text"] = (
        df_out[available_cols]
        .agg(" ".join, axis=1)
        .str.replace(";", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df_out["tokens"] = df_out["document_text"].apply(preprocess_text)
    df_out["clean_text"] = df_out["tokens"].apply(lambda x: " ".join(x))
    df_out["token_count"] = df_out["tokens"].apply(len)

    final_cols = [c for c in OUTPUT_COLS if c in df_out.columns]
    return df_out[final_cols]

def generate_corpus_file(input_path: Path, output_path: Path) -> None:
    #reading raw dataset from local path and saving ready-to-index corpus
    df_raw = pd.read_csv(input_path, sep=",")
    df_corpus = build_corpus_dataframe(df_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_corpus.to_csv(output_path, index=False)
    print(f"korpus berhasil dibuat ({len(df_corpus)} dokumen) -> {output_path}")

if __name__ == "__main__":
    # root folder
    project_root = Path(__file__).resolve().parent.parent
    raw_csv = project_root / "data" / "jobs_clean.csv"
    ready_csv = project_root / "data" / "corpus_ready.csv"

    # testing query token
    sample_query = "Junior C++ & Python Developer (Remote - Jakarta)"
    print("Kueri asli  :", sample_query)
    print("Hasil token :", preprocess_text(sample_query))

    # create corpus_ready.csv
    if raw_csv.exists():
        generate_corpus_file(raw_csv, ready_csv)