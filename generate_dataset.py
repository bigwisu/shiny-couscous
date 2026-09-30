import os
import re
import pandas as pd
import psycopg2
import warnings
from dotenv import load_dotenv

# Suppress pandas warning about not using SQLAlchemy for DBAPI connections
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
load_dotenv()

_RESULTS_HEADER = re.compile(
    r'(?:^|\n)\s*(?:'
    r'Results?|Findings?|Conclusions?|Main\s+results?|Principal\s+findings?'
    r')\s*[:\.]?\s*\n',
    re.IGNORECASE,
)

_RESULTS_SENTENCE = re.compile(
    r'\b(?:'
    r'We\s+(?:found|identified|included|enrolled|screened)\b'
    r'|(?:A\s+total\s+of\s+)?\d+\s+(?:studies|trials|articles|papers|records)\s+'
    r'(?:met|were\s+included|were\s+identified|satisfied|fulfilled)'
    r'|Among\s+the\s+\d+'
    r')\b',
    re.IGNORECASE,
)

def _strip_results(sr_abstract: str) -> str:
    """Return SR abstract up to (not including) the results section."""
    if not isinstance(sr_abstract, str):
        return ""
    m = _RESULTS_HEADER.search(sr_abstract) or _RESULTS_SENTENCE.search(sr_abstract)
    return sr_abstract[:m.start()].strip() if m else sr_abstract[:600]

def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        dbname=os.getenv("PG_DATABASE")
    )

def generate_dataset():
    conn = get_connection()
    
    query_tp = """
        SELECT 
            sr.sr_pmid, 
            sr.title AS sr_title, 
            sr.abstract AS sr_abstract, 
            d.pmid AS doc_pmid, 
            d.title AS doc_title, 
            d.abstract AS doc_abstract, 
            1 AS label
        FROM sr_document_mappings m
        JOIN systematic_reviews sr ON m.sr_pmid = sr.sr_pmid
        JOIN pubmed_documents d ON m.pmid = d.pmid
        WHERE m.is_inclusion = TRUE
        ORDER BY RANDOM()
        LIMIT 40
    """
    
    query_fp = """
        SELECT 
            sr.sr_pmid, 
            sr.title AS sr_title, 
            sr.abstract AS sr_abstract, 
            d.pmid AS doc_pmid, 
            d.title AS doc_title, 
            d.abstract AS doc_abstract, 
            0 AS label
        FROM sr_document_mappings m
        JOIN systematic_reviews sr ON m.sr_pmid = sr.sr_pmid
        JOIN pubmed_documents d ON m.pmid = d.pmid
        WHERE m.is_inclusion = FALSE
        ORDER BY RANDOM()
        LIMIT 160
    """
    
    print("Fetching True Positives...")
    df_tp = pd.read_sql(query_tp, conn)
    
    print("Fetching False Positives...")
    df_fp = pd.read_sql(query_fp, conn)
    
    conn.close()
    
    # Combine and shuffle
    df = pd.concat([df_tp, df_fp], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Construct sr_objective using logic from jev_client.py
    df['sr_objective'] = df['sr_abstract'].apply(_strip_results)
    
    # Reorder columns and drop sr_title and sr_abstract
    cols = ['sr_pmid', 'sr_objective', 'doc_pmid', 'doc_title', 'doc_abstract', 'label']
    df = df[cols]
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully at {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"True Positives: {len(df[df['label'] == 1])}")
    print(f"False Positives: {len(df[df['label'] == 0])}")

if __name__ == "__main__":
    generate_dataset()
