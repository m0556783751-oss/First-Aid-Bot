"""
rag.py - הלוגיקה של הבוט: טעינת האינדקס, הרחבת שאלה, חיפוש ויצירת תשובה.
"""

import chromadb
from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


class IndexNotBuiltError(Exception):
    """שגיאה מיוחדת שנזרקת כשהאינדקס עוד לא נבנה (צריך להריץ build_index.py)."""


# ---------- תבניות פרומפט ----------

# התבנית שמורה למודל לענות אך ורק מתוך ההקשר.
# {context_str} - LlamaIndex ימלא כאן את ה-chunks שנמצאו
# {query_str}   - LlamaIndex ימלא כאן את השאלה
QA_PROMPT = PromptTemplate(
    "אתה עוזר שעונה על שאלות בנושא עזרה ראשונה.\n"
    "לפניך קטעים מתוך מסמכי העזרה הראשונה:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "הוראות:\n"
    "1. ענה אך ורק לפי הקטעים שלמעלה.\n"
    "2. אל תנחש ואל תשתמש בידע כללי או בידע מחוץ לקטעים.\n"
    f"3. אם התשובה לא מופיעה בקטעים, ענה בדיוק: \"{config.NO_INFO_MESSAGE}\".\n"
    "4. ענה בעברית, בצורה קצרה וברורה.\n\n"
    "שאלה: {query_str}\n"
    "תשובה: "
)

# התבנית להרחבת שאלה קצרה לשאלה מלאה
EXPAND_PROMPT = PromptTemplate(
    "המשתמש שאל שאלה קצרה: \"{question}\"\n"
    "נסח אותה מחדש כשאלה מלאה וברורה בעברית, במשפט אחד.\n"
    "שמור על הנושא המקורי של השאלה ואל תוסיף נושאים או פרטים חדשים.\n"
    "החזר רק את השאלה המנוסחת, בלי שום הסבר נוסף."
)


# ---------- פונקציות ----------

def load_index() -> VectorStoreIndex:
    """טוען את האינדקס הקיים מ-Chroma (בלי לחשב embeddings מחדש)."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # מבקשים את האוסף עם הגדרת מדידת קוסינוס.
    # אם האוסף לא קיים - הוא ייווצר ריק, ונזהה את זה בבדיקת count למטה.
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata=config.COLLECTION_METADATA,
    )

    if collection.count() == 0:
        raise IndexNotBuiltError("האינדקס עוד לא נבנה או שהוא ריק. ודאי שיש קבצי docx בתיקייה data והריצי: python build_index.py")

    # Chroma לא משנה את שיטת המדידה של אוסף שכבר קיים, ולכן מוודאים שהוא באמת נבנה עם קוסינוס
    space = (collection.metadata or {}).get("hnsw:space")
    if space != config.COLLECTION_METADATA["hnsw:space"]:
        raise IndexNotBuiltError(f"האוסף נבנה עם מדידת '{space}' ולא קוסינוס. הריצי מחדש: python build_index.py")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def expand_question(question: str) -> str:
    """
    אם השאלה קצרה (פחות מ-5 מילים) - מבקש מה-LLM לנסח אותה כשאלה מלאה.
    שאלה מלאה נותנת וקטור מדויק יותר, ולכן חיפוש טוב יותר.
    אחרת - מחזיר את השאלה המקורית כמו שהיא.
    """
    if len(question.split()) >= config.SHORT_QUESTION_WORDS:
        return question

    response = Settings.llm.complete(EXPAND_PROMPT.format(question=question))
    expanded = response.text.strip()
    # אם משום מה חזרה תשובה ריקה - נשתמש בשאלה המקורית
    return expanded or question


def create_query_engine(index: VectorStoreIndex):
    """יוצר מנוע שאלות: חיפוש Top-K, סינון לפי סף דמיון, ויצירת תשובה עם הפרומפט שלנו."""
    return index.as_query_engine(
        similarity_top_k=config.TOP_K,
        # אחרי השליפה - זורקים chunks עם ציון דמיון נמוך מהסף
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)],
        text_qa_template=QA_PROMPT,
    )


def load_engine():
    """מגדיר את המודלים, טוען את האינדקס ומחזיר query engine מוכן לשימוש."""
    config.setup_models()
    index = load_index()
    return create_query_engine(index)


def ask(query_engine, question: str) -> dict:
    """
    מקבל שאלה ומחזיר מילון עם:
    - answer: התשובה
    - sources: רשימת מקורות, לכל אחד שם קובץ וציון דמיון
    - search_query: השאלה שבפועל חיפשנו לפיה (אחרי הרחבה, אם הייתה)
    """
    # שלב 1: הרחבת שאלה (רק אם היא קצרה)
    search_query = expand_question(question)

    # שלב 2: חיפוש + סינון + יצירת תשובה
    response = query_engine.query(search_query)

    # שלב 3: אם אף chunk לא עבר את הסף - לא שולחים תשובה מומצאת
    if not response.source_nodes:
        return {"answer": config.NO_INFO_MESSAGE, "sources": [], "search_query": search_query}

    # שלב 4: איסוף המקורות
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file_name": node.metadata.get("file_name", "לא ידוע"),
            "score": round(node.score or 0.0, 3),
        })

    return {"answer": str(response).strip(), "sources": sources, "search_query": search_query}


# אפשר להריץ את הקובץ הזה ישירות כדי לבדוק את הבוט בטרמינל, בלי Streamlit
if __name__ == "__main__":
    engine = load_engine()
    while True:
        q = input("\nשאלה (Enter ריק ליציאה): ").strip()
        if not q:
            break
        result = ask(engine, q)
        print("\nתשובה:", result["answer"])
        for s in result["sources"]:
            print(f"  - {s['file_name']} (ציון: {s['score']})")
