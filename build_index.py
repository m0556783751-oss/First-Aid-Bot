"""
build_index.py - בונה את האינדקס מאפס.
מריצים אותו פעם אחת, ושוב בכל פעם שמוסיפים או משנים מסמכים בתיקייה data.

השלבים:
1. טעינת קבצי ה-docx
2. מחיקת האוסף הקודם ב-Chroma (אם קיים) ויצירת אוסף חדש
3. חלוקת המסמכים ל-chunks
4. יצירת וקטור (embedding) לכל chunk ושמירה ב-Chroma
"""

import os

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


def load_documents():
    """שלב 1: טוען את כל קבצי ה-Word מהתיקייה data."""
    if not os.path.isdir(config.DATA_DIR):
        raise FileNotFoundError(f"התיקייה {config.DATA_DIR} לא קיימת. צרי אותה ושימי בה קבצי docx.")

    reader = SimpleDirectoryReader(
        input_dir=config.DATA_DIR,
        required_exts=[".docx"],  # רק קבצי Word
        recursive=True,           # כולל תתי-תיקיות
        exclude=["~$*"],          # מדלג על קבצים זמניים ש-Word יוצר כשמסמך פתוח
    )
    documents = reader.load_data()
    # כל מסמך מקבל אוטומטית metadata עם שם הקובץ (file_name) - נשתמש בזה להצגת המקורות
    return documents


def create_fresh_collection():
    """שלב 2: מתחבר ל-Chroma, מוחק אוסף ישן (אם יש) ויוצר אוסף חדש וריק."""
    # PersistentClient שומר את הנתונים לדיסק, כך שהם נשארים גם אחרי סגירת התוכנית
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # מוחקים אוסף קודם כדי שלא יהיו כפילויות כשבונים מחדש
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("אוסף קודם נמחק.")
    except Exception:
        # האוסף לא היה קיים - זה בסדר, ממשיכים
        pass

    # יוצרים אוסף חדש שמשתמש במרחק קוסינוס להשוואת וקטורים
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata=config.COLLECTION_METADATA,
    )
    return collection


def main():
    # מגדירים את מודלי Gemini (ה-embedding ישמש ליצירת הווקטורים)
    config.setup_models()

    print("טוען מסמכים...")
    documents = load_documents()
    if not documents:
        print("לא נמצאו קבצי docx בתיקייה data. אין מה לאנדקס.")
        return
    print(f"נטענו {len(documents)} מסמכים.")

    collection = create_fresh_collection()

    # עוטפים את האוסף של Chroma באובייקט ש-LlamaIndex יודע לעבוד איתו
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # שלב 3: מחלק את הטקסט לחתיכות לפי טוקנים, תוך שמירה על משפטים שלמים ככל האפשר
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # שלב 4: חלוקה + יצירת embedding לכל חתיכה + שמירה ב-Chroma - הכל בפקודה אחת
    print("מחלק ל-chunks ויוצר וקטורים (זה עלול לקחת קצת זמן)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    print(f"סיום! נשמרו {collection.count()} chunks בתיקייה {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
