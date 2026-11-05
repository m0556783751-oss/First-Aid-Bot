"""
config.py - כל ההגדרות והקבועים של הפרויקט במקום אחד.
כל שאר הקבצים מייבאים מכאן, כך שאם רוצים לשנות הגדרה - משנים רק כאן.
"""

import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# טוענים את המשתנים מקובץ .env (שם נמצא המפתח GOOGLE_API_KEY)
load_dotenv()

# ---------- נתיבים ----------
# התיקייה שבה נמצא הקובץ הזה = תיקיית הפרויקט
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# תיקיית המסמכים (קבצי Word)
DATA_DIR = os.path.join(BASE_DIR, "data")
# התיקייה שבה ChromaDB שומר את מסד הנתונים הווקטורי
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
# שם האוסף (collection) בתוך Chroma
COLLECTION_NAME = "first_aid"
# הגדרת האוסף: מדידת מרחק לפי קוסינוס (ולא המרחק האוקלידי שהוא ברירת המחדל של Chroma)
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# ---------- מודלים (Google Gemini) ----------
# מודל ה-embedding - אותו מודל משמש גם לבניית האינדקס וגם להמרת השאלה לווקטור
EMBED_MODEL_NAME = "gemini-embedding-001"
# מודל השפה שמנסח את התשובה
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

# ---------- חלוקה ל-chunks ----------
CHUNK_SIZE = 512      # גודל כל חתיכה בטוקנים
CHUNK_OVERLAP = 50    # כמה טוקנים חופפים בין חתיכה לחתיכה הבאה

# ---------- חיפוש ----------
TOP_K = 4                    # כמה חתיכות דומות לשלוף לכל שאלה
SIMILARITY_CUTOFF = 0.5      # ציון דמיון מינימלי - מתחתיו החתיכה נזרקת
SHORT_QUESTION_WORDS = 5     # שאלה עם פחות מילים מזה תורחב בעזרת ה-LLM

# ההודעה שהבוט מחזיר כשלא נמצא מידע רלוונטי
NO_INFO_MESSAGE = "לא מצאתי מידע על כך במסמכי העזרה הראשונה"


def get_api_key() -> str:
    """מחזיר את מפתח ה-API, או זורק שגיאה ברורה אם הוא חסר."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "לא נמצא GOOGLE_API_KEY. צרי קובץ .env לפי הדוגמה ב-.env.example והכניסי בו את המפתח."
        )
    return api_key


def setup_models() -> None:
    """
    מגדיר את המודלים הגלובליים של LlamaIndex.
    Settings הוא אובייקט גלובלי - כל רכיב של LlamaIndex (אינדקס, query engine)
    משתמש בו אוטומטית, כך שלא צריך להעביר את המודלים לכל פונקציה.
    """
    api_key = get_api_key()

    # מודל השפה - יוצר את התשובות
    Settings.llm = GoogleGenAI(model=LLM_MODEL_NAME, api_key=api_key)

    # מודל ה-embedding - הופך טקסט לווקטור (רשימת מספרים)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL_NAME, api_key=api_key)
