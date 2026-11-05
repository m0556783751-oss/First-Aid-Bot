"""
app.py - ממשק צ'אט ב-Streamlit.
הרצה: streamlit run app.py
"""

import streamlit as st

from rag import IndexNotBuiltError, ask, load_engine

st.set_page_config(page_title="בוט עזרה ראשונה", page_icon="🩹")

# ---------- יישור לימין (RTL) ----------
# Streamlit מיושר לשמאל כברירת מחדל, אז מוסיפים CSS שהופך את הכיוון לעברית
st.markdown(
    """
    <style>
    .stApp, .stMarkdown, .stChatMessage, .stTextInput, textarea, p, li, h1, h2, h3 {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩹 בוט עזרה ראשונה")
st.caption("שאלו כל שאלה - התשובות מבוססות רק על מסמכי העזרה הראשונה. במקרה חירום - חייגו מיד למד\"א 101.")


# ---------- טעינת המנוע פעם אחת ----------
# st.cache_resource גורם לכך שהמנוע ייטען רק בפעם הראשונה,
# ולא מחדש בכל פעם שהמשתמש שולח הודעה (Streamlit מריץ את כל הקובץ מחדש בכל פעולה)
@st.cache_resource
def get_engine():
    return load_engine()


try:
    engine = get_engine()
except IndexNotBuiltError as e:
    st.error(f"⚠️ {e}")
    st.stop()  # עוצר את הרצת הדף - אין טעם להמשיך בלי אינדקס
except RuntimeError as e:
    # למשל: מפתח API חסר
    st.error(f"⚠️ {e}")
    st.stop()


def show_sources(sources):
    """מציג את רשימת המקורות בתוך expander (אזור שנפתח בלחיצה)."""
    if not sources:
        return
    with st.expander("📄 מקורות"):
        for s in sources:
            st.markdown(f"- **{s['file_name']}** — ציון דמיון: {s['score']}")


# ---------- היסטוריית השיחה ----------
# session_state נשמר בין ההרצות של הדף, כך שההודעות הקודמות לא נמחקות
if "messages" not in st.session_state:
    st.session_state.messages = []

# מציגים את כל ההודעות הקודמות
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))


# ---------- קלט חדש מהמשתמש ----------
question = st.chat_input("כתבו שאלה...")

if question:
    # מציגים ושומרים את שאלת המשתמש
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # מחשבים ומציגים את תשובת הבוט
    with st.chat_message("assistant"):
        with st.spinner("מחפש במסמכים..."):
            try:
                result = ask(engine, question)
                answer, sources = result["answer"], result["sources"]
            except Exception as e:
                # למשל: חריגה ממכסת השכבה החינמית של Gemini, או בעיית רשת
                answer, sources = f"אירעה שגיאה: {e}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
