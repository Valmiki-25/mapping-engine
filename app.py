import os
import json
import pandas as pd
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

from scraper import scrape_courses
from embeddings import retrieve_courses


# -----------------------------------
# Load ENV variables
# -----------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

HISTORY_FILE = "Output/recommendation_history.json"


# -----------------------------------
# Save recommendation history
# -----------------------------------

def save_history(data):

    os.makedirs("Output", exist_ok=True)

    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []

    history.append(data)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


# -----------------------------------
# Recommendation Function
# -----------------------------------

def recommend(skill):

    st.write("### 🔎 Scraping Courses")

    scrape_courses(skill)

    st.write("### 📚 Retrieving Relevant Courses")

    courses = retrieve_courses(skill)

    if courses.empty:
        return {
            "best_course": "None",
            "reason": "No related courses found"
        }, courses

    courses = courses.copy()

    # Convert rating safely
    if "Rating" in courses.columns:
        courses["Rating"] = pd.to_numeric(courses["Rating"], errors="coerce")

    # Convert relevance score safely
    if "score" in courses.columns:
        courses["score"] = pd.to_numeric(courses["score"], errors="coerce")

    courses = courses.fillna("")

    # -----------------------------------
    # Detect no relevant course
    # -----------------------------------

    if courses["score"].max() < 0.25:

        history_data = {
            "skill": skill,
            "course_name": "None",
            "link": None,
            "skills": None,
            "explanation": "No related courses found"
        }

        save_history(history_data)

        return {
            "best_course": "None",
            "reason": "No related courses found"
        }, courses

    # -----------------------------------
    # Send only top 5 courses to LLM
    # -----------------------------------

    courses = courses.sort_values("score", ascending=False).head(5)

    # -----------------------------------
    # Prepare course list for LLM
    # -----------------------------------

    course_list = []

    for _, row in courses.iterrows():

        course_list.append({
            "course_name": row["Name"],
            "skills": row["Skills"],
            "rating": row["Rating"] if row["Rating"] != "" else None,
            "relevance_score": row["score"]
        })

    # -----------------------------------
    # Prompt
    # -----------------------------------

    prompt = f"""
You are an AI learning advisor.

User skill:
{skill}

Below are retrieved courses.

Courses:
{course_list}

Decision Rules:

1. Choose the BEST course for the user's skill.
2. Prefer higher relevance_score.
3. Prefer higher rating when available.
4. Only choose a course that clearly teaches the skill.
5. If no course teaches the skill return:

{{
"best_course": "None",
"reason": "No related courses found"
}}

IMPORTANT RULES:
- Do NOT generate code
- Do NOT explain reasoning steps
- Do NOT add markdown
- Return ONLY valid JSON

Output Format:

{{
"best_course": "course name OR None",
"reason": "short explanation"
}}
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[{"role": "user", "content": prompt}],

        temperature=0.2
    )

    result_text = response.choices[0].message.content.strip()

    # -----------------------------------
    # Clean JSON if model adds ```json
    # -----------------------------------

    result_text = result_text.replace("```json", "").replace("```", "").strip()

    # -----------------------------------
    # Parse JSON safely
    # -----------------------------------

    try:
        result_json = json.loads(result_text)

    except:
        result_json = {
            "best_course": "Parsing Error",
            "reason": result_text
        }

    # -----------------------------------
    # Extract selected course details
    # -----------------------------------

    selected_course = result_json.get("best_course")

    course_link = None
    course_skills = None

    if selected_course != "None":

        match = courses[courses["Name"] == selected_course]

        if not match.empty:

            course_link = match.iloc[0]["Link"] if "Link" in match.columns else None
            course_skills = match.iloc[0]["Skills"]

    # -----------------------------------
    # Save history
    # -----------------------------------

    history_data = {
        "skill": skill,
        "course_name": selected_course,
        "link": course_link,
        "skills": course_skills,
        "explanation": result_json.get("reason")
    }

    save_history(history_data)

    return result_json, courses


# -----------------------------------
# Streamlit UI
# -----------------------------------

st.set_page_config(
    page_title="AI Learning Advisor",
    layout="wide"
)

st.title("🎓 AI Learning Advisor")
st.write("Find the best Coursera course for your skill using AI.")

skill = st.text_input("Enter your skill")

if st.button("Recommend Course"):

    if skill.strip() == "":
        st.warning("Please enter a skill")

    else:

        with st.spinner("Processing..."):

            result, courses = recommend(skill)

        st.subheader("🏆 Best Course Recommendation")

        if result["best_course"] == "None":

            st.error("No related courses found")

        else:

            st.success(result["best_course"])

            # Show course link
            match = courses[courses["Name"] == result["best_course"]]

            if not match.empty and "Link" in match.columns:
                link = match.iloc[0]["Link"]
                if link:
                    st.markdown(f"[Open Course]({link})")

        st.write("**Reason:**", result["reason"])

        st.subheader("📚 Top Retrieved Courses")

        st.dataframe(
            courses,
            use_container_width=True
        )