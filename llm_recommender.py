import os
import json
import pandas as pd
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


OUTPUT_FILE = "Output/final_llm_recommendation.csv"


# -----------------------------------
# Recommendation Pipeline
# -----------------------------------

def recommend(skill):

    print("\n==============================")
    print("COURSE RECOMMENDATION SYSTEM")
    print("==============================")

    # -----------------------------------
    # Step 1: Scrape courses
    # -----------------------------------

    scrape_courses(skill)

    # -----------------------------------
    # Step 2: Retrieve top courses
    # -----------------------------------

    courses = retrieve_courses(skill)

    # -----------------------------------
    # Prepare course list for LLM
    # -----------------------------------

    course_list = []

    for _, row in courses.iterrows():

        course_list.append({
            "course_name": row["Name"],
            "skills": row["Skills"],
            "rating": float(row["Rating"]) if row["Rating"] != "" else None,
            "relevance_score": float(row["score"])
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
2. Prefer higher relevance_score (semantic match).
3. Prefer higher rating when available.
4. If a course has NO rating but its relevance_score is high, it can still be selected.
5. You MUST choose only from the provided course_name values.
6. If NONE of the courses are clearly related to the skill, return:

{{
"best_course": "None",
"reason": "No related courses found"
}}

Return STRICT JSON only.

Format:

{{
"best_course": "course name OR None",
"reason": "short explanation"
}}
"""

    # -----------------------------------
    # Call Groq LLM
    # -----------------------------------

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {"role": "user", "content": prompt}
        ],

        temperature=0.2
    )

    result_text = response.choices[0].message.content.strip()

    print("\nLLM Recommendation:\n")
    print(result_text)

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
    # Save result
    # -----------------------------------

    output_data = {
        "Skill": skill,
        "Best Course": result_json.get("best_course"),
        "Reason": result_json.get("reason")
    }

    pd.DataFrame([output_data]).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:", OUTPUT_FILE)


# -----------------------------------
# Run
# -----------------------------------

if __name__ == "__main__":

    skill = input("\nEnter Skill: ")

    recommend(skill)