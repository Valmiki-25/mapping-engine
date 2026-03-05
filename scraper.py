import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE = "Output/coursera_courses.csv"


def get_course_links(query):

    url = f"https://www.coursera.org/search?query={query.replace(' ', '+')}"
    print("Searching:", url)

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")

        courses = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            name = a.get_text(strip=True)

            if any(x in href for x in ["/learn/", "/specializations/", "/professional-certificates/"]):
                full_link = "https://www.coursera.org" + href if href.startswith("/") else href

                if full_link not in seen and name:
                    courses.append((name, full_link))
                    seen.add(full_link)

            if len(courses) >= 30:
                break

        return courses

    except Exception as e:
        print("Error fetching links:", e)
        return []


def fetch_skills(soup):

    skills = []

    for h2 in soup.find_all("h2"):
        if "skills" in h2.get_text(strip=True).lower():
            ul = h2.find_next("ul")
            if ul:
                skills = [a.get_text(strip=True) for a in ul.find_all("a")]
            break

    return skills


def extract_rating(soup):

    import re
    tag = soup.find(attrs={"aria-label": re.compile(r"\d\.\d stars")})

    if tag:
        return tag.get("aria-label").split()[0]

    return ""


def scrape_course(course):

    name, link = course

    try:
        r = requests.get(link, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")
        skills = fetch_skills(soup)
        rating = extract_rating(soup)

        return {
            "Name": name,
            "Link": link,
            "Skills": ", ".join(skills),
            "Rating": rating
        }

    except:
        return {
            "Name": name,
            "Link": link,
            "Skills": "",
            "Rating": ""
        }


def scrape_courses(skill):

    print("\n--- SCRAPING COURSES ---")

    courses = get_course_links(skill)

    results = []

    for c in courses:
        results.append(scrape_course(c))

    import os
    os.makedirs("Output", exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)

    return df