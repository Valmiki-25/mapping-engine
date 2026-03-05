import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager


HEADERS = {"User-Agent": "Mozilla/5.0"}

OUTPUT_FILE = "Output/coursera_courses.csv"


# ------------------------------------------------
# Setup Driver
# ------------------------------------------------

def setup_driver():

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    return driver


# ------------------------------------------------
# Scroll Page
# ------------------------------------------------

def scroll_page(driver):

    last_height = driver.execute_script("return document.body.scrollHeight")

    for _ in range(8):

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)

        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break

        last_height = new_height


# ------------------------------------------------
# Get Links
# ------------------------------------------------

def get_course_links(query):

    driver = setup_driver()

    try:

        url = f"https://www.coursera.org/search?query={query.replace(' ','+')}"
        print("Searching:", url)

        driver.get(url)

        WebDriverWait(driver,20).until(
            EC.presence_of_element_located((By.TAG_NAME,"a"))
        )

        scroll_page(driver)

        xpath = """
        //a[contains(@href,'/learn/')
        or contains(@href,'/specializations/')
        or contains(@href,'/professional-certificates/')]
        """

        elements = driver.find_elements(By.XPATH,xpath)

        courses=[]
        seen=set()

        for e in elements:

            link=e.get_attribute("href")
            name=e.text.strip()

            if link and name and link not in seen:

                courses.append((name,link))
                seen.add(link)

            if len(courses) >= 30:
                break

        return courses

    finally:
        driver.quit()


# ------------------------------------------------
# Extract Skills
# ------------------------------------------------

def fetch_skills(soup):

    skills=[]

    for h2 in soup.find_all("h2"):

        if "skills" in h2.get_text(strip=True).lower():

            ul=h2.find_next("ul")

            if ul:
                skills=[a.get_text(strip=True) for a in ul.find_all("a")]

            break

    return skills


# ------------------------------------------------
# Extract Rating
# ------------------------------------------------

def extract_rating(soup):

    tag = soup.find(attrs={"aria-label": re.compile(r"\d\.\d stars")})

    if tag:
        return tag.get("aria-label").split()[0]

    return ""


# ------------------------------------------------
# Scrape Course
# ------------------------------------------------

def scrape_course(course):

    name,link=course

    try:

        r=requests.get(link,headers=HEADERS,timeout=20)

        soup=BeautifulSoup(r.text,"lxml")

        skills=fetch_skills(soup)

        rating=extract_rating(soup)

        return {
            "Name":name,
            "Link":link,
            "Skills":", ".join(skills),
            "Rating":rating
        }

    except:

        return {
            "Name":name,
            "Link":link,
            "Skills":"",
            "Rating":""
        }


# ------------------------------------------------
# Main Scraper Function
# ------------------------------------------------

def scrape_courses(skill):

    print("\n--- SCRAPING COURSES ---")

    courses=get_course_links(skill)

    results=[]

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures=[executor.submit(scrape_course,c) for c in courses]

        for f in as_completed(futures):
            results.append(f.result())

    df=pd.DataFrame(results)

    df.to_csv(OUTPUT_FILE,index=False)

    print("Saved:",OUTPUT_FILE)

    return df