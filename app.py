import os
import json
import requests
import logging
import time
import argparse
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import nltk

# Ensure the necessary NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ----- Load Configuration from config.json -----
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found.")

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

NEWS_API_KEY = config.get("news_api_key")
COUNTRY = config.get("country", "us")
CATEGORY = config.get("category", "general")
MAX_HEADLINES_PER_SOURCE = config.get("max_headlines_per_source", 3)
CHROME_DRIVER_PATH = config.get("chrome_driver_path")
REMOTE_DEBUGGER_ADDRESS = config.get("remote_debugger_address", "127.0.0.1:9222")
CONTACTS = config.get("contacts", [])
WHATSAPP_URL = config.get("whatsapp_url", "https://web.whatsapp.com")
LOG_FILE = config.get("logging_file", "news_agent.log")
# Soccer league ID for fetching matches (e.g., 4328 for English Premier League)
SOCCER_LEAGUE_ID = config.get("soccer_league_id")
# --------------------------------------------------

# Setup logging (to file and console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def fetch_news():
    logging.info("Fetching news from NewsAPI")
    url = f"https://newsapi.org/v2/top-headlines?country={COUNTRY}&apiKey={NEWS_API_KEY}"
    if CATEGORY and CATEGORY.lower() != "general":
        url += f"&category={CATEGORY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            logging.info(f"Fetched {len(articles)} articles.")
            return articles
        else:
            logging.error(f"Failed to fetch news. Status code: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Exception during news fetching: {str(e)}")
        return []

def group_articles_by_source(articles, max_per_source=3):
    grouped = {}
    for article in articles:
        source = article.get("source", {}).get("name", "Unknown Source")
        headline = article.get("title", "").strip()
        if not headline:
            continue
        if source not in grouped:
            grouped[source] = []
        if len(grouped[source]) < max_per_source:
            grouped[source].append(headline)
    return grouped

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound = scores.get("compound", 0)
    if compound >= 0.05:
        return "Overall Positive"
    elif compound <= -0.05:
        return "Overall Negative"
    else:
        return "Neutral"

def fetch_inspirational_quote():
    try:
        response = requests.get("https://zenquotes.io/api/random")
        if response.status_code == 200:
            data = response.json()[0]
            quote = data.get("q", "")
            author = data.get("a", "")
            return f"\"{quote}\" - {author}"
        else:
            logging.warning("Could not fetch inspirational quote. Status code: " + str(response.status_code))
            return ""
    except Exception as e:
        logging.error(f"Error fetching inspirational quote: {str(e)}")
        return ""

def fetch_soccer_matches(league_id):
    logging.info(f"Fetching soccer matches for league id: {league_id}")
    url = f"https://www.thesportsdb.com/api/v1/json/1/eventsnextleague.php?id={league_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            events = data.get("events", [])
            logging.info(f"Fetched {len(events)} upcoming soccer matches.")
            return events
        else:
            logging.error(f"Failed to fetch soccer matches. Status code: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Exception during soccer matches fetching: {str(e)}")
        return []

def create_formatted_soccer_matches_message(events):
    
    if not events:
        return "Upcoming Soccer Matches:\n* No upcoming soccer matches."
    lines = ["Upcoming Soccer Matches:"]
    for event in events:
        home = event.get("strHomeTeam", "N/A")
        away = event.get("strAwayTeam", "N/A")
        date = event.get("dateEvent", "")
        time_event = event.get("strTime", "")
        lines.append(f"* {home} vs {away} - {date} {time_event}")
    return "\n".join(lines)

def create_formatted_message_from_grouping(grouped_articles, sentiment, quote, soccer_section=""):
    
    lines = ["Daily News Summary", ""]
    for source, headlines in grouped_articles.items():
        lines.append(f"{source}:")
        for headline in headlines:
            if not headline.endswith('.'):
                headline += '.'
            lines.append(f"* {headline}")
        lines.append("")
    if soccer_section:
        lines.append(soccer_section)
        lines.append("")
    lines.append(f"Overall Sentiment: {sentiment}")
    if quote:
        lines.append(f"Inspirational Quote: {quote}")
    lines.append("")
    lines.append("Have a great day!")
    return "\n".join(lines)

def send_whatsapp_message(contact_name, message):
    logging.info(f"Sending WhatsApp message to {contact_name}.")
    try:
        service = Service(CHROME_DRIVER_PATH)
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", REMOTE_DEBUGGER_ADDRESS)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(WHATSAPP_URL)
        
        # Wait for WhatsApp Web to load
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        logging.info("WhatsApp Web loaded.")
        
        # Locate the search box
        search_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[@role='textbox' and @aria-label='Search input textbox']")
            )
        )
        search_box.click()
        search_box.send_keys(contact_name)
        time.sleep(2)
        
        # Click the contact (ensure exact name match)
        contact = driver.find_element(By.XPATH, f'//span[@title="{contact_name}"]')
        contact.click()
        time.sleep(2)
        
        # Locate the message input box
        message_box = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.XPATH, "//footer//div[@contenteditable='true']"))
        )
        
        # Send the message using SHIFT+ENTER for line breaks so that all lines are in one bubble
        for line in message.split("\n"):
            message_box.send_keys(line)
            message_box.send_keys(Keys.SHIFT, Keys.ENTER)
        # Finally, press ENTER to send the entire message bubble
        message_box.send_keys(Keys.ENTER)
        
        logging.info(f"Message sent to {contact_name}.")
        time.sleep(5)
        driver.quit()
    except Exception as e:
        logging.error(f"Error sending message to {contact_name}: {str(e)}")

def job():
    logging.info("Job started.")
    articles = fetch_news()
    if not articles:
        logging.warning("No articles fetched; skipping message sending.")
        return
    grouped_articles = group_articles_by_source(articles, max_per_source=MAX_HEADLINES_PER_SOURCE)
    
    # Flatten headlines for sentiment analysis
    flattened_text = " ".join([" ".join(headlines) for headlines in grouped_articles.values()])
    sentiment = analyze_sentiment(flattened_text)
    
    # Fetch an inspirational quote
    quote = fetch_inspirational_quote()
    
    formatted_message = create_formatted_message_from_grouping(grouped_articles, sentiment, quote)
    logging.info("Formatted message:\n" + formatted_message)
    
    for contact in CONTACTS:
        send_whatsapp_message(contact, formatted_message)
    logging.info("Job completed.")

def main():
    # Run the job once
    job()

if __name__ == "__main__":
    main()
    
