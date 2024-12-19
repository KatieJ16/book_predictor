import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import streamlit as st
import math

def scrape_goodreads_ratings(user_id, max_pages=10):
    """
    Scrape a user's star ratings from Goodreads.
    
    Args:
    - user_id (str): Goodreads user ID or profile suffix.
    - max_pages (int): Maximum number of pages to scrape (each page contains ~30 books).
    
    Returns:
    - pd.DataFrame: A DataFrame containing book titles and ratings.
    """
    base_url = f"https://www.goodreads.com/review/list/{user_id}?shelf=read"
    headers = {"User-Agent": "Mozilla/5.0"}
    books = []

    st.write("Getting user data:")
    # Create a progress bar
    progress_bar = st.progress(0)
    
    for page in range(1, max_pages + 1):
        
        url = f"{base_url}&page={page}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all book entries in the table
        rows = soup.find_all("tr", class_="bookalike review")
        if not rows:
            print("No more data found.")
            break

        for row in rows:
            try:
                title = row.find("td", class_="field title").a.text.strip()
                rating_element = row.find("td", class_="field rating")
                rating = rating_element.find("span", class_="staticStars").get("title", "No rating")
                stars = map_rating(rating)
                books.append({"Title": title, "Rating": stars, "User_id": user_id})
#                 print(title, rating, stars)
            except AttributeError:
                # Handle rows with missing data
                continue

        print(f"Page {page} scraped successfully.")
        progress_bar.progress(page/(max_pages + 1))
        time.sleep(random.uniform(1, 2))  # Be kind to the server and avoid being blocked
        

    progress_bar.progress(100)
    # Return data as a pandas DataFrame
    return pd.DataFrame(books)

def map_rating(phrase):
    rating_map = {
        "liked it": 3,
        "really liked it": 4,
        "it was ok": 2, 
        "it was amazing": 5, 
        "did not like it": 1,
    }
    
    return rating_map.get(phrase, "Invalid rating")  # Default to "Invalid rating" if the phrase isn't in the dictionary

def get_user_data(user_id, save=False, num_entries=100):
    
    max_pages = math.ceil(num_entries/20)  # Adjust based on expected data
    ratings_data = scrape_goodreads_ratings(user_id, max_pages)
#     st.write("num entries = ", ratings_data.shape)

    if save:
        if not ratings_data.empty:
            ratings_data.to_csv('goodreads_ratings.csv', mode='a', header=False, index=False)
            print("Data saved to goodreads_ratings.csv.")
        else:
            print("No data retrieved.")
        
    return ratings_data
