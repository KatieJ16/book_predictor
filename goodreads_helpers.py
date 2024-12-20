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
        time.sleep(random.uniform(0,1))  # Be kind to the server and avoid being blocked
        

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

def get_user_data(user_id, save=False, num_entries=100, file_name="book_club_ratings.csv"):
    
    max_pages = math.ceil(num_entries/20)  # Adjust based on expected data
    if isinstance(user_id, int): #if just one user
        ratings_data = scrape_goodreads_ratings(user_id, max_pages)
    elif isinstance(user_id, list): #if many users
        for user in user_id:
            try:
                new_data = scrape_goodreads_ratings(user, max_pages)
                ratings_data = pd.concat([ratings_data, new_data], ignore_index=True)
            except:
                ratings_data = scrape_goodreads_ratings(user, max_pages)
    else: 
        st.write("Problem, Check that input is comma seperated.")
#     st.write("num entries = ", ratings_data.shape)

    if save:
        if not ratings_data.empty:
            ratings_data.to_csv(file_name, mode='a', header=False, index=False)
            print("Data saved to ", file_name)
        else:
            print("No data retrieved.")
        
    return ratings_data


# Function to fetch book data from Google Books API
def fetch_book_cover(title):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": title, "maxResults": 1}  # Search for the title
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            book_info = data["items"][0]["volumeInfo"]
            title = book_info.get("title", "No Title Available")
            cover_url = book_info.get("imageLinks", {}).get("thumbnail", None)
            return title, cover_url
    return None, None

def show_book_pic(book_title):
    with st.spinner("Searching for the book cover..."):
        title, cover_url = fetch_book_cover(book_title)

    if cover_url:
#         st.success(f"Found: {title}")
        st.image(cover_url, caption=title, width=192)
    else:
        st.error("Could not find a book cover for the given title.")

