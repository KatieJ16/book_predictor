import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import streamlit as st
import math
from PIL import Image,  ImageOps, ImageDraw


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

def get_goodreads_cover(book_title):
    # Create a search URL with the book title
    search_query = book_title.replace(" ", "+")
    url = f"https://www.goodreads.com/search?q={search_query}"
    
    # Send a GET request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the first search result with the book cover
        book_cover = soup.find("img", class_="bookCover")
        
        if book_cover and "src" in book_cover.attrs:
            return book_cover["src"]  # Return the URL of the book cover image
        else:
            return "No cover image found."
    else:
        return f"Failed to retrieve data. Status code: {response.status_code}"
    
def show_book_pic(book_title, show=False):
    with st.spinner("Searching for the book cover..."):
        title, cover_url = fetch_book_cover(book_title)

    if show:
        if cover_url:
    #         st.success(f"Found: {title}")
            st.image(cover_url, caption=title, width=192)
        else:
            st.error("Could not find a book cover for the given title.")
        
    return cover_url

# Function to crop an image into a circle
def crop_circle(image):
    # Create a mask
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)
    result = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    result.putalpha(mask)
    return result

def display_image_grid(image_list, pred_ratings_list=None, columns=3):
    """
    Displays a grid of images in Streamlit.
    
    Args:
        images (list): List of image objects or file paths.
        columns (int): Number of columns in the grid.
    """
#     st.write(image_list)
#     st.write(pred_ratings_list)
    rows = len(image_list) // columns + int(len(image_list) % columns != 0)
    for row in range(rows):
        cols = st.columns(columns)
        for col_index in range(columns):
            image_index = row * columns + col_index
            if image_index < len(image_list):
                with cols[col_index]:
#                     st.write(image_list[image_index])
                    cover_url = get_goodreads_cover(image_list[image_index].split("\n")[0])
                    try:
                        st.image(cover_url)#, caption=image_list[image_index])#, use_column_width=True)
                    except:
                        pass
                    if pred_ratings_list is not None:
                        st.write(image_list[image_index] + " - Predicted Rating:", str(round(pred_ratings_list[image_index], 1)))