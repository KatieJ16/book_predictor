import streamlit as st
import pickle

#make model 

#doing masked autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from goodreads_helpers import *
from sklearn.neighbors import NearestNeighbors
import sys
import os
from PIL import Image,  ImageOps, ImageDraw
from bs4 import BeautifulSoup
from io import BytesIO
import base64


# Add the parent directory to sys.path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_directory)

st.set_page_config(
    page_title="The Bibliobrain",
    page_icon="logo.png"  # Replace with the actual path to your logo image
)


# method_options = ['Average', 'Neural Network', 'K nearest neighbors', 'All']

if "data" not in st.session_state:
    st.session_state["data"] = None
    

    
#initialize the model
num_users = np.load("num_users.npy").item()
num_items = np.load("num_items.npy").item()
latent_dim = 377#100  # Number of latent features


#load info that we will need
# Load the list from the file
with open("titles.pkl", "rb") as file:
    titles = pickle.load(file)
    
with open("top_100.pkl", "rb") as file:
    top_100 = pickle.load(file)
    
with open("images_dict.pkl", "rb") as file:
    images_dict = pickle.load(file)
    
num_titles = len(titles)

# Load the list from the file
with open("ratings_df.pkl", "rb") as file:
    ratings_df = pickle.load(file)
    ratings_matrix = ratings_df.values

# App title
from PIL import Image

# Load the logo image
logo = Image.open("logo.png") 

# Create a container to hold the logo and title
with st.container():
    col1, col2 = st.columns([1, 3])  # Adjust column widths as needed (e.g., [1, 3] for a smaller logo)
    with col1:
        st.image(logo)#, width=50)  # Adjust width as needed

    with col2:
#         st.title("Your App Title") 
        st.title("Find Similar Books")#Book Recommendations")
        st.subheader("Use AI to recommend your next great Book!") 
# st.image("./logo.png")

st.write("Here we will find users similar to you. Then pick a book you like, and we will find books most similar to that book based on similar readers.")
# Input fields
st.write("Your goodreads user id number is the number in your url. Got to your profile and look at the number after the last /. My goodreads url is https://www.goodreads.com/user/show/169695558-katie, so my user id is 169695558.")
user_id = int(st.number_input("What is your User ID for goodreads:", step=1))

num_entries = 100#int(st.number_input("Number of Latest book reviews to consider (the more you have the better recommendations you'll get but the longer it will take):", step=1, value = 100))

book = 'The Great Gatsby'
if st.button("Get Your User Data"):
    #get user data
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        try:
            #get user id picture:
            url = f"https://www.goodreads.com/user/show/{user_id}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)

#                 st.write(response.status_code)

            # Check if the request was successful
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract the profile picture (typically within an <img> tag)
                profile_picture_tag = soup.find('img', class_='og:image')
                og_image_tag = soup.find('meta', property='og:image')
                profile_picture_url = og_image_tag['content']
                if og_image_tag:
                    img_response = requests.get(profile_picture_url)
                    if img_response.status_code == 200:
                        img = Image.open(BytesIO(img_response.content))

                    # Display the profile picture
                    st.image(crop_circle(img))#, caption="Goodreads Profile Picture")
            else:
#                 else:
                    print("Profile picture not found.")
        except:
            pass

    with col2:
        st.write("Finding best matched Readers! Comparing to ", num_users, " readers and ", num_items, "books.")
        ratings_data = get_user_data(user_id, num_entries=num_entries)
    

    #make matrix of ratings
    # ratings = np.full((num_users, num_titles), None)
    ratings = np.zeros((1, num_titles))
    try:
        titles_list = list(set(list(ratings_data["Title"])).intersection(titles))
    except:
        st.error("No user data imported. Please check your User ID number.")
   

    for index, row in ratings_data.iterrows():
        if row['Title'] in titles:
            try:
                ratings[0, titles.index(row["Title"])] = int(row["Rating"])
    #             print("found ", row["Title"])
            except:
                pass
            
    st.session_state["data"] = {"ratings": ratings, "titles_list": titles_list}

    # include_rereads = st.checkbox('Include Rereads?')
    # method = st.selectbox('Choose an method:', method_options)
if st.session_state["data"]:
    full_titles_list = st.session_state["data"]["titles_list"]
#     print("ratings = ", ratings)

    #load the model
    if st.session_state["data"]:
        ratings = st.session_state["data"]["ratings"]
        print("ratings = ", ratings)
    else:
        st.warning("No data available. Need to User data first.")
    #load knn 
    with open("knn_model_30.pkl", "rb") as file:
        knn = pickle.load(file)

    distances, indices = knn.kneighbors(ratings)
#     close_ratings = np.column_stack((ratings_matrix[indices[0]].T, ratings.T))
    close_ratings = ratings_matrix[indices[0]].T
    average_ratings = np.nanmean(np.where(close_ratings == 0, np.nan, close_ratings), axis=1)#np.mean(close_ratings, axis=1)
    nonzeros_per_column = np.count_nonzero(close_ratings, axis=1)
#     st.write("Number of non-zero elements in each column:", nonzeros_per_column)
    
#     st.write("titles_list = ", len(full_titles_list))
#     st.write("nonzeros_per_column = ", len(nonzeros_per_column))
#     st.write("close_ratings = ", close_ratings.shape)
#     titles_list = [full_titles_list[i] for i in range(len(full_titles_list)) if nonzeros_per_column[i] > 3]
    titles_list = [titles[i] for i in range(len(titles)) if nonzeros_per_column[i] > 3]

#     st.write("titles_list = ", len(titles_list))
    titles_list = list(set(full_titles_list) & set(titles_list))
#     st.write("titles_list = ", len(titles_list))
    book = st.selectbox('Pick a book to find similar books: ', titles_list)
    
    
# Predict button
if st.button("Recommend Books!"):
    if user_id:
        try:
            
            
            #get the ratings of nearby nieghbors
#             st.write("ratings_matrix[indices[0]] shape = ", ratings_matrix[indices[0]].shape)
#             st.write("ratings shape = ", ratings.shape)
            close_ratings = np.column_stack((ratings_matrix[indices[0]].T, ratings.T))
#             st.write("close_ratings shape = ", close_ratings.shape)
#             close_ratings = ratings_matrix[indices[0]].T
            
            #find books that are like selected book
            #find nearby books
            knn = NearestNeighbors(n_neighbors=100, metric='cosine')  # Using cosine similarity #math.ceil(num_users/10)
            knn.fit(close_ratings)
            
#             title_this = "Harry Potter and the Goblet of Fire\n        (Harry Potter, #4)"
            item_id = titles.index(book)
            average_ratings = np.nanmean(np.where(close_ratings == 0, np.nan, close_ratings), axis=1)#np.mean(close_ratings, axis=1)
            nonzeros_per_column = np.count_nonzero(close_ratings, axis=1)
#             st.write("Number of non-zero elements in each column:", nonzeros_per_column)
            st.write("Finding books similar to: ", titles[item_id], " - Group Rating: ", str(round(average_ratings[item_id], 1)))#, "Number rated", str(nonzeros_per_column[item_id]))
            # item_id = np.nanargmax(average_ratings) # Index of item to predict rating for
            # print("highest rated book = ", titles[item_id], "ratings = ", average_ratings[item_id])

            # Get the nearest neighbors for user 0 (excluding the user itself)
            distances, indices = knn.kneighbors([close_ratings[item_id]])

#             print(distances, indices)

            st.write("Most similar books:")
            for i, idx in enumerate(indices[0]):
                if idx == item_id:
                    pass
                else:
                    if nonzeros_per_column[idx] > 3:
                        col1, col2 = st.columns([0.2, 0.8])
                        cover_url = images_dict[titles[idx]]
                        with col1:
                            try:
                                st.image(cover_url)#, caption=image_list[image_index])#, use_column_width=True)
                            except:
                                pass
                        with col2:
                            st.write(  titles[idx], " - Predicted Rating:", str(round(average_ratings[idx], 1)))
#                         st.write(str(i+1), ": ", titles[idx], "- Predicted Rating: ", str(round(average_ratings[idx], 1)))#, "Number rated", str(nonzeros_per_column[idx]))


            #########
            #get neighbors of new 
#             distances, indices = knn.kneighbors(ratings)
#             ratings_matrix = ratings_df.values

#             pred_ratings_list = np.array([])
#             rankings_list = np.array([])
#             for item_id in range(num_titles):
#                 # Get the ratings for the neighbors on item 2
#                 neighbor_ratings = np.array([ratings_matrix[i, item_id] for i in indices[0] if not np.isnan(ratings_matrix[i, item_id])])

#                 predicted_rating = np.mean(neighbor_ratings[np.nonzero(neighbor_ratings)])
#                 rankings = np.sum(neighbor_ratings[np.nonzero(neighbor_ratings)])

#                 pred_ratings_list = np.append(pred_ratings_list, predicted_rating)
#                 rankings_list = np.append(rankings_list, rankings)

#             best_book_rating = np.max(pred_ratings_list)
#             best_book_idx = np.argmax(pred_ratings_list)

#             sorted_indices = np.argsort(pred_ratings_list)[::-1]
#             st.write("Top books are:")
#             list_num = 1 
#             for i, idx in enumerate(sorted_indices): 
#                 if (ratings[0, idx] > 0) or (np.isnan(pred_ratings_list[idx])):
#                     pass
#                 else:
#                     st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
#                     list_num += 1
#                 if list_num > 10:
#                     break
                            
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Need your User ID.")

# selected_option = st.selectbox(
#     'Choose a book:',
#     top_100
# )

# # Display the selected option
# st.write(f"You selected: {selected_option}")



# Footer
st.title("Support Me ☕")

st.write("If you enjoy this app, feel free to buy me a coffee! Your support is much appreciated!")

# Add a PayPal link
coffee_link = "https://www.paypal.com/paypalme/16katiej"
st.markdown(f"[Buy Me a Coffee! ☕]({coffee_link})", unsafe_allow_html=True)
st.write("Made with ❤️ using Streamlit.")

