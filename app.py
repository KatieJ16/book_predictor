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
from PIL import Image,  ImageOps, ImageDraw
from bs4 import BeautifulSoup
from io import BytesIO
import base64



method_options = ['Average', 'Neural Network', 'K nearest neighbors', 'All']

st.set_page_config(
    page_title="The Bibliobrain",
    page_icon="logo.png"  # Replace with the actual path to your logo image
)

st.sidebar.success("Select a demo above.")


#initialize the model
num_users = np.load("num_users.npy").item()
num_items = np.load("num_items.npy").item()
latent_dim = 377#100  # Number of latent features


class SparseAutoencoder(nn.Module):
    def __init__(self, num_items, latent_dim):
        super(SparseAutoencoder, self).__init__()
        hidden1 = latent_dim*2
        self.encoder1 = nn.Linear(num_items, hidden1)
        self.encoder2 = nn.Linear(hidden1, latent_dim)
        self.decoder1 = nn.Linear(latent_dim, hidden1)
        self.decoder2 = nn.Linear(hidden1, num_items)
        
    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.decoder1(x))
        x = self.decoder2(x)
        # Scale sigmoid output to [1, 5]
        return 1 + 4 * torch.sigmoid(x)
    
model = SparseAutoencoder(num_items, latent_dim)

# Load your machine learning model (replace "model.pkl" with your actual model file)
# Example: A model trained to predict numerical output based on text input
try:
    model.load_state_dict(torch.load("2model{}.pkl".format(latent_dim)))
except FileNotFoundError:
    st.error("Model file not found! Make sure 'model.pkl' is in the same directory.")

#load info that we will need
# Load the list from the file
with open("titles.pkl", "rb") as file:
    titles = pickle.load(file)
    
with open("top_100.pkl", "rb") as file:
    top_100 = pickle.load(file)
    
with open("suggest.pkl", "rb") as file:
    suggest = pickle.load(file)
    
with open("average_ratings_list.pkl", "rb") as file:
    average_ratings_list = np.array(pickle.load(file))
    
with open("images_dict.pkl", "rb") as file:
    images_dict = pickle.load(file)
    
num_titles = len(titles)

# Load the list from the file
with open("ratings_df.pkl", "rb") as file:
    ratings_df = pickle.load(file)

# App title
# Load the logo image
logo = Image.open("logo.png") 

# Create a container to hold the logo and title
with st.container():
    col1, col2 = st.columns([1, 3])  # Adjust column widths as needed (e.g., [1, 3] for a smaller logo)
    with col1:
        st.image(logo)#, width=50)  # Adjust width as needed

    with col2:
        st.title("The Bibliobrain")#Book Recommendations")
        st.subheader("Use AI to recommend your next great Read!") 

# Input fields
st.write("Your goodreads user id number is the number in your url. Got to your profile and look at the number after the last /. My goodreads url is https://www.goodreads.com/user/show/169695558-katie, so my user id is 169695558.")
user_id = int(st.number_input("What is your User ID for goodreads:", step=1))


num_entries = 250#st.slider("Number of Books to import (the more you have, the better the recommendations, but the longer it will take):", min_value=1, max_value=1000, value=100, step=1)

include_rereads = st.checkbox('Include Rereads?')
method = "Average"#st.selectbox('Choose an method (Choose Average for best results):', method_options)
# Predict button
if st.button("Predict"):
    if user_id:
        try:
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                try:
                    #get user id picture:
                    url = f"https://www.goodreads.com/user/show/{user_id}"
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = requests.get(url, headers=headers)


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
                            st.image(crop_circle(img))
                    else:
                        print("Profile picture not found.")
                except:
                    pass
                
            with col2:
                #get user data
                ratings_data = get_user_data(user_id, num_entries=num_entries)
            st.write("Finding best matches! Comparing to ", num_users, " readers and ", num_items, "books.")

            #make matrix of ratings
            ratings = np.zeros((1, num_titles)) #average_ratings_list.reshape((1, num_titles))#

            for index, row in ratings_data.iterrows():
                if row['Title'] in titles:
                    try:
                        ratings[0, titles.index(row["Title"])] = int(row["Rating"])
                    except:
                        pass

            if method == 'Average':
                sum_ratings = np.zeros(ratings.shape[1])
            if method in ('Neural Network', 'All', 'Average'):
                ratings_torch = torch.tensor(ratings).float()

                #Evaulating the model
                model.eval()
                with torch.no_grad():
                    reconstructed = model(ratings_torch)

                pred_ratings_list = reconstructed[0].detach().numpy()
                #give a list sorted out with books you've already read:
                sorted_indices = np.argsort(pred_ratings_list)[::-1]
                list_num = 1
                if method == 'Average':
                    sum_ratings += pred_ratings_list
                else:
                    st.write("Top books are:")
                    for idx in sorted_indices[:10]: 
                        if include_rereads:
                            if  (np.isnan(pred_ratings_list[idx])) :
                                continue
                            st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
                        else:#don't include rereads
                            if  (ratings[0, idx] > 0) or(np.isnan(pred_ratings_list[idx])) :
                                continue
                            st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
                        list_num += 1
                
            
            if method in ('K nearest neighbors', 'All', 'Average'):
                #load knn 
                with open("knn_model.pkl", "rb") as file:
                    knn = pickle.load(file)

                #get neighbors of new 
                distances, indices = knn.kneighbors(ratings)
                ratings_matrix = ratings_df.values

                pred_ratings_list = np.array([])
                rankings_list = np.array([])
                for item_id in range(num_titles):
                    # Get the ratings for the neighbors on item 2
                    neighbor_ratings = np.array([ratings_matrix[i, item_id] for i in indices[0] if not np.isnan(ratings_matrix[i, item_id])])

                    predicted_rating = np.mean(neighbor_ratings[np.nonzero(neighbor_ratings)])
                    rankings = np.sum(neighbor_ratings[np.nonzero(neighbor_ratings)])
                    
                    pred_ratings_list = np.append(pred_ratings_list, predicted_rating)
                    rankings_list = np.append(rankings_list, rankings)

                best_book_rating = np.max(pred_ratings_list)
                best_book_idx = np.argmax(pred_ratings_list)

                sorted_indices = np.argsort(pred_ratings_list)[::-1]
                list_num = 1 
                if method == 'Average':
                    sum_ratings += np.where(np.isnan(pred_ratings_list), 0, pred_ratings_list)
                else:
                    st.write("Top books are:")
                    for i, idx in enumerate(sorted_indices): 
                        if (ratings[0, idx] > 0) or (np.isnan(pred_ratings_list[idx])):
                            pass
                        else:
                            st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
                            list_num += 1
                        if list_num > 10:
                            break
                            
            #print our results for average
            if method == 'Average':
                list_num = 1
                sum_ratings = sum_ratings/2
                if include_rereads: #get average of their rating and predicted rating
#                     sum_ratings = (sum_ratings + ratings[0]) / 2 #Their own rating get's as much weight as the predicted
                    # Initialize an array to store the results
                    result = sum_ratings.copy()

                    # Loop through the arrays and calculate the average ignoring zero elements
                    for i in range(len(sum_ratings)):
                        if ratings[0][i] != 0 and sum_ratings[i] != 0:
                            result[i] = (ratings[0][i] + sum_ratings[i]) / 2
                        elif ratings[0][i] != 0:
                            result[i] = ratings[0][i]
                        elif sum_ratings[i] != 0:
                            result[i] = sum_ratings[i]
                        else:
                            result[i] = 0
                    sum_ratings = result
                    
                sorted_indices = np.argsort(sum_ratings)[::-1]
                score = np.zeros(sum_ratings.shape)
                for idx, rating in enumerate(sum_ratings):
                    score[idx] += sum_ratings[idx]
                    #add a point for every person that read the book in your group
                    neighbor_ratings = np.array([ratings_matrix[i, idx] for i in indices[0] if not np.isnan(ratings_matrix[i, idx])])
                    score[idx] += min(len(neighbor_ratings[np.nonzero(neighbor_ratings)]>5) *0.05, 1)
                    if include_rereads and (ratings[0, idx] == 5): #if they rated 5, give big boost.
                        score[idx] += 0.1
    
                score += ((average_ratings_list - 5))
                #sort based on score
                sorted_indices = np.argsort(score)[::-1]
                for idx in sorted_indices[:99]: 
                    if include_rereads:
                        if  (np.isnan(pred_ratings_list[idx])) :
                            continue
                        col1, col2 = st.columns([0.2, 0.8])
                        cover_url = images_dict[titles[idx]]
                        with col1:
                            try:
                                st.image(cover_url)
                            except:
                                pass
                        with col2:
                            st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(sum_ratings[idx], 1)))
                            list_num += 1
                    else:#don't include rereads
                        if  (ratings[0, idx] > 0) or(np.isnan(pred_ratings_list[idx])) :
                            continue
                        if suggest[idx]: #exclude later books in series
                            col1, col2 = st.columns([0.2, 0.8])
                            cover_url = images_dict[titles[idx]]
                            with col1:
                                try:
                                    st.image(cover_url)
                                except:
                                    pass
                            with col2:
                                st.write( str(list_num)+'.' , titles[idx], " - Predicted Rating:", str(round(sum_ratings[idx], 1)))#,  ' - Score: ', str(round(score[idx],1)))
                            list_num += 1
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Need your User ID.")


# Footer
st.title("Support Me ☕")

st.write("If you enjoy this app, feel free to buy me a coffee! Your support is much appreciated!")

# Add a PayPal link
coffee_link = "https://www.paypal.com/paypalme/16katiej"
st.markdown(f"[Buy Me a Coffee! ☕]({coffee_link})", unsafe_allow_html=True)
st.write("Made with ❤️ using Streamlit.")

