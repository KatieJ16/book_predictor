import streamlit as st
import pickle

#make model 

#doing masked autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from goodreads_helpers import *

# # Mask for observed values (1 for observed, 0 for missing)
# ratings_torch = torch.tensor(ratings).float()
# mask = (ratings_torch != 0).float()
# print(mask)



#Define autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, num_items, latent_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(num_items, latent_dim)
        self.decoder = nn.Linear(latent_dim, num_items)
        
    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        # Scale sigmoid output to [1, 5]
        return 1 + 4 * torch.sigmoid(decoded)
        return decoded

    
#initialize the model
num_users = np.load("num_users.npy").item()
num_items = np.load("num_items.npy").item()
latent_dim = 100  # Number of latent features

model = SparseAutoencoder(num_items, latent_dim)

# Load your machine learning model (replace "model.pkl" with your actual model file)
# Example: A model trained to predict numerical output based on text input
try:
    model.load_state_dict(torch.load("model.pkl"))
except FileNotFoundError:
    st.error("Model file not found! Make sure 'model.pkl' is in the same directory.")

#load info that we will need
# Load the list from the file
with open("titles.pkl", "rb") as file:
    titles = pickle.load(file)
    
with open("top_100.pkl", "rb") as file:
    top_100 = pickle.load(file)
    
num_titles = len(titles)

# Load the list from the file
with open("ratings_df.pkl", "rb") as file:
    ratings_df = pickle.load(file)

# App title
st.title("Book Recommendations")

# Input fields
st.write("Your goodreads user id number is the number in your url. Got to your profile and look at the number after the last /. My goodreads url is https://www.goodreads.com/user/show/169695558-katie, so my user id is 169695558.")
user_id = int(st.number_input("What is your User ID for goodreads:", step=1))

num_entries = int(st.number_input("Number of Latest book reviews to consider (the more you have the better recommendations you'll get but the longer it will take):", step=1, value = 100))

# st.write("user_id = ", user_id)
include_rereads = st.checkbox('Include Rereads?')
# Predict button
if st.button("Predict"):
    if user_id:
        try:
            #get user data
            ratings_data = get_user_data(user_id, num_entries=num_entries)
#             st.write("ratings_data = ", ratings_data)
            
            #make matrix of ratings
            # ratings = np.full((num_users, num_titles), None)
            ratings = np.zeros((1, num_titles))

            for index, row in ratings_data.iterrows():
                if row['Title'] in titles:
                    try:
                        ratings[0, titles.index(row["Title"])] = int(row["Rating"])
            #             print("found ", row["Title"])
                    except:
                        pass

#             st.write("ratings = ", ratings)
            ratings_torch = torch.tensor(ratings).float()
            
            #Evaulating the model
            model.eval()
            with torch.no_grad():
                reconstructed = model(ratings_torch)
                
            st.write("Finding best matches! Comparing to ", num_users, " readers and ", num_items, "books.")
            pred_ratings_list = reconstructed[0].detach().numpy()
#             st.write(pred_ratings_list)
            #give a list sorted out with books you've already read:
            sorted_indices = np.argsort(pred_ratings_list)[::-1]
            st.write("Top books are:")
            list_num = 1
            for idx in sorted_indices[:100]: 
                #     print("ratings_matrix[user_id, idx]= ", ratings_matrix[user_id, idx])
                if include_rereads:
                    if  (np.isnan(pred_ratings_list[idx])) :
                        continue
                    st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
                    list_num += 1
                else:#don't include rereads
                    if  (ratings[0, idx] > 0) or(np.isnan(pred_ratings_list[idx])) :
                        continue
                    st.write( str(list_num) , titles[idx], " - Predicted Rating:", str(round(pred_ratings_list[idx], 1)))
                    list_num += 1
#             st.success(f"Prediction: {prediction[0]}")  # Display the prediction
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

