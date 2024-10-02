import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import gdown

file_id ="1p_izegPi0iS0Sbd2MBRIhxVN3wunUfjW"
url = f'https://drive.google.com/uc?id={file_id}'

def ser_to_numpy(ser):
    return ser.to_numpy().reshape(-1,1)

def overall_player_rating_transform(ser):
    def rating_transform(x):
        return ['Overwhelmingly Positive', 'Very Positive', 'Mostly Positive','Mixed', 'Mostly Negative'].index(x)

    return ser.apply(rating_transform).to_frame()
def target_transform(ser):
    return (ser=="Recommended").astype(np.float16).to_numpy().reshape(-1,1)

# HybMLP architecture
class HybMLP(nn.Module):
    def __init__(self, num_items,num_users, items_embedding_dim,users_embedding_dim,hidden_layers_size=(50,50,50,50),dropout_prob=(0.25,0.25,0.25),output_size=1):
        super(HybMLP,self).__init__()
        self.item_embedding = nn.Embedding(num_embeddings=num_items,embedding_dim=items_embedding_dim)
        self.user_embedding = nn.Embedding(num_embeddings=num_users,embedding_dim=users_embedding_dim)
        self.hidden_layer1 = nn.Linear(in_features=items_embedding_dim+users_embedding_dim+358, out_features=hidden_layers_size[0])
        self.dropout1 = nn.Dropout(dropout_prob[0])
        self.hidden_layer2=nn.Linear(in_features=hidden_layers_size[0],out_features=hidden_layers_size[1])
        self.dropout2 = nn.Dropout(dropout_prob[1])

        self.hidden_layer3=nn.Linear(in_features=hidden_layers_size[1],out_features=hidden_layers_size[2])
        self.dropout3 = nn.Dropout(dropout_prob[2])

        self.hidden_layer4=nn.Linear(in_features=hidden_layers_size[2],out_features=hidden_layers_size[3])
        self.dropout4 = nn.Dropout(dropout_prob[3])

        self.output_layer = nn.Linear(in_features=hidden_layers_size[3],out_features=output_size)

    def forward(self,x1,x2,x3):
        item_vector=self.item_embedding(x1)
        user_vector=self.user_embedding(x2)
        concatenation=torch.cat((item_vector, user_vector, x3), dim=1)

        first_hidden=self.hidden_layer1(concatenation)
        dropout_output1=self.dropout1(first_hidden)
        relu1=F.relu(dropout_output1)

        second_hidden=self.hidden_layer2(relu1)
        dropout_output2=self.dropout2(second_hidden)
        relu2=F.relu(dropout_output2)

        third_hidden=self.hidden_layer3(relu2)
        dropout_output3=self.dropout3(third_hidden)
        relu3=F.relu(dropout_output3)

        forth_hidden=self.hidden_layer4(relu3)
        dropout_output4=self.dropout4(forth_hidden)
        relu4=F.relu(dropout_output4)

        output=self.output_layer(relu4)
        return torch.sigmoid(output)
# initialize model
HybMLP_model=HybMLP(num_items=227,num_users=666536,items_embedding_dim= 20,users_embedding_dim=100,hidden_layers_size=(64,132,16,6),dropout_prob=(0.75,0.75,0.5,0.75))
try:
    HybMLP_model.load_state_dict(torch.load("HybMLP_model.pth"))
except FileNotFoundError:
    gdown.download(f'https://drive.google.com/uc?id={file_id}', 'HybMLP_model.pth', quiet=False)
    HybMLP_model.load_state_dict(torch.load("HybMLP_model.pth"))
HybMLP_model.eval()




def read_data():
    return (pd.read_csv('game_data.csv'),pd.read_csv('user_data.csv'))


game_data,user_data = read_data()
def recommendation_to_user(user_id:int=32):
    game_data,user_data = read_data()
    input_data=game_data.assign(recommendation=0).merge(user_data[user_data.user_id==user_id].assign(recommendation=0),on='recommendation')
    columns_transformer= joblib.load("column_transformer.pkl")


    transformed=columns_transformer.transform(input_data.drop(columns=["game_name","link"]))
    transformed=torch.from_numpy(transformed.toarray()).to(torch.float32)


    x1=transformed[:,0].to(torch.int32)
    x2=transformed[:,1].to(torch.int32)
    x3=transformed[:,2:-1].to(torch.int16)
    with torch.no_grad():
        input_data["recommendation"]=HybMLP_model(x1,x2,x3).reshape(-1,1)


    return input_data.sort_values(by=["recommendation"],ascending=False)[["game_name","link"]].iloc[3:13,:].reset_index()


def check_game_for_recommendation(game_name:str,user_id:int=32):
    game_data,user_data = read_data()
    input_data=game_data[game_data.game_name==game_name].assign(recommendation=0).merge(user_data[user_data.user_id==user_id].assign(recommendation=0),on='recommendation')
    columns_transformer= joblib.load("column_transformer.pkl")
    transformed=columns_transformer.transform(input_data.drop(columns=["game_name","link"]))
    transformed=torch.from_numpy(transformed.toarray()).to(torch.float32)
    x1=transformed[:,0].to(torch.int32)
    x2=transformed[:,1].to(torch.int32)
    x3=transformed[:,2:-1].to(torch.int16)
    with torch.no_grad():
        return HybMLP_model(x1,x2,x3)[0,0]>0.44











with st.sidebar:

    selected = option_menu('Games Recommendation App',

                           ['Games Recommendation For User',
                            'Game To User Recommendation Prediction'],
                           icons=['person','controller'],
                           default_index=0)

if (selected == 'Games Recommendation For User'):
    st.title("Games Recommendation For User")

    user_id=st.number_input("User Id", min_value=0, max_value=666535, value=50)

    if st.button('recommend'):
        games=recommendation_to_user(user_id)
        for idx in range(games.shape[0]):
            game_name=games.loc[[idx],["game_name"]].values[0][0]
            link=games.loc[[idx],["link"]].values[0][0]
            st.markdown(f'[{game_name}]({link})',unsafe_allow_html=True)




if (selected == 'Game To User Recommendation Prediction'):
    st.title("Check Game Recommendation To User")

    user_id=st.number_input("User Id", min_value=0, max_value=666535, value=12585)
    game_name=st.selectbox(
        "Select Game Name",game_data["game_name"].values
    )
    if st.button('Check'):
        if check_game_for_recommendation(game_name,user_id):
            st.success("Recommend!")
        else:
            st.warning("Don't Recommend!")


