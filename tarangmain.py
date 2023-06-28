import numpy as np
import pandas as pd
import os
import json
# import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten, Dropout, Activation
from tensorflow.keras import regularizers


# Define the base directory where the user-specific folders are located
base_directory = "D:/github/tarang/"

# Get a list of all user-specific folders
user_folders = [folder for folder in os.listdir(base_directory) if folder.startswith("my_spotify_data")]

# Initialize lists to store the data
search_data = []
streaming_data = []
user_data = []

# Iterate over the user folders
for user_folder in user_folders:
    # Construct the path to the JSON files within each user folder
    search_json_path = os.path.join(base_directory, user_folder, "MyData", "SearchQueries.json")
    streaming_json_path = os.path.join(base_directory, user_folder, "MyData", "StreamingHistory.json")
    user_json_path = os.path.join(base_directory, user_folder, "MyData", "UserData.json")

    # Read and process the search history JSON file
    with open(search_json_path, "r") as file:
        search_json_data = json.load(file)
        for item in search_json_data:
            search_data.append({
                "searchTime": item["searchTime"],
                "searchQuery": item["searchQuery"]
            })

    # Read and process the streaming history JSON file
    with open(streaming_json_path, "r") as file:
        streaming_json_data = json.load(file)
        for item in streaming_json_data:
            streaming_data.append({
                "endTime": item["endTime"],
                "artistName": item["artistName"],
                "trackName": item["trackName"],
                "msPlayed": item["msPlayed"]
            })

    # Read and process the user data JSON file
    with open(user_json_path, "r") as file:
        user_json_data = json.load(file)
        user_data.append({
            "username": user_json_data["username"],
            "email": user_json_data["email"],
            "country": user_json_data["country"],
            "birthdate": user_json_data["birthdate"],
            "gender": user_json_data["gender"]
        })

# Convert the search history data to a pandas DataFrame
search_data = pd.DataFrame(search_data)
search_data['searchTime'] = pd.to_datetime(search_data['searchTime'])
search_data['hour_of_day'] = search_data['searchTime'].dt.hour
search_data['day_of_week'] = search_data['searchTime'].dt.dayofweek
search_data['query_length'] = search_data['searchQuery'].apply(lambda x: len(x))

# Convert the streaming history data to a pandas DataFrame
streaming_data = pd.DataFrame(streaming_data)
streaming_data['endTime'] = pd.to_datetime(streaming_data['endTime'])

# Convert the user data to a pandas DataFrame
user_data = pd.DataFrame(user_data)
user_data['birthdate'] = pd.to_datetime(user_data['birthdate'])
user_data['age'] = (pd.to_datetime('today') - user_data['birthdate']).astype('<m8[Y]')
user_data['gender'] = LabelEncoder().fit_transform(user_data['gender'])

# Merge data
merged_data = pd.merge(search_data, streaming_data, how='outer', on='username')
merged_data = pd.merge(merged_data, user_data, how='inner', on='username')

# Encode categorical variables
merged_data['country'] = LabelEncoder().fit_transform(merged_data['country'])

# Normalize numerical features
scaler = MinMaxScaler()
merged_data[['msPlayed']] = scaler.fit_transform(merged_data[['msPlayed']])

# Split data into train and test sets
train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=42)

# Prepare inputs for LSTM architecture
search_inputs = train_data[['hour_of_day', 'day_of_week', 'query_length']]
search_inputs = np.asarray(search_inputs)
search_inputs = np.expand_dims(search_inputs, axis=2)

# Prepare inputs for collaborative filtering
user_inputs = train_data[['age', 'gender', 'country']]
user_inputs = np.asarray(user_inputs)
user_inputs = to_categorical(user_inputs)

# Prepare targets for collaborative filtering
target_inputs = train_data[['msPlayed']]
target_inputs = np.asarray(target_inputs)

# LSTM architecture for search history
lstm_input = Input(shape=(3, 1))
lstm_output = LSTM(64)(lstm_input)
lstm_output = Dense(32, activation='relu')(lstm_output)

# Autoencoder architecture for collaborative filtering
user_input = Input(shape=(user_inputs.shape[1],))
user_embedding = Dense(64, activation='relu')(user_input)

target_input = Input(shape=(target_inputs.shape[1],))
target_embedding = Dense(64, activation='relu')(target_input)

merged_layer = Concatenate()([user_embedding, target_embedding])
encoded = Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5))(merged_layer)
decoded = Dense(64, activation='relu')(encoded)

# Compile the model
autoencoder = Model(inputs=[user_input, target_input], outputs=decoded)
autoencoder.compile(optimizer=Adam(lr=0.001), loss=MeanSquaredError())

# Train the model
autoencoder.fit([user_inputs, target_inputs], target_inputs, epochs=10, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

# Extract embeddings from the trained model
embedding_model = Model(inputs=user_input, outputs=user_embedding)
user_embeddings = embedding_model.predict(user_inputs)

# Generate recommendations based on user preferences
