#Video scoring algorithm

#is to be applied to a social media system in a multitude of way 
#(instructions will be provided)


#import all libraries and elements
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

#put dataset in a panda (list) & check for errors with the data
try:
    df = pd.read_csv('tiktok_collected_liked_videos.csv', delimiter=',')
except FileNotFoundError:
    print('file not found')
except pd.errors.ParserError:
    print('error analyzing')

#throwing out unwanted data from dataset (elements that probably will not determine 
#weather the user will like the vide)
columns_to_drop = ['user_name', 'user_id', 'video_id', 'video_desc', 'video_time', 'video_link']
df.drop(columns=columns_to_drop, inplace=True)

#artificially sets wheather the user likes to 0 or 1 depending on the amount of likes the video has
#this does not determine the accuracy in the long run as when the user likes the video it will be 
#added to the dataset
threshold = average_likes = df['n_likes'].mean()
df['userliked'] = df['n_likes'].apply(lambda x: 1 if x > threshold else 0)

#sets all values to floats to process data
df = df.astype(float)

#sets the targets and features
X=df[['video_length', 'n_shares', 'n_comments', 'n_likes']].values
y=df['userliked'].values

#ensures larger values do not have a disproportionally larger effect on the data
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=.2,random_state=42)

#build the actual network
model= Sequential()
model.add(Dense(32, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5)) #for overfitting
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#classifier algorithm (determines if user will like video)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

#train the model
history=model.fit(X_train,y_train,epochs=100,batch_size=10)

#evaluates model on the 'test' dataset
loss, accuracy = model.evaluate(X_test, y_test)

#results of the algorithm
print('Loss:', loss)
print('Accuracy:', accuracy)

data = np.array([[14,852,2860,1300000]])
data_scaled=scaler.transform(data)
prediction = model.predict(data_scaled)

# Print the prediction
print("Estimated userliked value:", prediction)