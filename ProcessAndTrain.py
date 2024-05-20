import numpy as np
from sklearn.model_selection import train_test_split

def ProcessAndTrain(data_processor, model, file_path):
    df=data_processor.load_and_process_data(file_path)
    if df is None:
        return
    
    X=df[['video_length', 'n_shares', 'n_comments', 'n_likes']].values
    y=df['userliked'].values

    data_processor.fit_scaler(X)
    X_scaled = data_processor.transform_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=.2,random_state=42)

    model.train(X_train, y_train)

    #evaluates model on the 'test' dataset
    loss, accuracy = model.evaluate(X_test, y_test)

    #results of the algorithm
    print('Loss:', loss)
    print('Accuracy:', accuracy)

def predictUser_like(data_processor, model, video_object):
    data = np.array([[video_object['video_length'], video_object['n_shares'], video_object['n_comments'], video_object['n_likes']]])
    data_scaled = data_processor.transform_features(data)
    prediction = model.predict(data_scaled)
    return prediction[0][0]  # Return the scalar value


def addVideoandUser_like(data_processor, file_path, video_object):
    df = data_processor.load_and_process_data(file_path)
    if df is None:
        return
    new_data = {
        'video_length': video_object['video_length'],
        'n_shares': video_object['n_shares'],
        'n_comments': video_object['n_comments'],
        'n_likes': video_object['n_likes'],
        'userliked': video_object['userliked']
    }
    # Append new video data to the DataFrame
    df = df.append(new_data, ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)