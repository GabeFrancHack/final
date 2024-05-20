from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataProcess:
    def __init__(self, threshold=3000000):
        self.scaler=StandardScaler()
        self.threshold=threshold

    def load_and_Process(self, file_path):
        #put dataset in a panda (list) & check for errors with the data
        try:
            df = pd.read_csv('file_path', delimiter=',')
        except FileNotFoundError:
            print('file not found')
            return None
        except pd.errors.ParserError:
            print('error analyzing')
            return None
        
        #throwing out unwanted data from dataset (elements that probably will not determine 
        #weather the user will like the vide)
        columns_to_drop = ['user_name', 'user_id', 'video_id', 'video_desc', 'video_time', 'video_link']
        df.drop(columns=columns_to_drop, inplace=True)

        df['userliked'] = df['n_likes'].apply(lambda x: 1 if x > self.threshold else 0)

        #sets all values to floats to process data
        df = df.astype(float)

        return df
    
    def fit_scaler(self, X):
        self.scaler.fit(X)

    def transform_features(self, X):
        return self.scaler.transformer(X)