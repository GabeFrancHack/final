from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

class VideoLikePredict:
    def __init__(self, input_shape):
        #build the actual network
        model= Sequential()
        model.add(Dense(32, input_shape=(input_shape,), activation='relu'))
        model.add(Dropout(0.5)) #for overfitting
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        #classifier algorithm (determines if user will like video)
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=100, batch_size=10):
        self.model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
    
    def predict(self, X):
        return self.model.predict(X)