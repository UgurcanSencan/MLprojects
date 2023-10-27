
import pickle

cv = pickle.load(open("models/cv.pkl","rb")) # tokenizer/preprocessing
clf = pickle.load(open("models/clf.pkl","rb")) # model


def make_prediction(email):
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email) # y_predicted
    prediction = 1 if prediction == 1 else -1
    return prediction
