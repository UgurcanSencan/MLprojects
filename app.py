from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl","rb")) # tokenizer/preprocessing
clf = pickle.load(open("models/clf.pkl","rb")) # model

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('email-content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email) # y_predicted
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)



