from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Step 1: Load the dataset with specified encoding
df = pd.read_csv('stress.csv', encoding='ISO-8859-1')  # Adjust encoding as necessary

# Step 2: Prepare the data
X = df['text']  # Feature: text content
y_stress = df['label']  # Target 1: Stress (1: Stressed, 0: Not Stressed)
y_subreddit = df['subreddit']  # Target 2: Subreddit classification

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train_stress, y_test_stress = train_test_split(
    X, y_stress, test_size=0.2, random_state=42
)
_, _, y_train_sub, y_test_sub = train_test_split(X, y_subreddit, test_size=0.2, random_state=42)

# Step 4: Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression models for both tasks
model_stress = LogisticRegression(max_iter=1000)
model_stress.fit(X_train_tfidf, y_train_stress)

model_subreddit = LogisticRegression(max_iter=1000, multi_class='ovr')
model_subreddit.fit(X_train_tfidf, y_train_sub)

# Step 6: Calculate and print accuracy scores for both models
y_pred_stress = model_stress.predict(X_test_tfidf)
y_pred_sub = model_subreddit.predict(X_test_tfidf)

accuracy_stress = accuracy_score(y_test_stress, y_pred_stress)
accuracy_subreddit = accuracy_score(y_test_sub, y_pred_sub)

print(f"Accuracy for Stress Prediction Model: {accuracy_stress:.2f}")
print(f"Accuracy for Type Prediction Model: {accuracy_stress:.2f}")

# Prediction function
def predict(input_text):
    input_tfidf = vectorizer.transform([input_text])  # Vectorize the input
    prediction_stress = model_stress.predict(input_tfidf)[0]  # Predict stress
    prediction_subreddit = model_subreddit.predict(input_tfidf)[0]  # Predict subreddit
    
    if prediction_stress == 1:
        return "Stressed", prediction_subreddit
    else:
        return "Not Stressed", "Relaxed"

# Route for home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_sentence = request.form["sentence"]
        stress_result, subreddit_result = predict(input_sentence)
        return render_template("index.html", prediction=stress_result, subreddit=subreddit_result)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
