import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Load training data
data = pd.read_csv("train.csv")  # Contains 'text' and 'label' columns

# 2. Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# 3. Train the model
model = MultinomialNB()
model.fit(X, y)

# 4. Test the model on user input
print("Sentiment Analyzer is ready. Type 'exit' to quit.")
while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() == "exit":
        break
    input_vec = vectorizer.transform([sentence])
    prediction = model.predict(input_vec)
    print("Predicted sentiment:", prediction[0])

