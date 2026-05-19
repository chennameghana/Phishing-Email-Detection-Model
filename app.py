import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("phishing_email.csv")

# Show column names
print(data.columns)

# Show first 5 rows
print(data.head())

# CHANGE COLUMN NAMES AFTER SEEING OUTPUT
X = data['URL']
y = data['Label']

# Convert text into numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Test sample URL
sample_url = ["http://free-login-bank-security.com"]

sample_vector = vectorizer.transform(sample_url)
prediction = model.predict(sample_vector)

print("\nPrediction for sample URL:", prediction[0])