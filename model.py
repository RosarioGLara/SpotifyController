import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# load your datap
df = pd.read_csv("dataset/dataset.csv", header=None)

# separate features and labels
X = df.iloc[:, :-1]   # first 126 columns
y = df.iloc[:, -1]    # last column (labels)

print(X.shape)  # should be (samples, 126)
print(y.unique())  # see your gestures

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(list(encoder.classes_))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
scaler = StandardScaler()

## TODO: Model selection

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)
label = encoder.inverse_transform(prediction)

print("Predicted gesture:", label[0])
y_pred = model.predict(X_test)

for i in range(10):  # show first 10 predictions
    actual = encoder.inverse_transform([y_test[i]])[0]
    predicted = encoder.inverse_transform([y_pred[i]])[0]
    
    print(f"Actual: {actual} | Predicted: {predicted}")

joblib.dump(model, "models/gesture_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/encoder.pkl")
