import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline


# Sample dataset
data = {
    "email": [
        "Congratulations you won a free prize claim now",
        "Urgent verify your bank account immediately",
        "Click here to receive your reward",
        "Your account is blocked update your password",
        "You have won a lottery claim your money",
        "Free gift waiting for you click now",
        "Confirm your account to avoid suspension",
        "Urgent payment required immediately",
        "Claim your free mobile phone today",
        "Your password expired verify now",
        "Limited offer claim your cash prize",
        "Update your banking details immediately",
        "You are selected for a free reward",
        "Click the link to unlock your account",
        "Win exciting prizes by registering now",
        "Send your details to receive money",
        "Your account needs urgent verification",
        "Congratulations claim your free voucher",
        "Verify your identity to get a reward",
        "You have received a special cash offer",

        "The college meeting is scheduled tomorrow",
        "Please find the attached class timetable",
        "Your assignment submission is due Friday",
        "The project discussion starts at ten",
        "Your examination results are available",
        "The library will remain open today",
        "Please attend the department meeting",
        "Your class notes are attached",
        "The semester examination begins Monday",
        "Thank you for attending the meeting",
        "Your attendance report is ready",
        "The assignment has been received",
        "The college event starts next week",
        "Please check the updated lecture schedule",
        "Your application has been processed",
        "The teacher shared the study material",
        "The meeting has been postponed",
        "Your course registration is complete",
        "The workshop begins at nine",
        "Please submit your project report"
    ],

    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",
        "Phishing", "Phishing", "Phishing", "Phishing",

        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe",
        "Safe", "Safe", "Safe", "Safe", "Safe"
    ]
}


def train_model():
    df = pd.DataFrame(data)

    X = df["email"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25,
        random_state=42, stratify=y
    )

    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n--- MODEL PERFORMANCE ---")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test, predictions,
        labels=["Phishing", "Safe"]
    ))

    return model


def detect_email(model):
    print("\n--- EMAIL DETECTION ---")

    email = input("Enter email text: ")

    result = model.predict([email])[0]

    print("\nPrediction:", result)


def main():
    print("==============================")
    print("  PHISHING EMAIL DETECTION")
    print("==============================")

    model = train_model()

    while True:
        detect_email(model)

        choice = input("\nCheck another email? (yes/no): ")

        if choice.lower() != "yes":
            print("Thank you for using the program.")
            break


if __name__ == "__main__":
    main()