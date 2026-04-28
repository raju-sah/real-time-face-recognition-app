import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

def train_classifier(data_dir):
    """
    Trains an SVM classifier on the generated embeddings.
    """
    print(f"Loading data from {data_dir}...")
    embeddings = np.load(os.path.join(data_dir, 'embeddings.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    
    with open(os.path.join(data_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)

    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42, stratify=labels)

    # 2. Train SVM Classifier
    # Probability=True is needed for confidence scores in the API
    print("Training SVM classifier...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    # Map numeric labels back to names for the report
    target_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 4. Save the model
    model_path = os.path.join(data_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'processed_dataset')
    
    train_classifier(data_dir)
