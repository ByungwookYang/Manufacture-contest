from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from b_analysis.feature_engineering import preprocess_data

def validate_model(model, processed_train):
    X, y, categorical_cols = preprocess_data(processed_train)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state = 1998)

    predictions = model.predict(X_val)
    f1 = f1_score(y_val, predictions)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)

    return {
        "f1_score": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
