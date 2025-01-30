import pickle
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        report[model_name] = score
    return report