# API Library
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import mysql.connector

# Machine Learning Library
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

app = FastAPI()

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="db_kesehatan_mental"
    )

def get_all(query, params=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params or ())

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return results

def execute_query(query, params=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params or ())

    conn.commit()
    cursor.close()
    conn.close()

@app.get('/')
def root():
    return {"message": "Halo, ini adalah pesan halaman root!"}

@app.get('/data-training')
def get_data_training():
    return get_all("SELECT * FROM tb_data WHERE Jenis='Training'")

@app.get('/data-testing')
def get_data_testing():
    return get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

@app.get('/classification-result')
def get_classification(p1: str, p2: str, p3: str):
    columns = [
        "kd_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "P1", "P2", "P3", "Kelas", "Jenis"
    ]

    data_training = get_all("SELECT * FROM tb_data WHERE Jenis='Training'")
    data_testing  = get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

    df_train = pd.DataFrame(data_training, columns=columns)
    df_test  = pd.DataFrame(data_testing, columns=columns)

    for col in ['P1', 'P2', 'P3']:
        df_train[col] = df_train[col].astype(str).str.lower()
        df_test[col]  = df_test[col].astype(str).str.lower()

    le_kelas = LabelEncoder()
    df_train["Kelas"] = le_kelas.fit_transform(df_train["Kelas"])
    df_test["Kelas"]  = le_kelas.transform(df_test["Kelas"])

    x_train = df_train[['P1', 'P2', 'P3']]
    y_train = df_train['Kelas']
    x_test = df_test[['P1', 'P2', 'P3']]
    y_test = df_test['Kelas']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_enc = encoder.fit_transform(x_train)
    x_test_enc = encoder.transform(x_test)

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(x_train_enc, y_train)

    testing_raw = pd.DataFrame([[p1, p2, p3]], columns=['P1', 'P2', 'P3'])
    for col in testing_raw.columns:
        testing_raw[col] = testing_raw[col].astype(str).str.lower()
    testing_enc = encoder.transform(testing_raw)

    y_pred_nb = nb_model.predict(testing_enc)[0]

    kelas_map = {i: label for i, label in enumerate(le_kelas.classes_)}
    prediksi = kelas_map[y_pred_nb]

    y_proba = nb_model.predict_proba(testing_enc)[0]
    confidence = float(max(y_proba))

    y_pred_test = nb_model.predict(x_test_enc)
    akurasi = accuracy_score(y_test, y_pred_test)

    return {
        "P1": p1,
        "P2": p2,
        "P3": p3,
        "confidence": confidence,
        "akurasi": akurasi,
        "result": prediksi
    }

@app.api_route("/split-data", methods=["GET", "POST"])
def split_data():
    conn = get_connection()

    # Ambil data tertentu
    df = pd.read_sql("SELECT kd_data, P1, P2, P3, Kelas, Jenis FROM tb_data", conn)

    X = df[["P1", "P2", "P3"]]
    y = df["Kelas"]

    # Split data 80% train, 20% test)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ambil kd_data
    train_ids = df.loc[X_train.index, "kd_data"].tolist()
    test_ids = df.loc[X_test.index, "kd_data"].tolist()

    # Update db
    cursor = conn.cursor()
    cursor.execute("UPDATE tb_data SET Jenis='belum'")
    conn.commit()

    for kd in train_ids:
        cursor.execute("UPDATE tb_data SET Jenis='Training' WHERE kd_data=%s", (kd,))
    for kd in test_ids:
        cursor.execute("UPDATE tb_data SET Jenis='Testing' WHERE kd_data=%s", (kd,))
    conn.commit()

    conn.close()

    return {
        "message": "Split data sukses",
        "train_count": len(train_ids),
        "test_count": len(test_ids)
    }

@app.api_route("/model-evaluation")
def model_evaluation():
    columns = [
        "kd_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "P1", "P2", "P3", "Kelas", "Jenis"
    ]

    # Ambil data training & testing
    data_training = get_all("SELECT * FROM tb_data WHERE Jenis='Training'")
    data_testing = get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

    df_train = pd.DataFrame(data_training, columns=columns)
    df_test = pd.DataFrame(data_testing, columns=columns)

    # Hanya encode target (Kelas)
    le_kelas = LabelEncoder()
    df_train["Kelas"] = le_kelas.fit_transform(df_train["Kelas"])
    df_test["Kelas"] = le_kelas.transform(df_test["Kelas"])

    # Pisahkan fitur & label
    x_train = df_train[["P1", "P2", "P3"]]
    y_train = df_train["Kelas"]
    x_test = df_test[["P1", "P2", "P3"]]
    y_test = df_test["Kelas"]

    # One-Hot Encoding (langsung ke fitur)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_enc = encoder.fit_transform(x_train)
    x_test_enc = encoder.transform(x_test)

    # Model Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(x_train_enc, y_train)

    # Prediksi
    y_pred = nb_model.predict(x_test_enc)

    # Evaluasi keseluruhan
    acc = accuracy_score(y_test, y_pred)

    # Evaluasi per kelas
    report = classification_report(
        y_test,
        y_pred,
        target_names=le_kelas.classes_,
        output_dict=True,
        zero_division=0
    )

    # Ambil hasil per kelas (precision, recall, f1)
    per_class_metrics = {
        kelas: {
            "precision": round(report[kelas]["precision"], 4),
            "recall": round(report[kelas]["recall"], 4),
            "f1_score": round(report[kelas]["f1-score"], 4)
        }
        for kelas in le_kelas.classes_
    }

    return {
        "accuracy": round(acc, 4),
        "per_class": per_class_metrics
    }

@app.api_route('/actual-predicted')
def get_confusion_matrix_actual_vs_predicted():
    columns = [
        "kd_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "P1", "P2", "P3", "Kelas", "Jenis"
    ]

    # Ambil data training & testing
    data_training = get_all("SELECT * FROM tb_data WHERE Jenis='Training'")
    data_testing = get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

    df_train = pd.DataFrame(data_training, columns=columns)
    df_test = pd.DataFrame(data_testing, columns=columns)

    # Encode label (kelas)
    le_kelas = LabelEncoder()
    df_train['Kelas'] = le_kelas.fit_transform(df_train['Kelas'])
    df_test['Kelas'] = le_kelas.transform(df_test['Kelas'])

    # Pisahkan fitur dan label
    x_train = df_train[['P1', 'P2', 'P3']]
    y_train = df_train['Kelas']
    x_test = df_test[['P1', 'P2', 'P3']]
    y_test = df_test['Kelas']

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_enc = encoder.fit_transform(x_train)
    x_test_enc = encoder.transform(x_test)

    # Model Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(x_train_enc, y_train)

    # Prediksi
    y_pred = nb_model.predict(x_test_enc)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = le_kelas.classes_

    # Format hasil: Aktual -> {Prediksi: count}
    result = {}
    for i, actual_class in enumerate(class_names):
        result[actual_class] = {}
        for j, predicted_class in enumerate(class_names):
            result[actual_class][predicted_class] = int(cm[i][j])

    return result

@app.api_route('/confusion-matrix-metrics')
def get_confusion_matrix_metrics():
    columns = [
        "kd_data","D1","D2","D3","D4","D5","D6","D7",
        "A1","A2","A3","A4","A5","A6","A7",
        "S1","S2","S3","S4","S5","S6","S7",
        "P1","P2","P3","Kelas","Jenis"
    ]

    data_training = get_all("SELECT * FROM tb_data WHERE Jenis='Training'")
    data_testing  = get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")
    df_train = pd.DataFrame(data_training, columns=columns)
    df_test  = pd.DataFrame(data_testing, columns=columns)

    # Pastikan fitur kategorikal bersih
    df_train[['P1','P2','P3']] = df_train[['P1','P2','P3']].fillna("Unknown").astype(str)
    df_test[['P1','P2','P3']]  = df_test[['P1','P2','P3']].fillna("Unknown").astype(str)

    # Encode label target
    le_kelas = LabelEncoder()
    df_train['Kelas'] = le_kelas.fit_transform(df_train['Kelas'])
    df_test['Kelas']  = le_kelas.transform(df_test['Kelas'])

    # Split fitur dan label
    x_train = df_train[['P1','P2','P3']]
    y_train = df_train['Kelas']
    x_test  = df_test[['P1','P2','P3']]
    y_test  = df_test['Kelas']

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_enc = encoder.fit_transform(x_train)
    x_test_enc  = encoder.transform(x_test)

    # Model
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(x_train_enc, y_train)
    y_pred = nb_model.predict(x_test_enc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = le_kelas.classes_

    result = {}
    for i, class_name in enumerate(class_names):
        tp = int(cm[i][i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        tn = int(np.sum(cm) - tp - fp - fn)
        result[class_name] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    return result

@app.api_route('/model-experiment')
def get_model_experiment(split_percentage: float = 0.2):
    columns = [
        "kd_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "P1", "P2", "P3", "Kelas", "Jenis"
    ]

    data_all = get_all("SELECT * FROM tb_data")
    df = pd.DataFrame(data_all, columns=columns)

    df[['P1', 'P2', 'P3']] = df[['P1', 'P2', 'P3']].astype(str)
    df['Kelas'] = df['Kelas'].astype(str)

    # Label encoding target
    label_encoder_kelas = LabelEncoder()
    df['Kelas'] = label_encoder_kelas.fit_transform(df['Kelas'])

    # Split data dinamis
    x_train, x_test, y_train, y_test = train_test_split(
        df[['P1', 'P2', 'P3']],
        df['Kelas'],
        test_size=split_percentage,
        random_state=42,
        stratify=df['Kelas']
    )

    # One-hot encoding parameter kategorial
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(df[['P1', 'P2', 'P3']])
    x_train_enc = onehot_encoder.fit_transform(x_train)
    x_test_enc = onehot_encoder.transform(x_test)

    # Model K-NN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train_enc, y_train)
    y_pred_knn = knn_model.predict(x_test_enc)

    # Model Decision Tree
    tree_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    tree_model.fit(x_train_enc, y_train)
    y_pred_tree = tree_model.predict(x_test_enc)

    # Model Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(x_train_enc, y_train)
    y_pred_nb = nb_model.predict(x_test_enc)

    # Model SVM
    svm_model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svm_model.fit(x_train_enc, y_train)
    y_pred_svm = svm_model.predict(x_test_enc)

    # Fungsi untuk menghitung metrik
    def get_value(metric, y_pred):
        if metric == 'acc':
            return round(accuracy_score(y_test, y_pred), 4)
        elif metric == 'f1-score':
            return round(f1_score(y_test, y_pred, average='weighted'), 4)
        elif metric == 'precision':
            return round(precision_score(y_test, y_pred, average='weighted'), 4)
        elif metric == 'recall':
            return round(recall_score(y_test, y_pred, average='weighted'), 4)

    # Hasil akhir per model
    results = {
        "knn": {
            "akurasi": get_value('acc', y_pred_knn),
            "f1_score": get_value('f1-score', y_pred_knn),
            "precision": get_value('precision', y_pred_knn),
            "recall": get_value('recall', y_pred_knn)
        },
        "decision_tree": {
            "akurasi": get_value('acc', y_pred_tree),
            "f1_score": get_value('f1-score', y_pred_tree),
            "precision": get_value('precision', y_pred_tree),
            "recall": get_value('recall', y_pred_tree)
        },
        "naive_bayes": {
            "akurasi": get_value('acc', y_pred_nb),
            "f1_score": get_value('f1-score', y_pred_nb),
            "precision": get_value('precision', y_pred_nb),
            "recall": get_value('recall', y_pred_nb)
        },
        "svm": {
            "akurasi": get_value('acc', y_pred_svm),
            "f1_score": get_value('f1-score', y_pred_svm),
            "precision": get_value('precision', y_pred_svm),
            "recall": get_value('recall', y_pred_svm)
        }
    }

    return results