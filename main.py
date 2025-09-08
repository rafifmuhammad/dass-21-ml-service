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
async def get_data_training():
    return get_all("SELECT * FROM tb_data WHERE Jenis='Training'")

@app.get('/data-testing')
async def get_data_testing():
    return get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

@app.get('/classification-result')
async def get_classification(p1: int, p2: int, p3: int):
    columns = [
        "kd_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "S1", "S2", "S3", "S4", "S5", "S6", "S7", "P1", "P2", "P3", "Kelas", "Jenis"
    ]

    data_training = get_all("SELECT * FROM tb_data WHERE Jenis='Training'")
    data_testing = get_all("SELECT * FROM tb_data WHERE Jenis='Testing'")

    df_train = pd.DataFrame(data_training, columns=columns)
    df_test = pd.DataFrame(data_testing, columns=columns)

    # Lakukan encoding data
    target_columns = ['P1', 'P2', 'P3']
    label_column = 'Kelas'
    le = LabelEncoder()

    df_train = df_train.copy()
    df_test = df_test.copy()

    encoders = {}

    # Encoding P1-P3
    for col in target_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le

    # Encode label 
    le_kelas = LabelEncoder()
    df_train[label_column] = le_kelas.fit_transform(df_train[label_column])
    df_test[label_column] = le_kelas.transform(df_test[label_column])

    # df_all = pd.concat([df_train, df_test])

    # drop_cols = [
    #     "kode_data", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    #     "A1", "A2", "A3", "A4", "A5", "A6", "A7",
    #     "S1", "S2", "S3", "S4", "S5", "S6", "S7", "Jenis", "Kelas"
    # ]

    x_train = df_train[['P1', 'P2', 'P3']]
    y_train = df_train['Kelas']
    x_test = df_test[['P1', 'P2', 'P3']]
    y_test = df_test['Kelas']

    # x_train, x_test, y_train, y_test = train_test_split(df_all[['P1', 'P2', 'P3']], df_all['Kelas'], test_size=0.2, random_state=42, stratify=df_all['Kelas'])

    # One-Hot Encoding untuk fitur kategori (Naive Bayes)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_enc = encoder.fit_transform(x_train)
    x_test_enc = encoder.transform(x_test)

    # Encode target label
    le_target = LabelEncoder()
    y_train_enc = le_target.fit_transform(y_train)
    y_test_enc = le_target.transform(y_test)

    # Terapkan SMOTE
    # smote = SMOTE(random_state=42)
    # x_train_smote, y_train_smote = smote.fit_resample(x_train_enc, y_train_enc)

    # print("Setelah SMOTE:", Counter(y_train_smote))

    # Model Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)

    # Training
    testing_raw = np.array([[p1, p2, p3]])   # 3 fitur
    testing_enc = encoder.transform(testing_raw)  # hasilnya 15 fitur
    nb_model.fit(x_train_enc, y_train)

    y_pred_nb = nb_model.predict(testing_enc)[0]
    prediksi = le_kelas.inverse_transform([y_pred_nb])[0]
    # kelas_asli = y_pred_nb
    kelas_map = {0: "Normal", 1: "Ringan", 2: "Sedang", 3: "Berat", 4: "Sangat Berat"}
    prediksi = kelas_map[y_pred_nb]
    # return prediksi

    # Tambahan cek confidence
    y_proba = nb_model.predict_proba(testing_enc)[0]
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

    # Label Encoding untuk fitur P1-P3
    target_columns = ['P1', 'P2', 'P3']
    label_column = 'Kelas'
    le = LabelEncoder()

    for col in target_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    # Encode label (kelas)
    le_kelas = LabelEncoder()
    df_train[label_column] = le_kelas.fit_transform(df_train[label_column])
    df_test[label_column] = le_kelas.transform(df_test[label_column])

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

    # Ambil hasil per kelas saja (precision, recall, f1)
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

    # Label Encoding untuk fitur P1-P3
    target_columns = ['P1', 'P2', 'P3']
    label_column = 'Kelas'

    for col in target_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    # Encode label (kelas)
    le_kelas = LabelEncoder()
    df_train[label_column] = le_kelas.fit_transform(df_train[label_column])
    df_test[label_column] = le_kelas.transform(df_test[label_column])

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

    # Buat confusion matrix
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

    # Label Encoding untuk fitur P1-P3
    target_columns = ['P1', 'P2', 'P3']
    label_column = 'Kelas'

    for col in target_columns:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    # Encode label (kelas)
    le_kelas = LabelEncoder()
    df_train[label_column] = le_kelas.fit_transform(df_train[label_column])
    df_test[label_column] = le_kelas.transform(df_test[label_column])

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

    # Buat confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = le_kelas.classes_

    # Hitung TP, TN, FP, FN untuk setiap kelas
    result = {}

    for i, class_name in enumerate(class_names):
        # True Positive: diagonal element
        tp = int(cm[i][i])
        
        # False Positive: sum of column i minus TP
        fp = int(np.sum(cm[:, i]) - tp)
        
        # False Negative: sum of row i minus TP
        fn = int(np.sum(cm[i, :]) - tp)
        
        # True Negative: total - tp - fp - fn
        tn = int(np.sum(cm) - tp - fp - fn)
        
        result[class_name] = {
            "tp": tp,
            "tn": tn, 
            "fp": fp,
            "fn": fn
        }

    return result