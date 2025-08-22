import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import numpy as np
from glob import glob
from xgboost import XGBClassifier
from sklearn.utils import resample
from tqdm import tqdm


def salvar_resultados_em_txt(accuracy, precision, recall, f1, pasta="results"):
    os.makedirs(pasta, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    caminho_arquivo = os.path.join(pasta, f"resultado_{timestamp}.txt")

    with open(caminho_arquivo, 'w') as f:
        f.write(f"Acurácia: {accuracy:.4f}\n")
        f.write(f"Precisão: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    print(f"Resultados salvos em: {caminho_arquivo}")

def extrair_estatisticas_por_pacote(df_total):
    devicegroup = df_total.groupby(['label'])
    all_devices_samples = []

    for name, amostra in tqdm(devicegroup, unit="label"):
        if len(amostra.index) > 2:
            amostras_tamanho_x = np.array_split(amostra, len(amostra) / 4)
            for device in amostras_tamanho_x:
                device = pd.DataFrame(device)
                device['iat'] = device['timestamp'].diff()
                device['iat'].fillna(device['iat'].mean(), inplace=True)

                all_devices_samples.append(pd.DataFrame(data={
                    "label": [str(device["label"].values[0])],
                    "mean_n_bytes": [device['packet_length'].mean()], 
                    "stdev_n_bytes": [device['packet_length'].std(ddof=0)],
                    "min_n_bytes": [device['packet_length'].min()],
                    "max_n_bytes": [device['packet_length'].max()],
                    "sum_n_bytes": [device['packet_length'].sum()],
                    "median_n_bytes": [device['packet_length'].median()],
                    "mean_iat": [device['iat'].mean()], 
                    "stdev_iat": [device['iat'].std(ddof=0)],
                    "min_iat": [device['iat'].min()],
                    "max_iat": [device['iat'].max()],
                    "sum_iat": [device['iat'].sum()],
                    "median_iat": [device['iat'].median()]
                }))

    all_devices_samples = pd.concat(all_devices_samples)
    # os.makedirs(PACKETPLOTSDIR, exist_ok=True)
    # all_devices_samples.to_csv(PACKETPLOTSDIR + (file.split('/')[-1]), index=False)
    # print('Estatísticas salvas em:', PACKETPLOTSDIR + (file.split('/')[-1]))
    return all_devices_samples

def carregar_e_preprocessar_dados(caminho):
    

    # Pega todos os arquivos CSV que contenham "reduzido" no nome
    arquivos = glob(f"{caminho}/*full_packet_features_rotulado_reduzido.csv")
    arquivos = [f for f in arquivos if "reduzido" in f]

    # Carrega e concatena
    df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

    df = balanceamentoDataset(df)

    df_total = extrair_estatisticas_por_pacote(df)

    df_ameacas = df_total[df_total['label'] == 'injection'].reset_index(drop=True)
    #df_ameacas = df[df['label'].isin(['scanning','xss'])].reset_index(drop=True)
    #df_ameacas = df_ameacas[features_boas + ['label']]

    print(df_ameacas['label'].value_counts())

    df_total = df_total[df_total['label'] != 'injection'].reset_index(drop=True)
    #df = df[~df['label'].isin(['injection', 'password'])].reset_index(drop=True)
    #df = df[~df['label'].isin(['scanning','xss'])].reset_index(drop=True)

    print(df_total['label'].value_counts())

    X = df_total.drop('label', axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_total['label'])
    NB_labels = len(np.unique(y))

    del df, df_total

    return X, y, df_ameacas, NB_labels, label_encoder


def carregar_e_preprocessar_dadosV1(caminho):
    

    # Pega todos os arquivos CSV que NÃO contenham "reduzido" no nome
    arquivos = glob(f"{caminho}/*full_packet_features_rotulado.csv")
    arquivos = [f for f in arquivos if "reduzido" not in f]

    # Carrega e concatena
    df = pd.concat([pd.read_csv(f) for f in arquivos], ignore_index=True)

    df = balanceamentoDataset(df)

    features_boas = ['packet_length','ip_header_length','ttl','timestamp']

    df_ameacas = df[df['label'] == 'ransomware'].reset_index(drop=True)
    #df_ameacas = df[df['label'].isin(['scanning','xss'])].reset_index(drop=True)
    df_ameacas = df_ameacas[features_boas + ['label']]

    print(df_ameacas['label'].value_counts())

    df = df[df['label'] != 'ransomware'].reset_index(drop=True)
    #df = df[~df['label'].isin(['injection', 'password'])].reset_index(drop=True)
    #df = df[~df['label'].isin(['scanning','xss'])].reset_index(drop=True)

    print(df['label'].value_counts())

    X = df[features_boas] 

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    NB_labels = len(np.unique(y))

    del df

    return X, y, df_ameacas, NB_labels, label_encoder

def balanceamentoDataset(df):

    # Parâmetros de balanceamento
    target_max = 1_000_000
    target_min = 100_000

    # Cria novo DataFrame balanceado
    balanced_df = pd.DataFrame()

    # Verifica a contagem original por classe
    print("Distribuição original:\n", df['label'].value_counts())

    # Aplica undersampling ou oversampling conforme necessário
    for label, group in df.groupby('label'):
        count = len(group)
        
        if count > target_max:
            # Undersampling
            sampled = group.sample(n=target_max, random_state=42)
        elif count < target_min:
            # Oversampling
            sampled = resample(group, 
                               replace=True, 
                               n_samples=target_min, 
                               random_state=42)
        else:
            # Mantém como está
            sampled = group

        balanced_df = pd.concat([balanced_df, sampled])

    # Embaralha as linhas
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verifica a nova distribuição
    print("\nDistribuição após balanceamento:\n", balanced_df['label'].value_counts())

    return balanced_df


def treinar_modelo(X, y):

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2,stratify=y_temp, random_state=42, shuffle=True)

    #clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Cria o classificador
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',  # ou 'logloss' se for binário
        random_state=42,
        early_stopping_rounds = 10,
        tree_method='hist'  # mais rápido, bom para grandes volumes
    )

    clf.fit(X_train, y_train,eval_set = [(X_val, y_val)])
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    salvar_resultados_em_txt(accuracy, precision, recall, f1)

    return clf, X_train, X_val, X_test, y_train, y_val, y_test

def salvar_dados(temp_dir, X_train, X_val, X_test, y_train, y_val, y_test, df_ameacas, modelo, NB_labels,label_encoder):
    os.makedirs(temp_dir, exist_ok=True)
    
    joblib.dump(X_train, os.path.join(temp_dir, "X_train.pkl"))
    joblib.dump(X_val, os.path.join(temp_dir, "X_val.pkl"))
    joblib.dump(X_test, os.path.join(temp_dir, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(temp_dir, "y_train.pkl"))
    joblib.dump(y_val, os.path.join(temp_dir, "y_val.pkl"))
    joblib.dump(y_test, os.path.join(temp_dir, "y_test.pkl"))
    joblib.dump(NB_labels, os.path.join(temp_dir, "NB_labels.pkl"))
    joblib.dump(label_encoder.classes_, os.path.join(temp_dir, "name_classes.pkl"))
    
    df_ameacas.to_parquet(os.path.join(temp_dir, "df_ameacas.parquet"), index=False, compression='snappy')
    joblib.dump(modelo, os.path.join(temp_dir, "modelo_random_forest.pkl"))

def main():
    caminho = "./rotulados"
    temp_dir = "./temp_dir"

    print("[INFO] Carregando e pré-processando os dados...")
    X, y, df_ameacas, NB_labels, label_encoder = carregar_e_preprocessar_dados(caminho)
    print("[INFO] Dados carregados com sucesso.")
    print(f"[INFO] Formato de X: {X.shape}, y: {y.shape}")
    print(f"[INFO] Total de amostras de ameaças: {df_ameacas.shape[0]}")

    print("[INFO] Treinando o modelo...")
    modelo, X_train, X_val, X_test, y_train, y_val, y_test = treinar_modelo(X, y)
    print("[INFO] Modelo treinado com sucesso.")
    print(f"[INFO] Tamanhos -> Treino: {X_train.shape[0]}, Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")

    print("[INFO] Salvando dados e modelo na pasta temporária...")
    salvar_dados(temp_dir, X_train, X_val, X_test, y_train, y_val, y_test, df_ameacas, modelo, NB_labels, label_encoder)
    print("[INFO] Dados salvos com sucesso em:", temp_dir)

if __name__ == "__main__":
    main()