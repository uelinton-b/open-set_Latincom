import pandas as pd
import glob
import os

# === 1. Carrega o CSV de rótulos ===
df_rotulos = pd.read_csv('SecurityEvents_Network_datasets/GroundTruth_Completo.csv')
df_rotulos['ts'] = df_rotulos['ts'].astype(int)

# === 2. Agrupa rótulos por (timestamp, src_ip) ===
rotulos_grouped = df_rotulos.groupby(['ts', 'src_ip'])

# === 3. Cria a pasta de saída, se não existir ===
output_dir = 'rotulados'
os.makedirs(output_dir, exist_ok=True)

# === 4. Lista os arquivos .parquet ===
parquet_files = glob.glob('featuresPackts/*_full_packet_features.parquet')

# === 5. Processa cada arquivo ===
for file in parquet_files:
    print(f"🔄 Processando: {file}")
    
    df_base = pd.read_parquet(file)
    # Remove linhas com qualquer valor nulo
    df_base = df_base.dropna()
    # Remove linhas com ip_src vazio, se existir
    df_base = df_base[df_base['ip_src'] != '']

    df_base['timestamp_sec'] = df_base['timestamp'].astype(float).astype(int)
    df_base['label'] = 'normal'

    def rotular_linha(row):
        ts = row['timestamp_sec']
        ip_src = row['ip_src']

        if pd.isna(ip_src):
            return 'normal'
        if (ts, ip_src) in rotulos_grouped.groups:
            ataques = rotulos_grouped.get_group((ts, ip_src))
            return ataques['type'].values[0]
        return 'normal'

    df_base['label'] = df_base.apply(rotular_linha, axis=1)
    df_base['attack'] = df_base['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_base.drop(columns=['timestamp_sec'], inplace=True)

    # === 6. Salva o dataset completo ===
    base_name = os.path.basename(file).replace('.parquet', '_rotulado.csv')
    output_path_full = os.path.join(output_dir, base_name)
    df_base.to_csv(output_path_full, index=False)

    # === 7. Seleciona colunas para versão reduzida ===
    colunas_reduzidas = [
        'timestamp',
        'ip_src',
        'ip_dst',
        'src_port',
        'dst_port',
        'protocol',
        'packet_length',
        'label',
        'attack'
    ]

    df_reduzido = df_base[colunas_reduzidas].copy()

    # === 8. Salva a versão reduzida ===
    reduzido_name = base_name.replace('_rotulado.csv', '_rotulado_reduzido.csv')
    output_path_reduzido = os.path.join(output_dir, reduzido_name)
    df_reduzido.to_csv(output_path_reduzido, index=False)

    print(f"✅ Salvo completo:  {output_path_full}")
    print(f"✅ Salvo reduzido: {output_path_reduzido}")

print("🏁 Todos os arquivos foram rotulados e salvos em 'rotulados/'")
