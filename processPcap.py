import os
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, PcapReader
import pandas as pd
import logging

# Configurar logging para exibir mensagens informativas e de erro
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Definições de Caminho ---
# Seu diretório base com as subpastas de tipos de ataque
BASE_DIR = './datasets/Ataque/up/'
# Onde os arquivos Parquet resultantes serão salvos
OUTPUT_DIR = './featuresPackts/'

# Crie o diretório de saída se ele não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.info(f"Diretório de saída para arquivos Parquet: {OUTPUT_DIR}")

# --- 2. Função para Extrair Features de um Único Pacote ---
def extract_packet_features(packet):
    """
    Extrai features de nível de pacote individual de um objeto Scapy Packet.
    Retorna um dicionário com as features.
    """
    features = {
        'timestamp': packet.time,
        'packet_length': len(packet),
        'ip_src': None,
        'ip_dst': None,
        'ip_header_length': None, # Em bytes
        'ttl': None,
        'protocol': None, # Nome do protocolo (TCP, UDP, ICMP)
        'src_port': None,
        'dst_port': None,
        'tcp_flags_SYN': False,
        'tcp_flags_ACK': False,
        'tcp_flags_FIN': False,
        'tcp_flags_RST': False,
        'tcp_flags_PSH': False,
        'tcp_flags_URG': False,
        'tcp_header_length': None, # Em bytes
        'num_tcp_options': None,
        # Adicione mais features de pacote aqui se necessário (e.g., ICMP type/code)
    }

    # Verifica se o pacote possui a camada IP
    if packet.haslayer(IP):
        ip_layer = packet[IP]
        features['ip_src'] = ip_layer.src
        features['ip_dst'] = ip_layer.dst
        # ihl é em 32-bit words, então multiplicamos por 4 para obter bytes
        features['ip_header_length'] = ip_layer.ihl * 4
        features['ttl'] = ip_layer.ttl

        # Determina o protocolo da camada de transporte e extrai features específicas
        if ip_layer.haslayer(TCP):
            tcp_layer = ip_layer[TCP]
            features['protocol'] = 'TCP'
            features['src_port'] = tcp_layer.sport
            features['dst_port'] = tcp_layer.dport
            # Verifica as flags TCP individualmente
            flags_str = str(tcp_layer.flags)
            features['tcp_flags_SYN'] = 'S' in flags_str
            features['tcp_flags_ACK'] = 'A' in flags_str
            features['tcp_flags_FIN'] = 'F' in flags_str
            features['tcp_flags_RST'] = 'R' in flags_str
            features['tcp_flags_PSH'] = 'P' in flags_str
            features['tcp_flags_URG'] = 'U' in flags_str
            # dataofs é em 32-bit words, multiplica por 4 para obter bytes
            features['tcp_header_length'] = tcp_layer.dataofs * 4
            features['num_tcp_options'] = len(tcp_layer.options)

        elif ip_layer.haslayer(UDP):
            udp_layer = ip_layer[UDP]
            features['protocol'] = 'UDP'
            features['src_port'] = udp_layer.sport
            features['dst_port'] = udp_layer.dport
            # Features TCP não aplicáveis para UDP
            features['tcp_header_length'] = None
            features['num_tcp_options'] = None

        elif ip_layer.haslayer(ICMP):
            icmp_layer = ip_layer[ICMP]
            features['protocol'] = 'ICMP'
            # Portas e features TCP não aplicáveis para ICMP
            features['src_port'] = None
            features['dst_port'] = None
            features['tcp_header_length'] = None
            features['num_tcp_options'] = None
            # Você pode adicionar features ICMP específicas aqui, como icmp_layer.type e icmp_layer.code

    return features

# --- 3. Processamento dos Pcaps ---
def process_pcaps_in_directory(base_directory, output_directory):
    """
    Percorre os diretórios de tipos de ataque, processa os pcaps
    e salva as features em dois arquivos Parquet por tipo de ataque.
    """
    attack_type_folders = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    if not attack_type_folders:
        logging.warning(f"Nenhum subdiretório encontrado em '{base_directory}'. Verifique o caminho.")
        return

    for attack_type_folder in attack_type_folders:
        attack_path = os.path.join(base_directory, attack_type_folder)
        logging.info(f"\nIniciando processamento para o tipo de ataque: '{attack_type_folder}'")

        all_packet_features_for_attack = [] # Lista para armazenar features de todos os pcaps deste tipo de ataque

        pcap_files = [f for f in os.listdir(attack_path) if f.endswith('.pcap')]

        if not pcap_files:
            logging.info(f"Nenhum arquivo .pcap encontrado em '{attack_path}'. Pulando.")
            continue

        for pcap_file_name in pcap_files:
            pcap_file_path = os.path.join(attack_path, pcap_file_name)
            logging.info(f"  Extraindo de: '{pcap_file_name}'")

            try:
                # Usando PcapReader para lidar eficientemente com pcaps grandes (lê pacote por pacote)
                with PcapReader(pcap_file_path) as reader:
                    for packet in reader:
                        packet_features = extract_packet_features(packet)
                        all_packet_features_for_attack.append(packet_features)

            except Exception as e:
                logging.error(f"Erro ao processar '{pcap_file_path}': {e}")
                continue

        # --- Geração e Salvamento dos DataFrames ---
        if all_packet_features_for_attack:
            df_attack_features = pd.DataFrame(all_packet_features_for_attack)

            # --- 1. Salvar DataFrame COMPLETO ---
            output_full_parquet_name = f"{attack_type_folder}_full_packet_features.parquet"
            output_full_parquet_path = os.path.join(output_directory, output_full_parquet_name)
            df_attack_features.to_parquet(output_full_parquet_path, index=False)
            logging.info(f"  Features COMPLETAS salvas em: '{output_full_parquet_path}' ({len(df_attack_features)} pacotes)")

            # --- 2. Salvar DataFrame LEVE (somente tupla, timestamp, tamanho) ---
            # Colunas necessárias para a tupla (IPs, Portas, Protocolo), timestamp e tamanho
            light_features_columns = [
                'timestamp', 'packet_length', 'ip_src', 'ip_dst', 'src_port', 'dst_port', 'protocol'
            ]
            # Cria um novo DataFrame apenas com as colunas selecionadas
            df_light_features = df_attack_features[light_features_columns]

            output_light_parquet_name = f"{attack_type_folder}_light_packet_features.parquet"
            output_light_parquet_path = os.path.join(output_directory, output_light_parquet_name)
            df_light_features.to_parquet(output_light_parquet_path, index=False)
            logging.info(f"  Features LEVES salvas em: '{output_light_parquet_path}' ({len(df_light_features)} pacotes)")

        else:
            logging.info(f"  Nenhum pacote válido com features extraídas para o tipo de ataque '{attack_type_folder}'.")

# --- 4. Execução Principal ---
if __name__ == "__main__":
    logging.info("Iniciando o processo de extração de features de pcaps.")
    process_pcaps_in_directory(BASE_DIR, OUTPUT_DIR)
    logging.info("Processamento concluído!")