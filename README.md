# Open Set Latincom - Detecção de Ataques Zero-Day em Redes IoT

Este repositório contém o código para análise de tráfego de redes IoT e **detecção de ataques zero-day**, utilizando técnicas de **Open Set Recognition**.

O projeto utiliza o dataset ToN-IoT da UNSW, que é uma coleçãoo de dados heterogéneos coletados de sensores IoT e IIoT, sistemas operacionais (Windows 7 e 10, Ubuntu 14 e 18) e tráfego de rede. Os dados foram coletados em um ambiente de rede realista projetado no Cyber Range e IoT Labs da UNSW Canberra @ADFA, incluindo técnicas de ataque como DoS, DDoS e ransomware contra aplicações web, gateways IoT e sistemas computacionais.

---

## Instalação das Dependéncias

Certifique-se de ter Python >= 3.8 e instale as bibliotecas necessárias:

- `requirements.txt`
  - pip install -r requirements.txt


## Estrutura do Projeto

- `processPcap.py` e `rotulagem.py`  
  - Ponto inicial da análise de tráfego.  
  - Processam arquivos PCAP, filtram pacotes, extraem features e realizam a rotulagem dos dados.

- `run.sh`  
  - Script auxiliar para executar processos sequenciais do pipeline de análise e salva o tempo de processamento.

- `process.py`  
  - Calcula métricas estatásticas sobre as features extraídas.  
  - Treina o modelo inicial.

- `train.py`  
  - Executa o fluxo principal de treinamento e avaliação.  
  - Carrega o modelo com `joblib.load(os.path.join(temp_dir, "modelo_random_forest.pkl"))`.  
  - Calcula **Mean Activation Vectors (MAVs)** com `calcular_mavs`.  
  - Define thresholds para detecção de ataques zero-day (open set).
  - Gera previsões para o conjunto de teste fechado e para ameaças open set.  
  - Executa a classificação final com detecção de ataques zero-day usando `final_classification`.

## Estrutura de Pastas

- `datasets/` - Arquivos PCAP originais (precisam ser adicionados pelo usuario)  
- `featuresPackts/` - Features extraídas dos PCAPs  
- `rotulados/` - Dados rotulados prontos para treinamento  
- `resultPlots/` - Graficos e visualizações  
- `results/` - Resultados intermediários do processo  
- `temp_dir/` - Diretórios temporários de processamento  
---

## Como usar

1. Certifique-se de ter instalado as dependencias do projeto (Python >= 3.8).  
2. Coloque seus arquivos PCAP na pasta apropriada (`datasets/`).  
3. Execute o pipeline inicial para processar os PCAPs e rotular os dados.
4. run.sh para executar o treinamento e testes.
