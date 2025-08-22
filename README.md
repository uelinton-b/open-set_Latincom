# Open Set Latincom - Detecção de Ataques Zero-Day em Redes IoT

Este repositório contém o cédigo para análise de tráfego de redes IoT e **detecção de ataques zero-day**, incluindo ataques de varredura de portas (port scan), utilizando técnicas de **Open Set Recognition**.

---

## Estrutura do Projeto

- `processPcap.py` e `rotulagem.py`  
  - Ponto inicial da análise de tráfego.  
  - Processam arquivos PCAP, filtram pacotes, extraem features e realizam a rotulagem dos dados.

- `run.sh`  
  - Script auxiliar para executar processos sequenciais do pipeline de análise e salva o tempo de processamento.

- `process.py`  
  - Calcula métricas estat?sticas sobre as features extraídas.  
  - Treina o modelo inicial.

- `train.py`  
  - Executa o fluxo principal de treinamento e avaliação.  
  - Carrega o modelo com `joblib.load(os.path.join(temp_dir, "modelo_random_forest.pkl"))`.  
  - Calcula **Mean Activation Vectors (MAVs)** com `calcular_mavs`.  
  - Define thresholds para detecção de ataques zero-day (open set).
  - Gera previsões para o conjunto de teste fechado e para ameaças open set.  
  - Executa a classifica??o final com detec??o de ataques zero-day usando `final_classification`.

---

## Como usar

1. Certifique-se de ter instalado as dependencias do projeto (Python >= 3.8).  
2. Coloque seus arquivos PCAP na pasta apropriada (`datasets/`).  
3. Execute o pipeline inicial para processar os PCAPs e rotular os dados:
