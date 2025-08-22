import numpy as np
from collections import defaultdict
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


def calcular_mavs(clf, X_train, y_train):
	probas = clf.predict_proba(X_train)
	mavs = defaultdict(list)

	for prob, label in zip(probas, y_train):
		mavs[label].append(prob)

	for classe in mavs:
		mavs[classe] = np.mean(mavs[classe], axis=0)

	#Mean_vectors = np.array([mavs[i] for i in range(len(np.unique(y_train)))])
	return mavs


def calcular_thresholds(clf, X_val, y_val, mavs, TH_value):
	probas_valid = clf.predict_proba(X_val)
	distancias_por_classe = defaultdict(list)
	thresholds_1 = {}
	thresholds_2 = defaultdict(lambda: defaultdict(list))
	thresholds_2_final = {}
	thresholds_3_final = {}

	for prob, label in zip(probas_valid, y_val):
		label = int(label)
		dist = np.linalg.norm(prob - mavs[label])
		distancias_por_classe[label].append(dist)

		for other_label in mavs:
			if other_label != label:
				dist_to_other = np.linalg.norm(prob - mavs[other_label])
				thresholds_2[label][other_label].append(dist_to_other)

	# Método 1
	for classe, dists in distancias_por_classe.items():
		dists_sorted = sorted(dists)
		index = int(len(dists_sorted) * TH_value)
		thresholds_1[classe] = dists_sorted[index]

	# Método 2
	for classe, other_dists in thresholds_2.items():
		min_dists = []
		for other_label, dists in other_dists.items():
			min_dists.extend(dists)
		thresholds_2_final[classe] = np.mean(min_dists)

	# Método 3
	for classe, dists in distancias_por_classe.items():
		dist_true_class = np.mean(dists)
		other_classes = [c for c in mavs if c != classe]
		sum_other_distances = sum(
			np.mean([
				np.linalg.norm(prob - mavs[other_class])
				for prob, lbl in zip(probas_valid, y_val) if lbl == other_class
			]) for other_class in other_classes
		)

		if sum_other_distances == 0:
			thresholds_3_final[classe] = dist_true_class
		else:
			thresholds_3_final[classe] = dist_true_class / sum_other_distances

	return thresholds_1, thresholds_2_final, thresholds_3_final


def final_classification(NB_CLASSES,name_classes, model_predictions_test, model_predictions_open, y_test, y_open, Mean_vectors, Threasholds_1, Threasholds_2, Threasholds_3):

	print("\n", "############## Distance Method 1 #################################")
	prediction_classes = []
	for i in range(len(model_predictions_test)):

		d = np.argmax(model_predictions_test[i], axis=0)
		if np.linalg.norm(model_predictions_test[i] - Mean_vectors[d]) > Threasholds_1[d]:
			prediction_classes.append(NB_CLASSES)
		else:
			prediction_classes.append(d)
	prediction_classes = np.array(prediction_classes)

	prediction_classes_open = []
	for i in range(len(model_predictions_open)):
		d = np.argmax(model_predictions_open[i], axis=0)
		if np.linalg.norm(model_predictions_open[i] - Mean_vectors[d]) > Threasholds_1[d]:
			prediction_classes_open.append(NB_CLASSES)
		else:
			prediction_classes_open.append(d)
	prediction_classes_open = np.array(prediction_classes_open)
	print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES,name_classes, 'method-1')

	print("\n", "############## Distance Method 2 #################################")
	prediction_classes = []
	for i in range(len(model_predictions_test)):
		d = np.argmax(model_predictions_test[i], axis=0)
		dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
		
		Tot = 0
		count = 0
		for k in range(NB_CLASSES):
			if k != d:
				Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i])
				count += 1
		media_outros = Tot / count if count > 0 else 1e-6

		if media_outros < Threasholds_2[d]:
			prediction_classes.append(NB_CLASSES)
		else:
			prediction_classes.append(d)

	prediction_classes_open = []
	for i in range(len(model_predictions_open)):
		d = np.argmax(model_predictions_open[i], axis=0)
		dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
		
		Tot = 0
		count = 0
		for k in range(NB_CLASSES):
			if k != d:
				Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i])
				count += 1
		media_outros = Tot / count if count > 0 else 1e-6

		if media_outros < Threasholds_2[d]:
			prediction_classes_open.append(NB_CLASSES)
		else:
			prediction_classes_open.append(d)

	prediction_classes = np.array(prediction_classes)
	prediction_classes_open = np.array(prediction_classes_open)
	print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES,name_classes, 'method-2')

	print("\n", "############## Distance Method 3 #################################")
	prediction_classes = []
	epsilon = 1e-8
	#Tot = dist / (Tot + epsilon)
	
	for i in range(len(model_predictions_test)):
		d = np.argmax(model_predictions_test[i], axis=0)
		dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
		
		Tot = 0
		for k in range(NB_CLASSES):
			if k != d:
				Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i])
		Tot = Tot if Tot != 0 else 1e-6

		razao = dist / Tot
		if razao > Threasholds_3[d]:
			prediction_classes.append(NB_CLASSES)
		else:
			prediction_classes.append(d)

	prediction_classes_open = []
	for i in range(len(model_predictions_open)):
		d = np.argmax(model_predictions_open[i], axis=0)
		dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
		
		Tot = 0
		for k in range(NB_CLASSES):
			if k != d:
				Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i])
		Tot = Tot if Tot != 0 else 1e-6

		razao = dist / (Tot + 1e-8)
		if razao < Threasholds_3[d]:
			prediction_classes_open.append(NB_CLASSES)
		else:
			prediction_classes_open.append(d)

	prediction_classes = np.array(prediction_classes)
	prediction_classes_open = np.array(prediction_classes_open)
	print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES,name_classes, 'method-3')

def Micro_F1(Matrix, NB_CLASSES):
	epsilon = 1e-8
	TP = FP = TN = 0

	for k in range(NB_CLASSES):
		TP += Matrix[k][k]
		FP += (np.sum(Matrix, axis=0)[k] - Matrix[k][k])
		TN += (np.sum(Matrix, axis=1)[k] - Matrix[k][k])

	Micro_Prec = TP / (TP + FP + epsilon)
	Micro_Rec = TP / (TP + TN + epsilon)
	return 2 * Micro_Prec * Micro_Rec / (Micro_Rec + Micro_Prec + epsilon)


def Macro_F1(Matrix, NB_CLASSES):
	epsilon = 1e-8
	Precisions = np.zeros(NB_CLASSES)
	Recalls = np.zeros(NB_CLASSES)

	for k in range(NB_CLASSES):
		Precisions[k] = Matrix[k][k] / (np.sum(Matrix, axis=0)[k] + epsilon)
		Recalls[k] = Matrix[k][k] / (np.sum(Matrix, axis=1)[k] + epsilon)

	Precision = np.mean(Precisions)
	Recall = np.mean(Recalls)

	return 2 * Precision * Recall / (Precision + Recall + epsilon)

def plot_confusion_matrix(cm, labels, title='Matriz de Confusão', cmap='YlGnBu', filename=None):
    # Normaliza a matriz por linha (por classe real)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_percent = cm_normalized * 100  # Converte para porcentagem

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=cmap,
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Percentage (%)'})
    #plt.title(title)
    plt.ylabel('True Label',fontsize=12)
    plt.xticks(rotation=45, ha='right',fontsize=12)
    plt.xlabel('Predicted Label',fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    if filename:
        output_dir = './resultPlots'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Matriz de Confusão salva em: {filepath}")
    plt.close()
    
    #plt.show() # Continua mostrando a matriz na tela

## Função Modificada para Plotar Métricas por Classe (Agora combinada)
def plot_metrics_per_class(precisions, recalls, fscores, labels, KLND_type, set_name):
    x = np.arange(len(labels))  # Posições dos rótulos
    width = 0.25  # Largura das barras

    fig, ax = plt.subplots(figsize=(14, 7)) # Aumentei o tamanho para acomodar mais classes
    rects1 = ax.bar(x - width, precisions, width, label='Precisão')
    rects2 = ax.bar(x, recalls, width, label='Recall')
    rects3 = ax.bar(x + width, fscores, width, label='F-score')

    ax.set_ylabel('Pontuação')
    ax.set_title(f'Métricas por Classe para o {set_name} - Método {KLND_type}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])

    fig.tight_layout()

    # Salvar o gráfico
    output_dir = './resultPlots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f'metrics_per_class_combined_{KLND_type}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Gráfico de métricas por classe combinado salvo em: {filepath}")

    #plt.show()

def print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES,name_classes, KLND_type):

	y_test = y_test.astype(int)
	y_open = np.array(y_open).astype(int)
	
	acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
	print('Test accuracy Normal model_Closed_set :', acc_Close)

	acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
	print('Test accuracy Normal model_Open_set :', acc_Open)

	y_test = y_test[:len(prediction_classes)]
	y_open = y_open[:len(prediction_classes_open)]

	# Criar um conjunto de dados combinado para cálculo das métricas
	y_true_combined = np.concatenate((y_test, y_open))
	y_pred_combined = np.concatenate((prediction_classes, prediction_classes_open))

	#name_classes = ['Backdoor', 'DDos', 'Dos', 'MITM', 'Normal', ''Ransomware',', 'Scanning', 'XSS']
	name_classes = ['Backdoor','DDos','Dos','MITM','Normal','Ransomware','Password','Scanning', 'XSS']


    # Definir todos os rótulos possíveis para as métricas combinadas
	all_labels_combined = list(range(NB_CLASSES)) + [NB_CLASSES] # De 0 até NB_CLASSES-1 (conhecidas) e NB_CLASSES (aberta)
	#display_labels_combined = [str(i) for i in range(name_classes)] + ['Open']
	display_labels_combined = list(name_classes) + ['Open Set']
	labels_combined = list(name_classes) + ['Open Set']

	prec_combined, rec_combined, f_combined, _ = precision_recall_fscore_support(
        y_true_combined, y_pred_combined, labels=all_labels_combined, average=None, zero_division=0
    )
    
	for i, label_display in enumerate(display_labels_combined):
		print(f"Classe {label_display}: Precisão={prec_combined[i]:.4f}, Recall={rec_combined[i]:.4f}, F-score={f_combined[i]:.4f}")
    
	plot_metrics_per_class(prec_combined, rec_combined, f_combined, display_labels_combined, KLND_type, 'Conjunto Combinado')

	cm_combined = confusion_matrix(y_true_combined, y_pred_combined, labels=all_labels_combined)
	print(f"\nMatriz de Confusão Combinada (Fechado + Aberto) - Método {KLND_type}:")
	print(cm_combined)

	plot_confusion_matrix(cm_combined, labels_combined,
	                      title=f'Matriz de Confusão Combinada ({KLND_type})',
	                      filename=f'confusion_matrix_combined_{KLND_type}.png')

	Matrix = []
	for i in range(NB_CLASSES + 1):
		Matrix.append(np.zeros(NB_CLASSES + 1))

	for i in range(len(y_test)):
		Matrix[y_test[i]][prediction_classes[i]] += 1

	for i in range(len(y_open)):
		Matrix[y_open[i]][prediction_classes_open[i]] += 1


	print("\n", "Micro")
	F1_Score_micro = Micro_F1(Matrix, NB_CLASSES)
	print("Average Micro F1_Score: ", F1_Score_micro)

	print("\n", "Macro")
	F1_Score_macro = Macro_F1(Matrix, NB_CLASSES)
	print("Average Macro F1_Score: ", F1_Score_macro)

	text_file = open("./results/results_open.txt", "a")

	text_file.write('########' + KLND_type + '#########\n')
	text_file.write('Test accuracy Normal model_Closed_set :'+ str(acc_Close) + '\n')
	text_file.write('Test accuracy Normal model_Open_set :'+ str(acc_Open) + '\n')
	text_file.write("Average Micro F1_Score: " + str(F1_Score_micro) + '\n')
	text_file.write("Average Macro F1_Score: " + str(F1_Score_macro) + '\n')
	text_file.write('\n')
	text_file.close()

def ajustaOpen(df_ameacas, NB_labels):

	df_ameacas = df_ameacas[df_ameacas['label'] == 'injection'].reset_index(drop=True)
	X_open = df_ameacas.drop(columns=['label'])
	X_open = X_open.to_numpy()

	total = NB_labels
	#print(total)
	#print(len(df_ameacas))
	y_open = [total] * len(df_ameacas)
	#print(y_open)
	
	del df_ameacas
	return X_open, y_open

def main():
	# Caminho do diretório temporário onde os arquivos estão salvos
	temp_dir = "./temp_dir"  

	X_train = joblib.load(os.path.join(temp_dir, "X_train.pkl"))
	X_val = joblib.load(os.path.join(temp_dir, "X_val.pkl"))
	X_test = joblib.load(os.path.join(temp_dir, "X_test.pkl"))

	y_train = joblib.load(os.path.join(temp_dir, "y_train.pkl"))
	y_val = joblib.load(os.path.join(temp_dir, "y_val.pkl"))
	y_test = joblib.load(os.path.join(temp_dir, "y_test.pkl"))

	NB_labels = joblib.load(os.path.join(temp_dir, "NB_labels.pkl"))

	name_classes = joblib.load(os.path.join(temp_dir, "name_classes.pkl"))
	print(name_classes)
	#exit()
	df_ameacas = pd.read_parquet(os.path.join(temp_dir, "df_ameacas.parquet"))

	X_open, y_open = ajustaOpen(df_ameacas, NB_labels)

	modelo = joblib.load(os.path.join(temp_dir, "modelo_random_forest.pkl"))

	print("[INFO] Calculando Mean Activation Vectors (MAVs)...")
	Mean_Vectors = calcular_mavs(modelo, X_train, y_train)

	print("[INFO] Calculando thresholds com TH_value=0.9...")
	Threasholds_1, Threasholds_2, Threasholds_3 = calcular_thresholds(
		modelo, X_val, y_val, Mean_Vectors, TH_value=0.9
	)

	print("[INFO] Gerando previsões para o conjunto de teste fechado...")
	model_predictions_test = modelo.predict_proba(X_test)

	print("[INFO] Gerando previsões para o conjunto de ameaças (open set)...")
	model_predictions_open = modelo.predict_proba(X_open)

	print("[INFO] Executando classificação final com detecção de open set...")
	final_classification(
		NB_labels,
		name_classes,
		model_predictions_test,
		model_predictions_open,
		y_test,
		y_open,
		Mean_Vectors,
		Threasholds_1,
		Threasholds_2,
		Threasholds_3
	)

	print("[INFO] Processo concluído com sucesso.")
if __name__ == "__main__":
	main()




# cm_closed = confusion_matrix(y_test, prediction_classes)
# 	print(f"\nMatriz de Confusão para o Conjunto Fechado ({KLND_type}):")
# 	print(cm_closed)

# 	# Plotar e salvar matriz de confusão para o conjunto fechado
# 	labels_closed = list(range(NB_CLASSES))
# 	plot_confusion_matrix(cm_closed, labels_closed,
#                           title=f'Matriz de Confusão - Conjunto Fechado ({KLND_type})',
#                           filename=f'confusion_matrix_closed_{KLND_type}.png') # Salva como PNG

# 	 ### Matriz de Confusão para o Conjunto Aberto
# 	all_possible_labels = list(range(NB_CLASSES)) + [NB_CLASSES]
# 	cm_open = confusion_matrix(y_open, prediction_classes_open, labels=all_possible_labels)
# 	print(f"\nMatriz de Confusão para o Conjunto Aberto ({KLND_type}):")
# 	print(cm_open)

#     # Plotar e salvar matriz de confusão para o conjunto aberto
# 	labels_open = list(range(NB_CLASSES)) + ['Open']
# 	plot_confusion_matrix(cm_open, labels_open,
#                           title=f'Matriz de Confusão - Conjunto Aberto ({KLND_type})',
#                           filename=f'confusion_matrix_open_{KLND_type}.png') # Salva como PNG