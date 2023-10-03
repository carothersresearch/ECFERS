import numpy as np
import pandas as pd
import pickle
import torch
import os


CURRENT_DIR = os.getcwd()+"/src/kinetic_estimator/KM_prediction_function"

def calcualte_esm1b_vectors(model, batch_converter, enzyme_list):
	#creating model input:
	df_enzyme = preprocess_enzymes(enzyme_list)
	model_input = [(df_enzyme["ID"][ind], df_enzyme["model_input"][ind]) for ind in df_enzyme.index]
	seqs = [model_input[i][1] for i in range(len(model_input))]
	

	#convert input into batches:
	
	#Calculate ESM-1b representations
	df_enzyme["enzyme rep"] = ""

	for ind in df_enzyme.index:
		batch_labels, batch_strs, batch_tokens = batch_converter([(df_enzyme["ID"][ind], df_enzyme["model_input"][ind])])
		with torch.no_grad():
		    results = model(batch_tokens, repr_layers=[33])
		df_enzyme["enzyme rep"][ind] = results["representations"][33][0, 1 : len(df_enzyme["model_input"][ind]) + 1].mean(0).numpy()
	return(df_enzyme)



def preprocess_enzymes(enzyme_list):
	# df_enzyme = pd.DataFrame(data = {"amino acid sequence" : list(set(enzyme_list))})
	df_enzyme = pd.DataFrame(data = {"amino acid sequence" : enzyme_list})
	df_enzyme["ID"] = ["protein_" + str(ind) for ind in df_enzyme.index]
	#if length of sequence is longer than 1020 amino acids, we crop it:
	df_enzyme["model_input"] = [seq[:1022] for seq in df_enzyme["amino acid sequence"]]
	return(df_enzyme)
