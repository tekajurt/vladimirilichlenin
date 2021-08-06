import torch
from transformers import BertForMaskedLM, BertTokenizer
import pandas as pd
import re
masked_indxs = []
tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("pytorch/")
#e=model.eval()

texto = input("Ingresar el texto que desee predecir:\n")
text = "[CLS] " + texto+" [MASK]" + " [SEP]"
string = text.split(" ")
print('el texto es',text)
index = string.index('[MASK]')
print('y el mask esta en la posicion ',index)
masked_indxs.append(index)
#text = "[CLS] para terminar con mi [MASK], debo tomar el tenedor y [MASK] [SEP]"
#masked_indxs = (5,13)

tokens = tokenizer.tokenize(text)
indexed_toxens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_toxens])

predictions = model(tokens_tensor)[0]
print('El texto es: ',text)
for i,midx in enumerate(masked_indxs):
    idxs = torch.argsort(predictions[0,midx], descending=True)
    predicted_token = tokenizer.convert_ids_to_tokens(idxs[:5])
    print('posibilidades: ',predicted_token)
