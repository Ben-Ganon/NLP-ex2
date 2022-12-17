from transformers import RobertaTokenizer, RobertaModel, pipeline
import numpy as np
from numpy.linalg import norm
import torch



def cosine(v1, v2):
    cos = np.dot(v1, v2)/(norm(v1) * norm(v2))
    return cos

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



featue_ex = pipeline('feature-extraction', model='roberta-base')
featuer_mask = pipeline('fill-mask', model='roberta-base')

mask_sen = "hello i am <mask>"

print(featuer_mask(mask_sen))

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')



sentence1 = "I took the dog for a walk"
sentence2 = "The dog's wanted to be taken for a walk"

encoded_in_1 = tokenizer(sentence1, return_tensors='pt')
encoded_in_2 = tokenizer(sentence2, return_tensors='pt')
dog_tok = tokenizer("dog", return_tensors='pt')
output1 = model(**encoded_in_1)
output2 = model(**encoded_in_2)



sentence3 = "a matrix is a rectangular array or table of numbers, symbols, or expressions, arranged in rows and columns, which is used to represent a mathematical object or a property of such an object."

sentence4 = "It is neo from the matrix, which is an action movie from 1999, it stars keanu reeves as neo from the matrix and also he is in  a coma the matrix isnt actually real it's all binary code he can do kung-foo"
encoded_in_3 = tokenizer(sentence3, return_tensors='pt')
encoded_in_4 = tokenizer(sentence4, return_tensors='pt')
output3 = model(**encoded_in_3)
output4 = model(**encoded_in_4)

cos = torch.nn.CosineSimilarity(dim=1)

# sent_1_embed = mean_pooling(output1[0][:,4,:], encoded_in_1['attention_mask'])
# sent_2_embed = mean_pooling(output2[0][:,2,:], encoded_in_2['attention_mask'])
sent_3_embed = mean_pooling(output3, encoded_in_3['attention_mask'])
sent_4_embed = mean_pooling(output4, encoded_in_4['attention_mask'])

print(cos(output1[0][:,3,:], output2[0][:,1,:]))
print(cos(output3[0][:,1,:], output4[0][:,5,:]))


# print(cos(sent_1_embed, sent_2_embed))
# print(cos(sent_3_embed, sent_4_embed))


