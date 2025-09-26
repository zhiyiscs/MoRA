import open_clip
import random
import torch
import io
import pyarrow as pa
import os
import numpy as np
from PIL import Image
import torch

def read_folder(folder):
    lis = []

    order = sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0]))
    for temp in order:
        x = torch.Tensor(np.load(folder + '/' + temp))
        lis.append(x)
    
    lis = torch.stack(lis,dim=0)

    return lis

training_image_embedding = read_folder("training_image_logit")
torch.save(training_image_embedding,"training_image_logit.pt")

test_image_embedding = read_folder("test_image_logit")
torch.save(test_image_embedding ,"test_image_logit.pt")



def kl_divergence(P, Q):
    # Ensure tensors have the same shape
    assert P.size() == Q.size(), "Tensors must have the same shape"

    P = torch.softmax(P,dim=-1)
    Q = torch.softmax(Q,dim=-1)
    
    # Compute element-wise logarithm
    log_P = torch.log(P)
    log_Q = torch.log(Q)
    
    # Compute element-wise difference
    log_diff = log_P - log_Q
    
    # Compute element-wise product
    prod = P * log_diff
    
    # Sum up all the elements
    kl_div = torch.sum(prod)
    
    return kl_div



flag = 0
i = 0

index_list = []
 
while flag < test_image_embedding.size(0):
    order = sorted(os.listdir("test_image_logit"), key=lambda x: int(x.split('.')[0]))
    if str(i) == order[flag].split('.')[0]:
        temp = test_image_embedding[flag]
        # Calculate cosine similarity between temp and each tensor in source
        cos_similarities = [torch.nn.functional.cosine_similarity(temp.unsqueeze(0), s, dim=1) for s in training_image_embedding]

        #kl = [kl_divergence(temp, training_image_embedding[s]) for s in range(training_image_embedding.size(0))]


        # Find the index of the tensor in source with the maximum similarity
        max_index = np.argmax([similarity.item() for similarity in cos_similarities])

        print(max_index)

        index_list.append(max_index)
        flag += 1
    else:
        index_list.append(i)
    i += 1



np.save("test_text_index.npy",index_list)



