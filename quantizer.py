import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import copy

class Quantizer:
    def __init__(self):
        pass
        
    @classmethod
    def from_data(cls, encoded_data, n_clusters, n_codebook):
        #penser a rajouter la colonne nulle
        quantizer = cls()

        quantizer.n_clusters = n_clusters
        quantizer.n_codebook = n_codebook
        quantizer.codebooks = []
        quantizer.ecart_type = []
        r = copy.deepcopy(encoded_data)

        for k in range(n_codebook):
            print("création du codebook n°", k+1)
            kmeans = KMeans(n_clusters = n_clusters-1, random_state = 42, max_iter=200)
            kmeans.fit(r)
            centers = kmeans.cluster_centers_

            labels = kmeans.labels_

            r -= centers[labels]

            #Changer l'ecart-type
            ecart_type = []
            global_std = r.std() 

            for i in range(n_clusters):
                cluster_r = r[labels==i]
                if len(cluster_r) <= 1:
                    ecart_type.append(global_std)
                else:
                    ecart_type.append(cluster_r.std())
            
            ecart_type = np.expand_dims(np.array(ecart_type), axis=1)
            quantizer.ecart_type.append(ecart_type)
            r /= quantizer.ecart_type[k][labels]
            quantizer.codebooks.append(centers)

        quantizer.codebooks = np.array(quantizer.codebooks)
        quantizer.ecart_type = np.array(quantizer.ecart_type[:-1])

        return quantizer
    
    @classmethod
    def from_codebooks(cls, codebooks, ecart_type):
        quantizer = cls()
        quantizer.n_codebook = len(codebooks)
        quantizer.n_clusters = len(codebooks[0])
        quantizer.codebooks = codebooks
        quantizer.ecart_type = ecart_type

        return quantizer

    def quantize(self, encoded_data):
        res = []
        r = copy.deepcopy(encoded_data)
        for k in range(self.n_codebook):
            r2 = np.sum(r**2, axis=1, keepdims=True)           
            c2 = np.sum(self.codebooks[k]**2, axis=1).reshape(1, -1)

            rc = r @ self.codebooks[k].T                        

            dist = r2 + c2 - 2 * rc
            near_cluster = np.argmin(dist, axis = 1)
            res.append(near_cluster)
            if k != self.n_codebook-1:
                r -= self.codebooks[k][near_cluster]
                r /= self.ecart_type[k][near_cluster]
        res = np.array(res)
        return res
    
    def deQuantize(self, encoded_quantized):
        r = copy.deepcopy(self.codebooks[-1][encoded_quantized[-1]])
        for k in range(self.n_codebook-2, -1, -1):
            r *= self.ecart_type[k][encoded_quantized[k]]
            r += self.codebooks[k][encoded_quantized[k]]
        return r