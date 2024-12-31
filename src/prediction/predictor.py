import os,sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data.data_loader import DataLoader
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
import csv

class Predictor:
    def __init__(self,model_path,path_to_sample):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.path_to_sample = path_to_sample
        self.samples=[]
    
    def process_sample(self):
        sample_id_counter=1
        for filename in os.listdir(self.path_to_sample):
            filepath=os.path.join(self.path_to_sample,filename)
            if filename.endswith('.csv'):
                try:
                    with open(filepath,'r') as file:
                        reader=csv.reader(file)
                        data=np.array([[row[2],row[3]] for row in reader],dtype=float)# take the 3rd and 4th column only 
                    sequence1=data[:,0]
                    sequence2=data[:,1]
                    sample1={'Sample ID':sample_id_counter,'Data':sequence1.tolist()}
                    sample2={'Sample ID':sample_id_counter+1,'Data':sequence2.tolist()}
                    self.samples.append(sample1)
                    self.samples.append(sample2)
                    sample_id_counter+=2
                except (IndexError, ValueError):
                    continue# Skip the file if frequency extraction fails

    def normlize_data(self):
        """normalize the data"""
        for sample in self.samples:
            data = np.array(sample['Data'])            
            # Calculate the min and max values
            min_val = np.min(data)
            max_val = np.max(data)
            # Apply min-max scaling
            if max_val > min_val:  # To avoid division by zero
                scaled_data = (data - min_val) / (max_val - min_val)
            else:
                scaled_data = data  # If min and max are the same, keep the data unchanged
            # Update the sample with the scaled data
            sample['Data'] = scaled_data.tolist()

    def downsample_data(self):
        data=[sample['Data'] for sample in self.samples]
        #split for test set
        x_predict=data
        #print("before downsampling:",x_predict.shape)
        x_predict1 = np.array(x_predict)
        x_pred_max = np.max(x_predict1.reshape(x_predict1.shape[0], -1, 5), axis=2)
        print(f"after downsampling: -> predict: {x_pred_max.shape}")
        return x_pred_max
    
    def predict(self,x_pred_max):
        y_pred_probs = self.model.predict(x_pred_max)
        y_pred = (y_pred_probs > 0.5).astype("int32")
        for i in range(len(y_pred)):
            print(f"Sample {i+1} is {'Damaged' if y_pred[i] == 1 else 'Undamaged'}")
    

