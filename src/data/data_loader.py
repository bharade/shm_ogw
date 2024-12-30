import numpy as np
# import the Path class from the pathlib module
from pathlib import Path
import pandas as pd
import os
import sys
import csv
import random
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self,baseline_path:Path,damage_path:Path):
        self.baseline_path=baseline_path
        self.damage_path=damage_path

    def load_data(self):
        """load and preprocess the data from csv files"""
        udam=[]
        dam=[]
        print("Loading and preprocessing Baseline data...")
        self._process_csv_files(self.baseline_path,udam,0)
        print("Loading and preprocessing Damage data...")
        self._process_csv_files(self.damage_path,dam,1)
        print("Normalizing data...")
        self._normlize_data(udam)
        self._normlize_data(dam)
        print("Augmenting damage data...")
        dam=self._augment_damage_data(dam)
        print("Data loading and preprocessing completed.")
        return udam,dam


    def _process_csv_files(self,folder_path,database,condition):
        """process the csv files in the folder path"""
        sample_id_counter=1
        # Define the allowed frequency range (40kHz to 160kHz)
        allowed_frequencies = set(range(40, 161, 20))  # [40, 60, 80, ..., 160]    
        for filename in os.listdir(folder_path):
            filepath=os.path.join(folder_path,filename)
            if filename.endswith('.csv'):
                try:
                    frequency_str = filename.split('_data_')[1].replace('kHz.csv', '')
                    frequency = int(frequency_str)
                    if frequency in allowed_frequencies:
                        with open(filepath,'r') as file:
                            reader=csv.reader(file)
                            data=np.array([[row[2],row[3]] for row in reader],dtype=float)# take the 3rd and 4th column only 
                        sequence1=data[:,0]
                        sequence2=data[:,1]
                        sample1={'Sample ID':sample_id_counter,'Condition':condition,'Data':sequence1.tolist()}
                        sample2={'Sample ID':sample_id_counter+1,'Condition':condition,'Data':sequence2.tolist()}
                        database.append(sample1)
                        database.append(sample2)
                        sample_id_counter+=2
                except (IndexError, ValueError):
                    continue# Skip the file if frequency extraction fails
    
    def _normlize_data(self,samples):
        """normalize the data"""
        for sample in samples:
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
    
    def _augment_damage_data(self,damage_samples,augment_factor=1, noise_level=0.01):
        """augment the damage data"""
        augmented_samples = []
        print(f"Augmenting damage data...{len(damage_samples)}")
        for sample in damage_samples:
            original_data = np.array(sample['Data'])
            for _ in range(augment_factor):
                # Create noise and apply to the original data
                noise = np.random.normal(0, noise_level, original_data.shape)
                augmented_data = original_data + noise
                # Create an augmented sample with the same metadata
                augmented_sample = {
                    'Sample ID': f"{sample['Sample ID']}_aug",
                    'Condition': sample['Condition'],  # Keep the same condition
                    'Data': augmented_data.tolist()
                }
                augmented_samples.append(augmented_sample)
        print(f"Augmented damage data...{len(augmented_samples)}")
        damage_samples.extend(augmented_samples)
        return damage_samples
    

    def split_data(self,udam,dam,test_size=0.2,val_size=0.2, random_state=42):
        """split the data into train, validation and test sets"""
        full_database=udam+dam
        random.shuffle(full_database)
        data=[sample['Data'] for sample in full_database]
        labels=[sample['Condition'] for sample in full_database]
        #split for test set
        x_train_val,x_test,y_train_val,y_test=train_test_split(data,labels,test_size=test_size,random_state=random_state)
        #split for validation set
        x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val,test_size=val_size/(1-test_size),random_state=random_state)

        return np.array(x_train),np.array(x_val),np.array(x_test),np.array(y_train),np.array(y_val),np.array(y_test)
    
    def downsample_data(self,x_train,x_val,x_test):
        print("before downsampling:",x_train.shape)
        # Convert lists to NumPy arrays for compatibility with machine learning models
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_val= np.array(x_val)
        # Print the shapes of the resulting arrays for verification
        print(f"x_train shape: {x_train.shape}")
        #print(f"y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        #print(f"y_test shape: {y_test.shape}")
        x_train_max = np.max(x_train.reshape(x_train.shape[0], -1, 5), axis=2)
        x_test_max = np.max(x_test.reshape(x_test.shape[0], -1, 5), axis=2)
        x_val_max = np.max(x_val.reshape(x_val.shape[0], -1, 5), axis=2)
        print(f"after downsampling: 1. train: {x_train_max.shape}\n 2. validation: {x_val_max.shape}\n 3. test:{x_test_max.shape}")
        return x_train_max,x_val_max,x_test_max
    



