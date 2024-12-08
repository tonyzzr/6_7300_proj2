from __future__ import annotations
from typing import List, Tuple
import numpy as np
import os
from tqdm import tqdm

#@title class PatientDataset():

class PatientDataset():
    def __init__(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]],
        feature_names: List[str],
    ) -> None:

        self.data = data
        self.feature_names = feature_names

        # ensure that all patients have the same number of features
        n_features = len(self.feature_names)
        for x, _ in self.data:
            assert x.shape[1] == n_features, "Number of features do not match across patients."

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns data for a single patient.
        There are `n_sample` timesteps for which data & sepsis labels are available.
        x: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        return self.data[idx]

    @classmethod
    def from_file(cls, file_name: str) -> PatientDataset:
        data = np.load(file_name)
        x_data = data['x']
        y_data = data['y']
        patient_to_num_records = data['patient_to_num_records']
        feature_names = data['feature_names'].tolist()

        # unzip the data
        unzipped_data = []
        start = 0
        for num_records in patient_to_num_records:
            end = start + num_records
            x = x_data[start:end]
            y = y_data[start:end]
            unzipped_data.append((x, y))
            start = end

        return cls(data=unzipped_data, feature_names=feature_names)
    
class M1Imputation():
    '''
        Impute missing values with global mean within the dataset.
    '''
    def __init__(self, dataset:PatientDataset):
        self.dataset = dataset
        

    def _compute_patient_mean(self):
        n_features = len(self.dataset.feature_names)

        patient_mean_features = np.zeros((len(self.dataset.data), n_features))
        for patient_idx in range(len(self.dataset.data)):
            x, y = self.dataset.data[patient_idx]
            patient_mean_features[patient_idx, :] = np.nanmean(x, axis=0)

        return patient_mean_features
    
    def _compute_global_mean(self):
        # n_features = len(self.dataset.feature_names)

        patient_mean_features = self._compute_patient_mean()
        global_mean_features = np.nanmean(patient_mean_features, axis=0)

        print(f"global_mean_features.shape = {global_mean_features.shape}")
        return global_mean_features
    
    def impute_data(self, x, global_mean_features):
        
        n_features = len(self.dataset.feature_names)

        _x = np.zeros_like(x)

        for feature_idx in range(n_features):
            if np.isnan(global_mean_features[feature_idx]):
                val = 0
            else:
                val = global_mean_features[feature_idx]

            # print(f'val = {val}')
            _x[:, feature_idx] = np.nan_to_num(x[:, feature_idx], 
                                              nan=val)
        
        return _x


    def impute_dataset(self):
        global_mean_features = self._compute_global_mean()

        imputed_data = []
        for patient_idx in range(len(self.dataset.data)):
            x, y = self.dataset.data[patient_idx]
            imputed_data.append((self.impute_data(x, global_mean_features), y))

        self.dataset_filled = imputed_data

        return self.dataset_filled

class M2Imputation(M1Imputation):
    '''
        Impute missing values with patient mean in the dataset.
        If no values recorded within the patient, impute global mean.
    '''
    def impute_dataset(self):
        global_mean_features = self._compute_global_mean()
        patient_mean_features = self._compute_patient_mean()

        imputed_data = []
        for patient_idx in range(len(self.dataset.data)):
            x, y = self.dataset.data[patient_idx]
            local_mean_features = patient_mean_features[patient_idx, :]
            imputed_data.append((self.impute_data(x, 
                                                  local_mean_features, 
                                                  global_mean_features), 
                                                  y))

        self.dataset_filled = imputed_data
        
        return self.dataset_filled 
    
    def impute_data(self, x, local_mean_features, global_mean_features):
        
        n_features = len(self.dataset.feature_names)

        _x = np.zeros_like(x)

        for feature_idx in range(n_features):
            if np.isnan(global_mean_features[feature_idx]):
                val = 0
            elif np.isnan(local_mean_features[feature_idx]):
                val = global_mean_features[feature_idx]
            else: 
                val = local_mean_features[feature_idx]

            # print(f'val = {val}')
            _x[:, feature_idx] = np.nan_to_num(x[:, feature_idx], 
                                              nan=val)
        
        return _x