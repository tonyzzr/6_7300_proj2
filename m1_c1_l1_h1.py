import torch.nn as nn

from model import NeuralNetModel, LogisticRegressionClassifier
from dataset import PatientDataset, M1Imputation, M2Imputation
from utils import UtilityFunction, RealOutcomesSimulator

import torch
import random
import numpy as np

import matplotlib.pyplot as plt


from model import EnhancedMLP

from loss import CustomBCELoss, UtilityLoss

def data_imputation(datasets={}, method=M2Imputation):

    imputed_datasets = {}
    
    for key in datasets.keys():
        m = method(dataset=datasets[key])
        imputed_datasets[key] = m.impute_dataset()

    return imputed_datasets

if __name__ == "__main__":
    # Set the random seed for PyTorch
    seed = 42  # Choose any integer
    torch.manual_seed(seed)

    # If you're using a GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multiple GPUs

    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Ensure deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train = PatientDataset.from_file("data/train.npz")
    dataset_val = PatientDataset.from_file("data/val.npz")
    dataset_test = PatientDataset.from_file("data/test.npz")
    print(f"Train: {len(dataset_train)} patients")
    print(f"Val: {len(dataset_val)} patients")
    print(f"Test: {len(dataset_test)} patients")


    '''
        M1 - C1 - L1 - H1
    '''
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter('runs/m1_c1_l1_h1')

    imputed_datasets = data_imputation(
        datasets={
            'train':dataset_train, 'val':dataset_val, 'test':dataset_test,
        },
        method=M1Imputation # M1
    )

    my_model = NeuralNetModel()
    device = 'cuda'
    my_model.fit(
        data = imputed_datasets,
        config = {
        'classifer':LogisticRegressionClassifier, # H1
        'lr':1e-3, 
        'criteria':CustomBCELoss(scale_factor=1),  # L1
        'n_epoch':50,
        'batch_size':128,
        'device':device,
        'class_balanced':False, # C1
        'writer':writer,
        })

    utility_fn = UtilityFunction()
    simulator_test = RealOutcomesSimulator(imputed_datasets['train'], utility_fn)
    # print(f"\nSimulating the hospitalization of patients in the test dataset with decisions made by your model, the utility achieved is: {simulator_test.compute_utility(my_model)}")
    utility = simulator_test.compute_utility(my_model)
    print(f" the utility achieved is: {utility['u_total']}")




