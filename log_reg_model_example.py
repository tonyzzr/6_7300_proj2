import torch.nn as nn

from model import NeuralNetModel, LogisticRegressionClassifier
from dataset import PatientDataset, M1Imputation
from utils import UtilityFunction, RealOutcomesSimulator

import torch
import random
import numpy as np

import matplotlib.pyplot as plt


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

    m1 = M1Imputation(dataset=dataset_train)
    dataset_train_m1_imputed = m1.impute_dataset()

    m1val = M1Imputation(dataset=dataset_val)
    dataset_val_m1_imputed = m1val.impute_dataset()

    m1test = M1Imputation(dataset=dataset_test)
    dataset_test_m1_imputed = m1test.impute_dataset()

    input_dim = dataset_train[0][0].shape[1] * 6

    my_model = NeuralNetModel()
    device = 'cuda'
    my_model.fit(
        train_data = dataset_train_m1_imputed,
        val_data = dataset_val_m1_imputed,
        config = {
        'classifer':LogisticRegressionClassifier(input_dim=input_dim).to(device),
        'lr':1e-2, 
        'criteria':nn.BCELoss(), 
        'n_epoch':0,
        'batch_size':128,
        'device':device,
        })

    utility_fn = UtilityFunction()
    simulator_test = RealOutcomesSimulator(dataset_train_m1_imputed, utility_fn)
    # print(f"\nSimulating the hospitalization of patients in the test dataset with decisions made by your model, the utility achieved is: {simulator_test.compute_utility(my_model)}")
    utility = simulator_test.compute_utility(my_model)
    print(f" the utility achieved is: {utility['u_total']}")

    plt.plot(utility['preds'][0], '.')
    plt.plot(dataset_val_m1_imputed[0][1], '.')
    plt.show()