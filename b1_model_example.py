from model import B1Model
from dataset import PatientDataset
from utils import UtilityFunction, RealOutcomesSimulator

if __name__ == "__main__":

    dataset_train = PatientDataset.from_file("data/train.npz")
    dataset_val = PatientDataset.from_file("data/val.npz")
    dataset_test = PatientDataset.from_file("data/test.npz")
    print(f"Train: {len(dataset_train)} patients")
    print(f"Val: {len(dataset_val)} patients")
    print(f"Test: {len(dataset_test)} patients")

    my_model = B1Model()
    '''
        Always predict 1.
    '''
    my_model.fit(dataset_train)

    utility_fn = UtilityFunction()
    simulator_test = RealOutcomesSimulator(dataset_test, utility_fn)
    
    utility = simulator_test.compute_utility(my_model)
    print(f"the utility achieved is: {utility['u_total']}")
    print(f"example prediction {utility['preds'][0]}")
    print(f"confusion matrix {utility['cm']}")