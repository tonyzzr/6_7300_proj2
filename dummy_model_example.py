from model import DummyModel
from dataset import PatientDataset
from utils import UtilityFunction, RealOutcomesSimulator

if __name__ == "__main__":

    dataset_train = PatientDataset.from_file("data/train.npz")
    dataset_val = PatientDataset.from_file("data/val.npz")
    dataset_test = PatientDataset.from_file("data/test.npz")
    print(f"Train: {len(dataset_train)} patients")
    print(f"Val: {len(dataset_val)} patients")
    print(f"Test: {len(dataset_test)} patients")

    my_model = DummyModel()
    my_model.fit(dataset_train)

    utility_fn = UtilityFunction()
    simulator_test = RealOutcomesSimulator(dataset_test, utility_fn)
    print(f"\nSimulating the hospitalization of patients in the test dataset with decisions made by your model, the utility achieved is: {simulator_test.compute_utility(my_model)}")