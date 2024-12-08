import numpy as np
import matplotlib.pyplot as plt

from dataset import PatientDataset


def plot_data_heatmap(dataset, patient_no = 0):

    fig, ax = plt.subplots(2, 1,
                        gridspec_kw={'height_ratios': [5, 1]},
                        figsize=(3, 8))

    ax[0].imshow(dataset[patient_no][0].T, aspect='auto', cmap='viridis')
    # ax[0].set_xlabel("time (h)")
    ax[0].set_ylabel("features")
    ax[0].set_xticks([])

    # ax[1].plot(dataset_train[patient_no][1])

    ax[1].imshow(dataset[patient_no][1].reshape(1, -1))
    ax[1].set_yticks([])

    ax[1].set_xlabel("time (h)")
    ax[1].set_ylabel("sepsis")

    plt.subplots_adjust(hspace=0.1)  # Adjust vertical space between subplots
    plt.show()

    return


if __name__ == "__main__":

    dataset_train = PatientDataset.from_file("data/train.npz")
    dataset_val = PatientDataset.from_file("data/val.npz")
    dataset_test = PatientDataset.from_file("data/test.npz")
    print(f"Train: {len(dataset_train)} patients")
    print(f"Val: {len(dataset_val)} patients")
    print(f"Test: {len(dataset_test)} patients")

    # print(f'dataset_train = {dataset_train.data}')

    x, y = dataset_train[0]
    print(f"\nA single patient's records:")
    print(f"x: (t, n_features), y: (t,)")
    print(f"x: {x.shape}, y: {y.shape}")

    # plot sepsis label

    sepsis_label = []
    for i in range(5000):
        sepsis_label.append(np.sum(dataset_train[i][1]))

    sepsis_label = np.array(sepsis_label)
    plt.plot(sepsis_label, '.')
    plt.show()

    non_zero_indices = np.nonzero(sepsis_label)[0]
    print(non_zero_indices)

    # plot heatmap

    patient_no = 0

    plot_data_heatmap(dataset=dataset_train, patient_no=patient_no)
