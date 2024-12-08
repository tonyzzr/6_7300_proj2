import numpy as np
import matplotlib.pyplot as plt

from dataset import PatientDataset, M1Imputation

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
    patient_no = 0


    x, y = dataset_train[patient_no]
    print(f"\nA single patient's records:")
    print(f"x: (t, n_features), y: (t,)")
    print(f"x: {x.shape}, y: {y.shape}")

    # plot sepsis label

    m1 = M1Imputation(dataset=dataset_train)
    dataset_train_m1_imputed = m1.impute_dataset()


    # plot heatmap
    plot_data_heatmap(dataset=dataset_train, patient_no=patient_no)
    plot_data_heatmap(dataset=dataset_train_m1_imputed, patient_no=patient_no)


    ## check if one feature is always nan
    
    # mean_features = []
    # x_feat_idx = 7
    # for i in range(5000):
    #     mean_features.append(np.nanmean(dataset_train[i][0][:, x_feat_idx]))

    # mean_features = np.array(mean_features)
    # print(mean_features)
    # plt.plot(mean_features, '.')
    # plt.show()

    # non_zero_indices = np.nonzero(sepsis_label)[0]
    # print(non_zero_indices)
