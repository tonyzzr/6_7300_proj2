import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from dataset import PatientDataset
from loss import UtilityLoss


#@title class BaseModel():

class BaseModel():
    def fit(self, data: PatientDataset) -> None:
        """
        Data is a list of (x,y) pairs, one for each patient.
        x: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of shape (n_samples,)
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> float:
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: 0, 1 label. 1 if sepsis is predicted at the current timestep, 0 otherwise.
        """
        raise NotImplementedError

#@title DummyModel

from sklearn.dummy import DummyClassifier
class DummyModel(BaseModel):
    def fit(self, data: PatientDataset) -> None:

        final_features = []
        final_labels = []
        for x, y in data:

            # x is a single patient's features over their entire history: (t, n_features)
            # y is a single patient's outcomes over their entire history: (t,)

            # walk over the patient's history and create a feature vector for each timestep
            for t in range(len(x)):
                history_up_to_timestep_t = x[:t+1]
                label_at_timestep_t = y[t]
                final_features.append(self.process_history(history_up_to_timestep_t))
                final_labels.append(label_at_timestep_t)

        final_features = np.array(final_features)
        final_labels = np.array(final_labels)

        self.model = DummyClassifier()
        self.model.fit(final_features, final_labels)

    def process_history(self, x):
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: np.ndarray of shape (d,) representing a fixed-size feature vector.
        """
        # For the dummy model, we just return the last timestep's features
        return x[-1]

    def predict(self, x) -> float:
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: 0, 1 label. 1 if sepsis is predicted at the current timestep, 0 otherwise.
        """
        return self.model.predict([self.process_history(x)])[0] # label

class PerfectModel(DummyModel):
    def __init__(self):
        super().__init__()


class B1Model(DummyModel):
    '''
        Always predict 1.
    '''
    def fit(self, data: PatientDataset) -> None:

        final_features = []
        final_labels = []
        for x, y in data:

            # x is a single patient's features over their entire history: (t, n_features)
            # y is a single patient's outcomes over their entire history: (t,)

            # walk over the patient's history and create a feature vector for each timestep
            for t in range(len(x)):
                history_up_to_timestep_t = x[:t+1]
                label_at_timestep_t = y[t]
                final_features.append(self.process_history(history_up_to_timestep_t))
                final_labels.append(label_at_timestep_t)

        final_features = np.array(final_features)
        final_labels = np.array(final_labels)

        self.model = DummyClassifier(strategy='constant', constant=1)
        self.model.fit(final_features, final_labels)


class B2Model(DummyModel):
    '''
        Always predict 0.
    '''
    def fit(self, data: PatientDataset) -> None:

        final_features = []
        final_labels = []
        for x, y in data:

            # x is a single patient's features over their entire history: (t, n_features)
            # y is a single patient's outcomes over their entire history: (t,)

            # walk over the patient's history and create a feature vector for each timestep
            for t in range(len(x)):
                history_up_to_timestep_t = x[:t+1]
                label_at_timestep_t = y[t]
                final_features.append(self.process_history(history_up_to_timestep_t))
                final_labels.append(label_at_timestep_t)

        final_features = np.array(final_features)
        final_labels = np.array(final_labels)

        self.model = DummyClassifier(strategy='constant', constant=0)
        self.model.fit(final_features, final_labels)

# Logistic Regression Model
class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for binary classification
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)) 
    
class FinalDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# Multilayer Perceptron Model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size=60):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return torch.sigmoid(x)
    
# Define the Enhanced MLP Model
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=[60, 30, 10]):
        super(EnhancedMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.relu = nn.ReLU()
        self.n_layer = len(hidden_sizes)

        self.layer1 = nn.Linear(self.input_dim, self.hidden_sizes[0])
        self.layer2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.layer3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.layer_last = nn.Linear(self.hidden_sizes[-1], 1)



    def forward(self, x):

        
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.layer_last(x)
        return torch.sigmoid(x)
    
class NeuralNetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.scaler = None

    def preprocess(self, data, class_balanced=False, mode='train'):
        from torch.utils.data import WeightedRandomSampler

        final_features = []
        final_labels = []
        final_patient_spesis_labels= []
        final_patient_spesis_time = []
        current_time = []

        # print(data.__getitem__(0))


        for x, y in data:

            # x is a single patient's features over their entire history: (t, n_features)
            # y is a single patient's outcomes over their entire history: (t,)

            # walk over the patient's history and create a feature vector for each timestep
            sepsis_flag = 0
            sepsis_flag_time = np.inf
            for t in range(len(x)):
                history_up_to_timestep_t = x[:t+1]
                label_at_timestep_t = y[t]
                final_features.append(self.process_history(history_up_to_timestep_t))
                final_labels.append(label_at_timestep_t)
                current_time.append(t)

                if label_at_timestep_t and (not sepsis_flag):
                    sepsis_flag = 1
                    sepsis_flag_time = t
            
            if sepsis_flag:
                final_patient_spesis_labels += [1]*len(x)
                final_patient_spesis_time += [sepsis_flag_time]*len(x)
            else:
                final_patient_spesis_labels += [0]*len(x)
                final_patient_spesis_time += [np.inf]*len(x)                

        # final_features = np.array(final_features)
        # final_labels = np.array(final_labels)

        self.X_train = np.array(final_features)
        self.y_train = np.array(final_labels).reshape(-1, 1)
        self.sepsis_flag = np.array(final_patient_spesis_labels).reshape(-1, 1)
        self.sepsis_time = np.array(final_patient_spesis_time).reshape(-1, 1)
        self.current_time = np.array(current_time).reshape(-1, 1)
        self.y_with_sepsis_flag = np.concatenate((self.y_train, 
                                                  self.sepsis_flag,
                                                  self.sepsis_time,
                                                  self.current_time), axis=1)

        if self.scaler is None:
            self.scaler = preprocessing.StandardScaler().fit(self.X_train)
            self.X_scaled = self.scaler.transform(self.X_train)
            print('Scaler created!')
        else:
            print('Scaler already exist!')
            self.X_scaled = self.scaler.transform(self.X_train)

        # NN-specific
        final_dataset = FinalDataset(
            x = torch.tensor(self.X_scaled).float().to(self.config['device']),
            y = torch.tensor(self.y_with_sepsis_flag).float().to(self.config['device']),
        )

        if class_balanced:
            targets = [label[0].cpu().int() for _, label in final_dataset]
            class_counts = torch.tensor([targets.count(0), targets.count(1)])  # Counts for each class
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in targets]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


            final_dataloader = DataLoader(dataset=final_dataset, 
                                        sampler=sampler, 
                                        batch_size=self.config['batch_size'])
        else:
            final_dataloader = DataLoader(dataset=final_dataset, 
                                        batch_size=self.config['batch_size'])            
        
        return final_dataloader
        

    def fit(self, 
            # train_data: PatientDataset, 
            # val_data: PatientDataset,
            data:dict,
            config={},
            ) -> None:

        self.data = data

        train_data = data['train']
        val_data = data['val']
        test_data = data['test']

        # print(train_data)

        self.config = config
        self.writer = config['writer']
        final_dataloader_train = self.preprocess(train_data, class_balanced=self.config['class_balanced'])
        final_dataloader_val = self.preprocess(val_data, class_balanced=False) 
        final_dataloader_test = self.preprocess(test_data, class_balanced=False)
            

        input_dim = self.X_scaled.shape[1]
        print(f'input_dim = {input_dim}')

        self.model = config['classifer'](input_dim=input_dim, ).to(self.config['device'])
        self.criteria = config['criteria']
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=config['lr'],
                                    betas=(0.9, 0.999))

        self.loss_hist = []
        for epoch in tqdm(range(config['n_epoch'])):
            loss_epoch = []
            cm_epoch = {
                'TP':0, 'TN':0, 'FP':0, 'FN':0,
            }
            for x, y_with_sepsis_flag in final_dataloader_train:

                y = y_with_sepsis_flag[:, 0]
                sepsis_flags = y_with_sepsis_flag[:, 1]
                sepsis_times = y_with_sepsis_flag[:, 2]
                current_times = y_with_sepsis_flag[:, 3]

                self.optimizer.zero_grad()

                # print(f'x.size() = {x.size()}')

                y_pred = self.model(x).squeeze(1)
                # y_pred = self.model(x).squeeze(1) * 0 # FOR UTILITY LOSS DEBUG ONLY
                # y_pred = self.model(x).squeeze(1) * 0 + 1 # FOR UTILITY LOSS DEBUG ONLY

                # print(f'y_pred.size() = {y_pred.size()}')
                # print(f'y.size() = {y.size()}')

                if isinstance(self.criteria, UtilityLoss):
                    loss = self.criteria(y_pred, 
                                         y, 
                                         sepsis_flags=sepsis_flags, 
                                         sepsis_times=sepsis_times,
                                         current_times=current_times,
                                         )
                    # print(loss)
                else:
                    loss = self.criteria(y_pred, y)

                # # COMMENT HERE FOR UTILITY LOSS DEBUG ONLY
                loss.backward()
                self.optimizer.step()
                # # COMMENT HERE FOR UTILITY LOSS DEBUG ONLY

                loss_epoch.append(loss.item())

                y_pred_binary = (y_pred.detach().cpu().numpy() >=0.5).astype(int)
                y_binary = y.detach().cpu().numpy().astype(int)

                TP = np.sum((y_binary==1) & (y_pred_binary == 1))
                TN = np.sum((y_binary==0) & (y_pred_binary == 0))
                FP = np.sum((y_binary==0) & (y_pred_binary == 1))
                FN = np.sum((y_binary==1) & (y_pred_binary == 0))

                cm_epoch['TP'] += TP
                cm_epoch['TN'] += TN
                cm_epoch['FP'] += FP
                cm_epoch['FN'] += FN

            val_loss = self.validation(final_dataloader_val)
            # for key in cm_epoch:
            #     cm_epoch[key] /= len(self.y_train)

            if isinstance(self.criteria, UtilityLoss):
                self.loss_hist.append(
                    {
                        'train': np.sum(np.array(loss_epoch)),
                        'val': val_loss,
                        'cm': cm_epoch,
                    }
                )
            else:

                self.loss_hist.append(
                    {
                        'train': np.mean(np.array(loss_epoch)),
                        'val': val_loss,
                        'cm': cm_epoch,
                    }
                )
            

            print(f"Epoch [{epoch+1}/{config['n_epoch']}], Training Loss: {self.loss_hist[-1]['train']:.4f}, Val Loss: {self.loss_hist[-1]['val']:.4f}")
            print(f"confusion matrix: {self.loss_hist[-1]['cm']}")
            self.writer.add_scalar('Loss/train', self.loss_hist[-1]['train'], epoch)
            self.writer.add_scalar('Loss/val', self.loss_hist[-1]['val'], epoch)
            
            # test_utility = self.test_utility()
            # self.writer.add_scalar('Utility/test', test_utility, epoch)

            test_utility_fast = self.test_utility_fast(final_dataloader_test)
            self.writer.add_scalar('Utility/test_fast', test_utility_fast, epoch)

            val_utility_fast = self.val_utility_fast(final_dataloader_val)
            self.writer.add_scalar('Utility/val_fast', val_utility_fast, epoch)

        print("Training complete.")

    def process_history(self, x):
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: np.ndarray of shape (d,) representing a fixed-size feature vector.
        """
        # For the dummy model, we just return the last timestep's features
        # return x[-1]

        if len(x.shape) < 2:
            x = x.reshape(1, -1)

        t = np.array([x.shape[0]])

        # x_mean = np.mean(x, axis=0)
        # x_std = np.std(x, axis = 0)
        x_2hr_mean = x[-min(x.shape[0], 2):, :].mean(axis=0)
        x_3hr_mean = x[-min(x.shape[0], 3):, :].mean(axis=0)
        x_5hr_mean = x[-min(x.shape[0], 5):, :].mean(axis=0)
        x_7hr_mean = x[-min(x.shape[0], 7):, :].mean(axis=0)
        x_12hr_mean = x[-min(x.shape[0], 12):, :].mean(axis=0)
        

        return np.concatenate((
                               t,
                               x[-1], 
                               x_2hr_mean, 
                               x_3hr_mean, 
                               x_5hr_mean, 
                               x_7hr_mean,
                               x_12hr_mean,), axis=0)

        

    def predict(self, x) -> float:
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: 0, 1 label. 1 if sepsis is predicted at the current timestep, 0 otherwise.
        """

        final_features = np.array(
            [self.process_history(x)]
        )
        _x = torch.tensor(final_features).float().to(self.config['device'])

        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(_x).squeeze(1)
            # print(y_pred)

        return y_pred.item() >= 0.5
    
    def validation(self, final_dataloader_val):

        self.model.eval()

        with torch.no_grad():
            loss_val = []
            for x, y_with_sepsis_flag in final_dataloader_val:

                y = y_with_sepsis_flag[:, 0]
                sepsis_flags = y_with_sepsis_flag[:, 1]
                sepsis_times = y_with_sepsis_flag[:, 2]
                current_times = y_with_sepsis_flag[:, 3]

                y_pred = self.model(x).squeeze(1)
                # y_pred = self.model(x).squeeze(1) * 0 # FOR UTILITY LOSS DEBUG ONLY
                # y_pred = self.model(x).squeeze(1) * 0 + 1 # FOR UTILITY LOSS DEBUG ONLY

                if isinstance(self.criteria, UtilityLoss):
                    loss = self.criteria(y_pred, 
                                         y, 
                                         sepsis_flags=sepsis_flags, 
                                         sepsis_times=sepsis_times,
                                         current_times=current_times,
                                         )
                    # print(loss)
                else:
                    loss = self.criteria(y_pred, y)

                loss_val.append(loss.item())

        if isinstance(self.criteria, UtilityLoss):
            return np.sum(np.array(loss_val))
        else:
            return np.mean(np.array(loss_val))
    
    def test_utility(self, ):
        from utils import UtilityFunction, RealOutcomesSimulator

        utility_fn = UtilityFunction()
        simulator_test = RealOutcomesSimulator(self.data['test'], utility_fn)

        utility = simulator_test.compute_utility(self)
        print(f" test - utility achieved is: {utility['u_total']:.4f}")
        # print(f"example prediction {utility['preds'][0]}")
        print(f"test - confusion matrix {utility['cm']}")

        return utility['u_total']
    
    def test_utility_fast(self, final_dataloader_test):

        self.model.eval()
        criteria = UtilityLoss()

        with torch.no_grad():
            loss_val = []
            for x, y_with_sepsis_flag in final_dataloader_test:

                y = y_with_sepsis_flag[:, 0]
                sepsis_flags = y_with_sepsis_flag[:, 1]
                sepsis_times = y_with_sepsis_flag[:, 2]
                current_times = y_with_sepsis_flag[:, 3]

                y_pred = self.model(x).squeeze(1)
                y_pred_binary = (y_pred >= 0.5)
                

                
                loss = criteria(y_pred, 
                                y_pred_binary.float(), 
                                sepsis_flags=sepsis_flags, 
                                sepsis_times=sepsis_times,
                                current_times=current_times,
                                )

                loss_val.append(loss.item())


        utility_loss = np.sum(np.array(loss_val))
        print(f" test (fast) - utility achieved is: {-utility_loss:.4f}")

        return utility_loss
    
    def val_utility_fast(self, final_dataloader_val):

        self.model.eval()
        criteria = UtilityLoss()

        with torch.no_grad():
            loss_val = []
            for x, y_with_sepsis_flag in final_dataloader_val:

                y = y_with_sepsis_flag[:, 0]
                sepsis_flags = y_with_sepsis_flag[:, 1]
                sepsis_times = y_with_sepsis_flag[:, 2]
                current_times = y_with_sepsis_flag[:, 3]

                y_pred = self.model(x).squeeze(1)
                y_pred_binary = (y_pred >= 0.5)

                
                loss = criteria(y_pred, 
                                y_pred_binary.float(), 
                                sepsis_flags=sepsis_flags, 
                                sepsis_times=sepsis_times,
                                current_times=current_times,
                                )

                loss_val.append(loss.item())

        utility_loss = np.sum(np.array(loss_val))
        print(f" val (fast) - utility achieved is: {-utility_loss:.4f}")

        return utility_loss

    def val_utility(self, ):
        from utils import UtilityFunction, RealOutcomesSimulator

        utility_fn = UtilityFunction()
        simulator_test = RealOutcomesSimulator(self.data['val'], utility_fn)

        utility = simulator_test.compute_utility(self)
        print(f" test - utility achieved is: {utility['u_total']:.4f}")
        # print(f"example prediction {utility['preds'][0]}")
        print(f"test - confusion matrix {utility['cm']}")

        return utility['u_total']

#####


class LRModel(BaseModel):
    def __init__(self, model=LogisticRegression()):
        super().__init__()
        self.model = model
    
    def preprocess(self, data):

        final_features = []
        final_labels = []
        for x, y in data:

            # x is a single patient's features over their entire history: (t, n_features)
            # y is a single patient's outcomes over their entire history: (t,)

            # walk over the patient's history and create a feature vector for each timestep
            for t in range(len(x)):
                history_up_to_timestep_t = x[:t+1]
                label_at_timestep_t = y[t]
                final_features.append(self.process_history(history_up_to_timestep_t))
                final_labels.append(label_at_timestep_t)

        self.X_train = np.array(final_features)
        self.y_train = np.array(final_labels)

        scaler = preprocessing.StandardScaler().fit(self.X_train)
        self.X_scaled = scaler.transform(self.X_train)

        return self.X_scaled, self.y_train

    def fit(self, data: PatientDataset) -> None:

        X_scaled, y_train = self.preprocess(data)
        self.model.fit(X_scaled, y_train)

    def process_history(self, x):
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: np.ndarray of shape (d,) representing a fixed-size feature vector.
        """
        # For the dummy model, we just return the last timestep's features
        return x[-1]

    def predict(self, x) -> float:
        """
        x: np.ndarray of shape (t, n_features) representing a patient's history at the current timestep.
        returns: 0, 1 label. 1 if sepsis is predicted at the current timestep, 0 otherwise.
        """
        return self.model.predict([self.process_history(x)])[0] > 0.5 # label
    