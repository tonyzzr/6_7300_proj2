from __future__ import annotations
from typing import List, Tuple
import numpy as np
import os
from tqdm import tqdm

from dataset import PatientDataset
from model import BaseModel, PerfectModel



#@title class UtilityFunction():
class UtilityFunction():
    def __init__(self) -> None:
        pass

    def get_utility(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """
        Returns the utility of a method's predictions on a single patient.

        predictions: np.ndarray with 0,1 predictions per timestep (n_timesteps,)
        labels: np.ndarray with 0,1 labels per timestep (n_timesteps,)
        returns: utility value
        """
        dt_early = -12
        dt_optimal = -6
        dt_late = 3.0
        max_u_tp = 1
        min_u_fn = -2
        u_fp = -0.05
        u_tn = 0

        assert len(predictions) == len(labels), "Number of predictions and labels do not match."

        # Does the patient eventually have sepsis?
        if np.any(labels):
            is_septic = True
            t_sepsis = np.argmax(labels) - dt_optimal
        else:
            is_septic = False
            t_sepsis = float('inf')

        n = len(labels)

        # Define slopes and intercept points for utility functions of the form
        # u = m * t + b.
        m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
        b_1 = -m_1 * dt_early
        m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
        b_2 = -m_2 * dt_late
        m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
        b_3 = -m_3 * dt_optimal

        # Compare predicted and true conditions.
        u = np.zeros(n)
        for t in range(n):
            if t <= t_sepsis + dt_late:
                # TP
                if is_septic and predictions[t]:
                    if t <= t_sepsis + dt_optimal:
                        u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                    elif t <= t_sepsis + dt_late:
                        u[t] = m_2 * (t - t_sepsis) + b_2
                # FP
                elif not is_septic and predictions[t]:
                    u[t] = u_fp
                # FN
                elif is_septic and not predictions[t]:
                    if t <= t_sepsis + dt_optimal:
                        u[t] = 0
                    elif t <= t_sepsis + dt_late:
                        u[t] = m_3 * (t - t_sepsis) + b_3
                # TN
                elif not is_septic and not predictions[t]:
                    u[t] = u_tn

        # Find total utility for patient.
        return np.sum(u).item()
    

#@title class RealOutcomesSimulator():

class RealOutcomesSimulator():
    def __init__(
        self,
        dataset: PatientDataset,
        utility_fn: UtilityFunction,
    ) -> None:
        self.dataset = dataset
        self.utility_fn = utility_fn

    def compute_utility(self, model: BaseModel) -> float:
        """
        Simulates the real outcomes of the patients in the dataset using decisions
        made by the model and returns the total utility.
        """
        utility = {
            'u_total': 0.0,
            # TODO: Feel free to edit this function and add more keys that might help you debug your model's performance!
            'preds':[],
            'cm':{
                'TP':0, 'TN':0, 'FP':0, 'FN':0,
            }
        }



        # for each patient
        for i in tqdm(range(len(self.dataset))):
            
            x, y = self.dataset[i]
            preds = []
            for t in range(len(x)):

                if isinstance(model, PerfectModel):
                    y_pred = y[t]
                else:
                    y_pred = model.predict(x[:t+1])

                preds.append(y_pred)

                _cm = confusion_matrix(y_pred, y[t])
                for key in _cm:
                    utility['cm'][key] += _cm[key]
            

            preds = np.array(preds)
            utility['preds'].append(preds)

            u = self.utility_fn.get_utility(preds, y)
            utility['u_total'] += u

        return utility
    
def confusion_matrix(y_pred_binary, y_binary):

    cm_epoch = {
                'TP':0, 'TN':0, 'FP':0, 'FN':0,
            }
    
    TP = np.sum((y_binary==1) & (y_pred_binary == 1))
    TN = np.sum((y_binary==0) & (y_pred_binary == 0))

    FP = np.sum((y_binary==0) & (y_pred_binary == 1))
    FN = np.sum((y_binary==1) & (y_pred_binary == 0))


    cm_epoch['TP'] += TP
    cm_epoch['TN'] += TN
    cm_epoch['FP'] += FP
    cm_epoch['FN'] += FN

    return cm_epoch