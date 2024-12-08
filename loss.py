import torch
import torch.nn as nn

from dataset import PatientDataset

class CustomBCELoss(nn.Module):
    def __init__(self, scale_factor=50):
        super(CustomBCELoss, self).__init__()
        self.scale_factor = scale_factor
        self.bce_loss = nn.BCELoss(reduction='none')  # Reduction is set to 'none' to handle scaling manually

    def forward(self, predictions, targets):
        # Compute the binary cross-entropy loss for each sample
        bce_loss = self.bce_loss(predictions, targets)
        # Scale the loss for datapoints where target == 1
        scaled_loss = torch.where(targets == 1, bce_loss * self.scale_factor, bce_loss)
        # Return the mean loss
        return scaled_loss.mean()


class UtilityLoss(nn.Module):
    '''
        This is the loss that take uitility function into account (L2).

        L1 is the Binary Cross-Entropy loss.
    '''
    def __init__(self,):
        super(UtilityLoss, self).__init__()

        self.dt_early = -12
        self.dt_optimal = -6
        self.dt_late = 3.0
        max_u_tp = 1
        min_u_fn = -2

        self.u_fp = -0.05
        self.u_tn = 0

        # Define slopes and intercept points for utility functions of the form
        # u = m * t + b.
        self.m_1 = float(max_u_tp) / float(self.dt_optimal - self.dt_early)
        self.b_1 = -self.m_1 * self.dt_early
        self.m_2 = float(-max_u_tp) / float(self.dt_late - self.dt_optimal)
        self.b_2 = -self.m_2 * self.dt_late
        self.m_3 = float(min_u_fn) / float(self.dt_late - self.dt_optimal)
        self.b_3 = -self.m_3 * self.dt_optimal

        return
    
    def forward(self, predictions, targets, 
                sepsis_flags=None, 
                sepsis_times=None, 
                current_times=None,
                ):
        assert len(predictions) == len(targets), "Number of predictions and labels do not match."

        # Compare predicted and true conditions.
        u_TP = torch.zeros(len(predictions))
        u_FP = torch.zeros(len(predictions))
        u_TN = torch.zeros(len(predictions))
        u_FN = torch.zeros(len(predictions))

        for patient_no in range(len(targets)):

            t = current_times[patient_no]
            is_septic = sepsis_flags[patient_no]
            t_sepsis = sepsis_times[patient_no] - self.dt_optimal



            if t <= t_sepsis + self.dt_late:

                ## in the case that is_septic == True
                # TP
                if is_septic:
                    if t <= t_sepsis + self.dt_optimal:
                        u_TP[patient_no] = max(self.m_1 * (t - t_sepsis) + self.b_1, self.u_fp)
                    elif t <= t_sepsis + self.dt_late:
                        u_TP[patient_no] = self.m_2 * (t - t_sepsis) + self.b_2

                    # FN
                    if t <= t_sepsis + self.dt_optimal:
                        u_FN[patient_no] = 0
                    elif t <= t_sepsis + self.dt_late:
                        u_FN[patient_no] = self.m_3 * (t - t_sepsis) + self.b_3
                
                ## in the case that is_septic == False
                else:
                    
                    # print('not septic patient')
                    u_TP[patient_no] = 0
                    u_FN[patient_no] = 0

                
                # FP
                u_FP[patient_no] = self.u_fp

                # TN
                u_TN[patient_no] = self.u_tn

        u_TP = u_TP.to(targets.device)
        u_FP = u_FP.to(targets.device)
        u_TN = u_TN.to(targets.device)
        u_FN = u_FN.to(targets.device)

        
        u_func = predictions * (u_TP*sepsis_flags + u_FP*(1-sepsis_flags)) \
            + (1-predictions) *  (u_FN*sepsis_flags + u_TN*(1-sepsis_flags))

        return torch.sum(-u_func)

class L1C1Loss(nn.BCELoss):
    '''
        BCE + non-weighted.
    '''
    def __init__(self,):
        return super().__init__()
    

class L2C1Loss(UtilityLoss):
    '''
        Utility + non-weighted.
    
    '''
    def __init__(self,):
        return super().__init__()
    

class L1C2Loss():
    '''
        BCE + weighted.
    '''
    def __init__(self, dataset:PatientDataset):
        return 