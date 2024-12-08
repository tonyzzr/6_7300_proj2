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


class UtilityLoss():
    '''
        This is the loss that take uitility function into account (L2).

        L1 is the Binary Cross-Entropy loss.
    '''
    def __init__(self, dataset:PatientDataset):
        return

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