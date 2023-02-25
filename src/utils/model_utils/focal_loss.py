import torch
import torch.nn as nn


class SigmoidFocalLoss(nn.Module):
    """
    The focal loss for fighting against class-imbalance
    """

    def __init__(self, alpha=1, gamma=1.5, sigmoid=True):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1e-5  # prevent training from Nan-loss error
        self.sigmoid = sigmoid

    def forward(self, preds, target):
        """
        logits & target should be tensors with shape [batch_size, num_classes]
        """
        if self.sigmoid:
            probs = torch.sigmoid(preds)
        else:
            probs = preds

        probs_t = torch.where(target == 1, probs, 1 - probs)
        alpha_t = torch.where(target == 1, self.alpha, 1)
        focal_loss = - alpha_t * (1 - probs_t) ** self.gamma * torch.log(probs_t + self.epsilon)
        return focal_loss.mean()
