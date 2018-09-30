import torch


# -----------------------------------------------------------------------------
# Loss function for AutoRec
# -----------------------------------------------------------------------------

class AutoRec_Loss(torch.nn.Module):

    def __init__(self):
        super(AutoRec_Loss,self).__init__()

    def forward(self,predicted_ratings, real_ratings, weights, reg_strength):
        mask = real_ratings.ge(0)
        masked_real_ratings = torch.masked_select(real_ratings, mask)
        masked_pred_ratings = torch.masked_select(predicted_ratings, mask)
        ratings_loss = torch.norm(maked_real_ratings - masked_pred_ratigns)
        weights_regularization = (reg_strength/2)*torch.norm(weight)
        return ratings_loss + weights_regularization
