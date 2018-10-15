import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Loss function for AutoRec
# -----------------------------------------------------------------------------

class AutoRecLoss(torch.nn.Module):

    def __init__(self):
        super(AutoRecLoss,self).__init__()

    def forward(self,predicted_ratings, real_ratings):
        mask = torch.isnan(real_ratings).le(0)
        masked_real_ratings = torch.masked_select(real_ratings, mask)
        masked_pred_ratings = torch.masked_select(predicted_ratings, mask)

        # Keep commented out!
        # ratings_loss = torch.norm(masked_real_ratings - masked_pred_ratigns)

        mse_loss = F.mse_loss(masked_pred_ratings, masked_real_ratings, reduction='sum')
        #rmse = torch.sqrt(mse_loss/len(masked_real_ratings))


        # No need to add here the regularization factor given that we
        # can achieve the same result by passing a non-zero value to the
        # optimizer argument named weight_decay;
        # weights_regularization = (reg_strength/2)*torch.norm(weight)
        # return ratings_loss + weights_regularization

        return mse_loss

# -----------------------------------------------------------------------------
# Loss function playground
# -----------------------------------------------------------------------------
'''
a = torch.empty(3,3)
a[:] = torch.tensor(float('nan'))
a[1][1] = 3
a[1][2] = 3
mask = torch.isnan(a).le(0)

len(torch.masked_select(torch.ones(3,3), mask))


a = torch.tensor([[1,1,1,1]], dtype=torch.float)
W = torch.randn(4,3,requires_grad=True)
Q = torch.randn(3,4,requires_grad=True)
c = torch.tensor([[-1.,2.,-3.,4.]])

y = torch.mm(a,W)
res = torch.mm(y,Q)

mask = c.ge(0)

encoder_handle = add_grad_hook(zero_weights)

res_masked = torch.masked_select(res, mask)
c_masked = torch.masked_select(c, mask)


loss= F.mse_loss(res_masked,c_masked, reduction='sum')


loss.backward()

print(Q.grad)
print(W.grad)



def add_grad_hook(encoder_hook):
    encoder_handle = W.register_hook(encoder_hook)
    return encoder_handle


def zero_weights(grad):
    column_index = 0
    grad_clone = grad.clone()
    #for rating in autorec.state_dict().get("input"):
    for rating in c[0]:
        print('rating: {}'.format(rating))
        if rating < 0:
            grad_clone[column_index, :] = 0
        column_index += 1
    #print(grad_clone)
    return grad_clone
'''
