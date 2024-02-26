from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preprocess import prepare
from utilities import train


data_dir = '/Users/tylerklimas/Desktop/LiverSegmentation/task03_liver'
# model_dir = 'D:/Youtube/Organ and Tumor Segmentation/results/results' 
data_in = prepare(data_dir, cache=False)


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # Use CPU
    print("CUDA is not available. Using CPU.")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(4, 4, 4, 4, 4), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 5)