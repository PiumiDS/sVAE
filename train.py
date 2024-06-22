import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from pandas.core.common import flatten
from skimage.transform import resize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import scipy.io
import random
import glob
from tqdm import tqdm
import sys, os, shutil, time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torchmetrics
from torchmetrics.classification import Accuracy
# from torch.optim import Adam

plt.rcParams.update({'font.size': 16})
torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print_flag = False
resume_flag = True

# create a transofrm to apply to each datapoint
train_transform = A.Compose([A.CenterCrop(height=48, width=48),
    # A.augmentations.transforms.ToFloat(max_value=255.0),
    ToTensorV2()])
test_transform = A.Compose([A.CenterCrop(height=48, width=48),
    # A.augmentations.transforms.ToFloat(max_value=255.0),
    ToTensorV2()])

batch_size = 128
lr = 0.00002
epochs = 1000
start_epoch = 0
kernel_size = 3
stride = 1
padding = 0
num_channels = 9
init_kernel = 16
latent_dim = 64
sub_latent_dims = [1152, 256, 128, 64]
num_classes = 6
kld_beta = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device status: ', device)

multi_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

# for NCI file handle
full_data_path = '/jobfs/100908406.gadi-pbs/' 
if not os.path.exists(f"output"): os.mkdir(f"output")
resume_path = f"output/last_checkpoint.pt"

train_image_paths = []
valid_image_paths = []
test_image_paths = []

#load all train data paths
all_npy = np.load('paths/image_paths_TRAIN.npy')
train_images_npy = all_npy[:int(len(all_npy)*0.85)]
valid_images_npy = all_npy[int(len(all_npy)*0.85):]

test_images_npy  = np.load('paths/image_paths_TEST.npy')
paths = np.concatenate((np.load('paths/image_paths_TRAIN.npy'), np.load('paths/image_paths_TEST.npy')))
labels = np.array([int(n.split('/')[-1].split('_')[-3])-1 for n in paths])

train_image_paths = list(flatten(train_images_npy))
random.shuffle(train_image_paths)
valid_image_paths = list(flatten(valid_images_npy))
random.shuffle(valid_image_paths)
test_image_paths = list(flatten(test_images_npy))
random.shuffle(test_image_paths)

# print('train_image_path example: ', train_image_paths[0])
# print('classes: ', classes)

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

def remove_inf_nan(x):
    x[x!=x]=0
    x[~torch.isfinite(x)]=0
    return x

class TMEDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = os.path.join(full_data_path, os.path.basename(self.image_paths[idx]))
        # read mIF image patches
        image = tifffile.imread(image_filepath)[:,:,:9]
        index = np.where(paths==self.image_paths[idx])
        # read label and one hot encode
        label = labels[index]
        one_hot_encoded = np.zeros(num_classes, dtype=int)
        one_hot_encoded[label] = 1
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = remove_inf_nan(image) # remove inf nan if present
        return image, one_hot_encoded, self.image_paths[idx]

train_dataset = TMEDataset(train_image_paths,train_transform)
valid_dataset = TMEDataset(valid_image_paths,test_transform)
test_dataset = TMEDataset(test_image_paths,test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# define a Conv VAE with encoder z' 
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        ############## encoder ################
        self.enc1 = nn.Conv2d(
            in_channels=num_channels, out_channels=init_kernel, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc1_bn = nn.BatchNorm2d(init_kernel)  # Added BatchNorm
        self.enc2 = nn.Conv2d(
            in_channels=init_kernel, out_channels=init_kernel*2, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc2_bn = nn.BatchNorm2d(init_kernel*2)
        self.enc3 = nn.Conv2d(
            in_channels=init_kernel*2, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc3_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc4 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc4_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc5 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc5_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc6 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc6_bn = nn.BatchNorm2d(init_kernel*4)  # Added BatchNorm
        self.enc7 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc7_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc8 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc8_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc9 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc9_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc10 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc10_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc11 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc11_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc12 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc12_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc13 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc13_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc14 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc14_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc15 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc15_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc16 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc16_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc17 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc17_bn = nn.BatchNorm2d(init_kernel*4)
        self.enc18 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=latent_dim, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )



        ############### decoder ###############
        self.dec1 = nn.ConvTranspose2d(
            in_channels=latent_dim, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec1_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec2_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec3_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec4_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec5_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec6 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec6_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec7 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec7_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec8 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec8_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec9 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec9_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec10 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec10_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec11 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec11_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec12 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec12_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec13 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec13_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec14 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec14_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec15 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec15_bn = nn.BatchNorm2d(init_kernel*4)
        self.dec16 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*2, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec16_bn = nn.BatchNorm2d(init_kernel*2)
        self.dec17 = nn.ConvTranspose2d(
            in_channels=init_kernel*2, out_channels=init_kernel, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dec17_bn = nn.BatchNorm2d(init_kernel)
        self.dec18 = nn.ConvTranspose2d(
            in_channels=init_kernel, out_channels=num_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )

        ############### classfier ###############
        self.lin1 = nn.Linear(sub_latent_dims[0], sub_latent_dims[1])
        self.lin1_bn = nn.BatchNorm1d(sub_latent_dims[1])
        self.lin2 = nn.Linear(sub_latent_dims[1], sub_latent_dims[2])
        self.lin2_bn = nn.BatchNorm1d(sub_latent_dims[2])
        self.lin3 = nn.Linear(sub_latent_dims[2], sub_latent_dims[3])
        self.lin3_bn = nn.BatchNorm1d(sub_latent_dims[3])
        self.lin4 = nn.Linear(sub_latent_dims[3], num_classes)



    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.leaky_relu(self.enc1_bn(self.enc1(x)))
        x = F.leaky_relu(self.enc2_bn(self.enc2(x)))
        x = F.leaky_relu(self.enc3_bn(self.enc3(x)))
        x = F.leaky_relu(self.enc4_bn(self.enc4(x)))
        x = F.leaky_relu(self.enc5_bn(self.enc5(x)))
        x = F.leaky_relu(self.enc6_bn(self.enc6(x)))
        x = F.leaky_relu(self.enc7_bn(self.enc7(x)))
        x = F.leaky_relu(self.enc8_bn(self.enc8(x)))
        x = F.leaky_relu(self.enc9_bn(self.enc9(x)))
        x = F.leaky_relu(self.enc10_bn(self.enc10(x)))
        x = F.leaky_relu(self.enc11_bn(self.enc11(x)))
        x = F.leaky_relu(self.enc12_bn(self.enc12(x)))
        x = F.leaky_relu(self.enc13_bn(self.enc13(x)))
        x = F.leaky_relu(self.enc14_bn(self.enc14(x)))
        x = F.leaky_relu(self.enc15_bn(self.enc15(x)))
        x = F.leaky_relu(self.enc16_bn(self.enc16(x)))
        x = F.leaky_relu(self.enc17_bn(self.enc17(x)))
        x = self.enc18(x)

        # get `mu` and `log_var`
        mu = x
        log_var = x

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z_flatten = torch.flatten(z, start_dim = 1)

        #select z' from z and use in classifier
        zdash = z_flatten[:, :sub_latent_dims[0]]
        y = F.leaky_relu(self.lin1_bn(self.lin1(zdash)))
        y = F.leaky_relu(self.lin2_bn(self.lin2(y)))
        y = F.leaky_relu(self.lin3_bn(self.lin3(y)))
        y = self.lin4(y)
        out_label = y
 
        # decoding
        x = F.leaky_relu(self.dec1_bn(self.dec1(z)))
        x = F.leaky_relu(self.dec2_bn(self.dec2(x)))
        x = F.leaky_relu(self.dec3_bn(self.dec3(x)))
        x = F.leaky_relu(self.dec4_bn(self.dec4(x)))
        x = F.leaky_relu(self.dec5_bn(self.dec5(x)))
        x = F.leaky_relu(self.dec6_bn(self.dec6(x)))
        x = F.leaky_relu(self.dec7_bn(self.dec7(x)))
        x = F.leaky_relu(self.dec8_bn(self.dec8(x)))
        x = F.leaky_relu(self.dec9_bn(self.dec9(x)))
        x = F.leaky_relu(self.dec10_bn(self.dec10(x)))
        x = F.leaky_relu(self.dec11_bn(self.dec11(x)))
        x = F.leaky_relu(self.dec12_bn(self.dec12(x)))
        x = F.leaky_relu(self.dec13_bn(self.dec13(x)))
        x = F.leaky_relu(self.dec14_bn(self.dec14(x)))
        x = F.leaky_relu(self.dec15_bn(self.dec15(x)))
        x = F.leaky_relu(self.dec16_bn(self.dec16(x)))
        x = F.leaky_relu(self.dec17_bn(self.dec17(x)))
        x = self.dec18(x)

        reconstruction = torch.sigmoid(x)
        return reconstruction, out_label, mu, log_var

def final_loss(bce_loss, y_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    :param y_loss: classification loss 
    """
    BCE = bce_loss
    LBL = y_loss 
    KLD = -kld_beta * 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(BCE.item(),LBL.item(), KLD.item())
    return BCE + LBL + KLD

def one_hot_ce_loss(outputs, targets):
    _, labels = torch.max(targets, dim=1)
    return criterion_lbl(outputs, labels)

def fit(model, dataloader):
    """
    The fucntion for training phase run per epoch
    :param model: semi-supervised VAE model
    :param dataloader: the training data loader
    Returns the loss and accuracy metrics
    """
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for i, (data, data_label,_) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        data = data
        data = data.to(device)
        data = data
        data_label = data_label.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        reconstruction, model_label, mu, logvar = model(data)
        # out = out.clamp(0, 1)
        bce_loss = criterion_img(reconstruction, data)
        lbl_loss = one_hot_ce_loss(model_label, data_label)
        loss = final_loss(bce_loss, lbl_loss, mu, logvar)
        loss.backward()
        running_accuracy += multi_acc(model_label, torch.argmax(data_label, dim=1))*data.shape[0]
        # print(multi_acc(model_label, torch.argmax(data_label, dim=1)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    train_label_accuracy = running_accuracy/len(dataloader.dataset)
    return train_loss, train_label_accuracy

def validate(model, dataloader):
    """
    The fucntion for validation and testing phase run per epoch
    :param model: trained semi-supervised VAE model
    :param dataloader: the validayion/ test data loader
    Returns the loss, accuracy metrics, true and predicted labels, and image list
    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    y_true = []
    y_pred = []
    im_list = []
    with torch.no_grad():
        for i, (data, data_label, data_path) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
            data = data
            data = data.to(device)
            data = data
            data_label = data_label.type(torch.FloatTensor).to(device)
            reconstruction, model_label, mu, logvar = model(data)
            bce_loss = criterion_img(reconstruction, data)
            lbl_loss = one_hot_ce_loss(model_label, data_label)
            loss = final_loss(bce_loss, lbl_loss, mu, logvar)
            # print(loss.item())
            running_loss += loss.item()
            running_accuracy += multi_acc(model_label, torch.argmax(data_label, dim=1))*data.shape[0]
            y_true.append(np.array(torch.argmax(data_label, dim=1).cpu()))
            y_pred.append(np.array(torch.argmax(model_label, dim=1).cpu()))
            im_list.append(np.array(data_path))
            # save the last batch input and output of every epoch
            if (i==int(len(valid_dataset)/dataloader.batch_size)-1):
                image_raw = np.moveaxis(np.hstack(np.array(data[:5].cpu())),0,-1)
                image_out = np.moveaxis(np.hstack(np.array(reconstruction[:5].cpu())),0,-1)
                combined_image = np.swapaxes(np.concatenate((image_raw,image_out), axis =1), 0, 1)
                # save_image(torch.cat((data[:8],reconstruction[:8])).cpu(), f"outputs/output{epoch}.png", nrow=8)
                # image write
                tifffile.imwrite(f"output/output{epoch}.tif", combined_image,  planarconfig='contig')


    val_loss = running_loss/len(dataloader.dataset)
    val_label_accuracy = running_accuracy/len(dataloader.dataset)
    return val_loss, val_label_accuracy, y_true, y_pred, im_list

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(best_model_dir, 'best_model.pt')
        torch.save(state, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_img = nn.BCELoss(reduction='sum')
criterion_lbl = nn.CrossEntropyLoss()

print(model)

#if resuming from previous checkpoints , load weights and optimizer 
if resume_flag:
    model, _, start_epoch = load_ckp(resume_path, model, optimizer)
    epochs = start_epoch + epochs
    print('resuming from epoch ', start_epoch)

train_loss = []
val_loss = []

min_train_loss = float('inf')
min_valid_loss = float('inf')
is_best = False

for epoch in range(start_epoch, epochs):
    """
    Runs for a given number of epochs unless stopping criterios is true
    The code below has been changed to allow the test phase analysis of reuslts. 
    If training from scratch you will need to define the stopping criterion again.

    """
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_accuracy          = fit(model, train_loader)
    val_epoch_loss, valid_accuracy, _, _, _   = validate(model, valid_loader)
    _, test_accuracy, y_true, y_pred, im_list = validate(model, test_loader)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Loss Train : {train_epoch_loss:.4f}", f" Validation : {val_epoch_loss:.4f}")
    print(f"Val Accuracy: {valid_accuracy:.4f}", f"Test Accuracy: {test_accuracy:.4f}")
    # print(accuracy_score(y_true,y_pred),\
    #     f1_score(y_true, y_pred, average="weighted"),\
    #     precision_score(y_true, y_pred, average="weighted"),\
    #     recall_score(y_true, y_pred, average="weighted"),\
    #     )
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    y_paths = np.concatenate(im_list).flatten()

    print(classification_report(y_true, y_pred,digits=4))
    
    sn.set(font_scale=2.5)
    df_cm = pd.DataFrame(confusion_matrix(y_true,y_pred), index = ['Tumour', 'iCAFs', 'myCAFs', 'T-cells', 'dPVLs', 'Exh. T-cells'],
                  columns = ['Tumour', 'iCAFs', 'myCAFs', 'T-cells', 'dPVLs', 'Exh. T-cells'])
    plt.figure(figsize = (28,24))
    plt.title('Cell type classification confusion matrix')
    sn.heatmap(df_cm, annot=False)
    plt.ylabel('True label (y)')
    plt.xlabel(r'$Predicted label (y_{\gamma})$')
    loc_, lblx_ = plt.xticks()
    plt.setp(lblx_, rotation=45)
    loc_, lbly_ = plt.yticks()
    plt.setp(lbly_, rotation=45)
    plt.savefig('heatmap.png')

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if (val_epoch_loss < min_valid_loss) and (train_epoch_loss < min_train_loss): 
        is_best =True
    save_ckp(checkpoint, is_best, f"output", f"output")
    
    if val_epoch_loss < min_valid_loss:
        min_valid_loss = val_epoch_loss
    if train_epoch_loss < min_train_loss:
        min_train_loss = train_epoch_loss
    is_best = False


# print('test results')
# test_loss = validate(model, test_loader)
# print(f"Test Loss: {test_loss:.4f}")


