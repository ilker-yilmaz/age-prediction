import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# EDA

image_dir = Path('C:/Users/ilker/Desktop/son_gelenler_2022/age-dataset/age_prediction_up/age_prediction/train') #tell python in which directory the training images are.
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
images.head()

def visualize(image):
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(image)

#choose an image id to see its age
image_id = 35
image_example_path = images.iloc[image_id].Filepath
age = images.iloc[image_id].Age

print(f'Age: {age}')
image = cv2.imread(image_example_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image)

sns.countplot(images.Age)
plt.xticks(rotation = 45)

images.Age = images.Age-20

# Prepare Data

from sklearn.model_selection import train_test_split
train, test = train_test_split(images, test_size = 0.1, random_state = 1)
train, valid = train_test_split(train, test_size = 0.2, random_state = 1)

# Dataset class

class ImageAgeDataset() :
  def __init__(self , images) :
    self.age = images.Age.values
    self.paths = images.Filepath.values

  def __len__(self) :
    return len(self.age)

  def __getitem__(self , item) :
    out = dict()
    path = self.paths[item]
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image , dtype = float)
    out['x'] = torch.tensor(image , dtype = torch.float).permute(2,0,1)/255
    out['y'] = torch.tensor(self.age[item], dtype= torch.long) #CrossEntropyLoss()'s forward(probs, y) function requires torch.long for y
    return out


train_dataset = ImageAgeDataset(train)
valid_dataset = ImageAgeDataset(valid)

# Model
class AlexNetwork(nn.Module):
  def __init__(self, n_classes):
    super(AlexNetwork, self).__init__()
    self.n_classes = n_classes
    self.conv_1 = nn.Conv2d(
      in_channels=3,
      out_channels=96,
      kernel_size=11,
      stride=4,
      padding=50
    )
    self.pool_1 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
    )
    self.conv_2 = nn.Conv2d(
      in_channels=96,
      out_channels=256,
      kernel_size=5,
      stride=1,
      padding=2
    )
    self.pool_2 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
    )
    self.conv_3 = nn.Conv2d(
      in_channels=256,
      out_channels=384,
      kernel_size=3,
      stride=1,
      padding=1
    )
    self.conv_4 = nn.Conv2d(
      in_channels=384,
      out_channels=384,
      kernel_size=3,
      stride=1,
      padding=1
    )
    self.conv_5 = nn.Conv2d(
      in_channels=384,
      out_channels=256,
      kernel_size=3,
      stride=1,
      padding=1
    )
    self.pool_3 = nn.MaxPool2d(
      kernel_size=3,
      stride=2,
    )
    self.nn = nn.Sequential(nn.Linear(in_features=9216, out_features=4096),
                            nn.ReLU(),
                            nn.Linear(in_features=4096, out_features=4096),
                            nn.ReLU(),
                            nn.Linear(in_features=4096, out_features=self.n_classes),
                            )

  def forward(self, x):
    x = self.conv_1(x)
    x = self.pool_1(x)
    x = self.conv_2(x)
    x = self.pool_2(x)
    x = self.conv_3(x)
    x = self.conv_4(x)
    x = self.conv_5(x)
    x = self.pool_3(x)
    x = x.view(-1, 9216)
    x = self.nn(x)
    return x

# Engine

#hparams
#DEVICE = 'cuda'
DEVICE = torch.device("cpu")
BATCH_SIZE = 50
MAX_WAIT = 3
EPOCHS = 100
LR = 1e1

#---------------------
train_dataloader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True ,
    num_workers = 8
)

valid_dataloader = torch.utils.data.DataLoader(
    dataset = valid_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 8
)
#---------------------
model = AlexNetwork(31)
model.to(DEVICE)
#---------------------
loss_fn = nn.CrossEntropyLoss()
#---------------------
optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
#---------------------
from tqdm.notebook import tqdm
import numpy as np
def train_step(model , optimizer , dataloader):
  model.train()
  total_loss = 0
  for data in tqdm(dataloader):
    x = data["x"].to(DEVICE)
    y = data["y"].to(DEVICE)
    y_hat = model(x).to(DEVICE) #to apply model(x), x and model must be in the same place (device)
    loss = loss_fn(y_hat , y) #enough for y_hat and y to be in the same device
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    total_loss += loss.item()
  #total_loss.backward()
  #optimizer.step()
  #optimizer.zero_grad()
  return total_loss/len(dataloader)



def valid_step(model, dataloader):
  model.eval()
  total_loss = 0
  for data in tqdm(dataloader):
    x = data["x"].to(DEVICE)
    y = data["y"].to(DEVICE)
    with torch.no_grad():
      y_hat = model(x).to(DEVICE)
      loss = loss_fn(y_hat , y)
    total_loss += loss.item()
  return total_loss/len(dataloader)

def main(model , optimizer , train_dataloader , valid_dataloader ):
  train_losses = []
  valid_losses = []
  min_valid_loss = np.inf
  patience = 0
  for epoch in range(EPOCHS):
    train_loss = train_step(model , optimizer , train_dataloader)
    train_losses.append(train_loss)
    valid_loss = valid_step(model , valid_dataloader)
    valid_losses.append(valid_loss)
    print(f'EPOCH = {epoch}')
    print(f'train_loss = {train_loss}')
    print(f'valid_loss = {valid_loss}')
    if valid_loss > min_valid_loss :
      patience += 1
    else:
      min_valid_loss = valid_loss
      patience = 0

    if patience > MAX_WAIT :
      print(f'EARLY STOPPING AT EPOCH = {epoch}')
      break
  return train_losses , valid_losses


train_losses, valid_losses = main(
    model = model,
    optimizer = optimizer,
    train_dataloader = train_dataloader,
    valid_dataloader = valid_dataloader
)
#
# test_dataset = ImageAgeDataset(test)
#
# len(test_dataset)
#
# #play around.. choose an id from 0 to 4043 to test the model on :)
# id = 0
# softmax = nn.Softmax()
# y_hat_id = softmax(model(test_dataset[id]['x'].to(DEVICE)))
# def pred(x):
#     for i in range(len(x[0])):
#         if (x[0][i] == torch.max(x[0]).item()):
#             return i
# true_age = train_dataset[id]['y'].item() + 20
# print(f'image {id} is {true_age} years old and is predicted {pred(y_hat_id)+20} years old')
# image = cv2.imread(test.iloc[id].Filepath)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# visualize(image)