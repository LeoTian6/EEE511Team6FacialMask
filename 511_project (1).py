from pathlib import Path 
from tqdm import tqdm 
import cv2 
import numpy as np 
import torch 
import os
import torch.nn as nn
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import Compose, ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
batch_size = 32
classes = ['Non-Masked', 'Masked']


# In[2]:


torch.cuda.empty_cache()


# In[3]:


data_path = Path(r'C:/Users/world/Desktop/511Finalpjt/data_set')
maskPath = data_path/'with_mask'
nonMaskPath = data_path/'without_mask' 
path_dirs = [ [maskPath,1],[nonMaskPath,0] ] #path and label


# In[4]:


class MaskvNoMask():
    LABELS = {'NON_MASKED': 0, 'MASKED': 0}
    training_data = []
    def make_training_data(self):
        for data_dir,label in path_dirs:
            for folder in tqdm(list(data_dir.iterdir())):
                folder_path = os.path.join(data_dir,folder)
                #try:
                #print(folder_path)
                img = cv2.imread(folder_path)
                img = cv2.resize(img, (100,100))
                self.training_data.append([np.array(img), label])
                #except :
                        #print(folder_path)

                if label == 0:
                    self.LABELS['NON_MASKED'] +=1
                if label == 1:
                    self.LABELS['MASKED'] += 1

        print(self.LABELS)
        np.random.shuffle(self.training_data)


# In[5]:


maskvnomask = MaskvNoMask()
maskvnomask.make_training_data()     
training_data = maskvnomask.training_data   


# In[6]:


class MaskDataset(Dataset):
        """ Masked faces dataset        0 = 'no mask'     1 = 'mask'       """
        def __init__(self, train_data):
            self.train_data = train_data
            self.transformations = Compose([
                ToTensor()        ])
        
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise NotImplementedError('slicing is not supported')                    
            return [
                self.transformations(self.train_data[key][0]),
                torch.tensor(self.train_data[key][1]) # pylint: disable=not-callable
            ]
        
        def __len__(self):
            return len(self.train_data)


# In[7]:


myDataset = MaskDataset(training_data)
myDataset[5]


# In[8]:


val_size = 1000
train_size = len(myDataset) - val_size

train_ds, val_ds = torch.utils.data.random_split(myDataset, [train_size, val_size])
len(train_ds), len(val_ds)


# In[9]:


img, label = myDataset[1001]
print(img.shape)
print(label)


# In[10]:


def show_example(data):
    img, label = data
    print('Label: ', classes[int(label.item())], "("+str(label.item())+")")
    plt.imshow(img.permute(1, 2, 0))
show_example(val_ds[77])


# In[14]:


count=0
for _,label in val_ds:
    if label==1:
        count+=1
count


# In[11]:


train_dl = DataLoader(train_ds, batch_size*2,shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2,shuffle=True)


# In[12]:


def show_batch(dl):
    for images, labels in dl:
        print(labels)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
show_batch(train_dl)


# In[13]:


show_batch(val_dl)


# In[3]:


def accuracy(outputs, labels):
    #_, preds = torch.max(outputs, dim=1)
    for i in range(len(outputs)):
        if outputs[i]>=0.5:
            outputs[i]=1
        else: outputs[i]=0
    return torch.tensor(torch.sum(outputs == labels).item() / len(outputs))       
    #return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[4]:


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)# Generate predictions
        loss = F.binary_cross_entropy(torch.squeeze(torch.sigmoid(out)), labels.float()) # Calculate loss
        #loss=F.cross_entropy(out, labels.long()) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.binary_cross_entropy(torch.squeeze(torch.sigmoid(out)), labels.float())
        #loss = F.cross_entropy(out, labels.long()) # Calculate loss
        #acc = accuracy(out, labels)
        acc = accuracy(torch.squeeze(torch.sigmoid(out)), labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[5]:


class MaskDetection(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(160000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        
    def forward(self, xb):
        return self.network(xb)


# In[17]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
device


# In[18]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[19]:


for images, labels in val_dl:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    print(labels.shape)
    break


# In[20]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[21]:


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)


# In[22]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0.001)
    for epoch in range(epochs):
        # Training Phase 
        print('epoch: ', epoch)
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[23]:


# Model (on GPU)
model = MaskDetection()
to_device(model, device)


# In[24]:


model.eval()


# In[25]:


sum(p.numel() for p in model.parameters())


# In[26]:


[evaluate(model, val_dl)]


# In[27]:


history = fit(10, 1e-3, model, train_dl, val_dl)


# In[8]:


torch.save(model, r'C:\Users\world\Desktop\511Finalpjt\detection\.pt')


# In[6]:


model=torch.load(r'C:\Users\world\Desktop\511Finalpjt\detection\.pth')


# In[7]:


model.cpu()


# In[42]:


def singleImage(path, label= None):
    img = cv2.imread(path)
    assert img is not None,"Immage wasn't read properly"
    face_cascade = cv2.CascadeClassifier(r"C:\Users\world\Desktop\511Finalpjt\haarcascade_frontalface_alt.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (i, (x, y, w, h)) in enumerate(faces): 
        cv2.rectangle(img, (x, y), (x+1*w, y+1*h), (220, 90, 230), 3)      
        img=img[y:y+h,x:x+w]
        plt.imshow(img)
    img = cv2.resize(img, (100, 100))
    img = torch.from_numpy(img)
    img = img.permute((2, 0,1)) # model expects image to be of shape [3, 100, 100]
    img = img.unsqueeze(dim=0).float() # convert single image to batch [1, 3, 100, 100]
    img = img.to('cuda') # Using the same device as the model
    pred = model(img)
    pred=1 / (1 + np.exp(-pred.item()))
    print(classes[pred.astype(int)])
    print(np.round(pred,1))


# In[43]:


singleImage(r"C:\Users\world\Desktop\511Finalpjt\detection\231.jpg")



# In[32]:


face_clsfr=cv2.CascadeClassifier(r"C:\New folder\haarcascade_frontalface_default.xml")
#Opening the video webcam of your PC
source=cv2.VideoCapture(1)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while(True):

    #Reading the Images from the live videocam
    ret,img=source.read()
    
    #To extract the Region of Interest(ROI) from the extracted image
    faces=face_clsfr.detectMultiScale(img)  

    for (x,y,w,h) in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        img1=torch.from_numpy(resized)
        img1=img1.permute((2,0,1))
        img1=img1.unsqueeze(dim=0).float()
        pred=model(img1)
        label=np.round(1 / (1 + np.exp(-pred.item())))
        #label=labels_dict[pred]
        #Giving the Rectangle color block for the face image
        cv2.rectangle(img,(x,y),(x+w,y+h),(220,90,230),2)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
     
    
    cv2.imshow('Face Mask Detector- ANC Final',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()