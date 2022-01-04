import os
import torch
import time
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image
import random
from nn_models.Discriminator import Discriminator
from nn_models.Generator import Generator
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

NUM_GPU = 1
BATCH_SIZE = 8
NUM_EPOCH = 1000
LEARNING_RATE = 0.00005
IMG_SIZE = 96
IMG_CHANNEL = 3
VECTOR_LENGTH = 100
CLIP_VALUE = 0.01
NUM_CRITIC = 1
MODELS_DIR=os.path.join("./checkpoint")
OUT_DIR = os.path.join("./output")
IMG_SAVE_FREQ = 1
LOAD_MODELS = False

if not(os.path.exists(MODELS_DIR)):
    os.makedirs(MODELS_DIR)

if not(os.path.exists(OUT_DIR)):
    os.makedirs(OUT_DIR)

class Trainer():
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.dataset = datasets.ImageFolder('data',transform=self.data_transforms)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )


        self.D = Discriminator(NUM_GPU,IMG_CHANNEL,IMG_SIZE).to(self.device)
        self.G = Generator(NUM_GPU,VECTOR_LENGTH,IMG_CHANNEL,IMG_SIZE).to(self.device)
        if(LOAD_MODELS):
            self.load_models()
        else:
            self.D.apply(self.weights_init)
            self.G.apply(self.weights_init)
        self.D_optim = torch.optim.RMSprop(self.D.parameters(),LEARNING_RATE)
        self.G_optim = torch.optim.RMSprop(self.G.parameters(),LEARNING_RATE)
        self.lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.D_optim, T_max=5, eta_min=5E-5)
        self.lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.G_optim, T_max=5, eta_min=5E-5)


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self,):
        for eid in range(NUM_EPOCH):
            self.run_epoch(eid)
            self.save_models()

    def run_epoch(self,eid):
        self.num_batch = len(self.dataloader)
        for bid, (real_inputs,_) in enumerate(self.dataloader):
            self.run_batch(real_inputs.to(self.device),eid,bid)
        self.lrg_scheduler.step()
    


    def run_batch(self,real_inputs,eid,bid):
        before_op_time = time.time()


        if(bid%NUM_CRITIC) == 0 :
            D_loss = self.train_discriminator(real_inputs)
        else:
            D_loss = 0.0

        G_loss = self.train_generator()

        if((bid+1)%int(self.num_batch/IMG_SAVE_FREQ) == 0):
            noise = torch.randn(BATCH_SIZE,VECTOR_LENGTH,1,1).to(self.device)
            self.save_image(self.G(noise.unsqueeze(0)).squeeze(0).detach().cpu(),os.path.join(OUT_DIR,"epoch_{}_batch_{}.jpg".format(eid+1,bid+1)))
        
        duration = time.time() - before_op_time
        samples_per_sec = BATCH_SIZE / duration
        print("Epoch:{:d}/{:d} || Batch:{:d}/{:d} || Examples/s: {:5.1f} || D_loss:{:f} || G_loss:{:f}".format(eid+1,NUM_EPOCH,bid+1,self.num_batch,samples_per_sec,D_loss,G_loss))

    def train_discriminator(self,real_inputs):
        noise = torch.randn(BATCH_SIZE,VECTOR_LENGTH,1,1).to(self.device)
        fake_inputs = self.G(noise).detach()
        real_outputs = self.D(real_inputs)
        fake_outputs = self.D(fake_inputs)

        D_loss=self.compute_discriminator_loss(real_outputs, fake_outputs)

        self.D_optim.zero_grad()
        D_loss.backward()
        self.D_optim.step()
        self.lrd_scheduler.step()
        for p in self.D.parameters():
            p.data.clamp_(-CLIP_VALUE,CLIP_VALUE)
        return D_loss

    def train_generator(self):
        noise = torch.randn(BATCH_SIZE,VECTOR_LENGTH,1,1).to(self.device)
        fake_inputs = self.G(noise)
        fake_outputs = self.D(fake_inputs)

        G_loss = self.compute_generator_loss(fake_outputs)

        self.G_optim.zero_grad()
        G_loss.backward()
        self.G_optim.step()
        return G_loss

    def compute_discriminator_loss(self,real_logits,fake_logits):
        # return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        return -torch.mean(real_logits)+torch.mean(fake_logits)

    def compute_generator_loss(self,fake_logits):
        return -torch.mean(fake_logits)

    def save_image(self,data,path):
        save_image(data.data,path,normalize=True)
        # img = transforms.ToPILImage()(data).convert('RGB')
        # img.save(path)
        
    def save_models(self,):
        torch.save(self.G.state_dict(),os.path.join(MODELS_DIR,"Generator.pth"))
        torch.save(self.D.state_dict(),os.path.join(MODELS_DIR,"Disciminator.pth"))

    def load_models(self):
        self.G.load_state_dict(torch.load(os.path.join(MODELS_DIR,"Generator.pth")))
        self.D.load_state_dict(torch.load(os.path.join(MODELS_DIR,"Disciminator.pth")))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
