
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), # 3x64x64 -> 64x32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 64x32x32 -> 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 128x16x16 -> 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 256x8x8 -> 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), # 512x4x4 -> 1x1x1
            nn.Sigmoid()

        )


    def forward(self, input):
        return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNImgSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'

        self.image_size = 64

        tgt_transform = transforms.Compose(
                            [transforms.Resize((self.image_size, self.image_size)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(self.image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        
        src_transform = transforms.Compose([ SkeToImageTransform(self.image_size),
                                             transforms.ToTensor(), 
                                             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])


        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):

        criterion = nn.BCELoss()

 

        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.001)
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.001)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(n_epochs):
            # For each batch in the dataloader
            for ske, image in self.dataloader:
                self.netD.zero_grad()

                # Train discriminator with real data
                label = torch.full((image.size(0),), self.real_label, dtype=torch.float32)
                #label = label.reshape(label.size(0),1,1,1)

                output = self.netD(image).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()

                # Train discriminator with fake data
                fake_data = self.netG(ske)
                label.fill_(self.fake_label)
                output = self.netD(fake_data.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                errD = errD_real + errD_fake

                # Update discriminator weights
                optimizerD.step()


                # Train generator
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake_data).view(-1)
                errG = criterion(output, label)
                errG.backward()

                # Update generator weights
                optimizerG.step()

            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, n_epochs, errD.item(), errG.item()))






    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(4) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

        if key & 0xFF == ord('q'):
            break

