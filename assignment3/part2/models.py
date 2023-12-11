################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-20
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """
        Convolutional Encoder network with Convolution and Linear layers, ReLU activations. The output layer
        uses a Fully connected layer to embed the representation to a latent code with z_dim dimension.
        Inputs:
            z_dim - Dimensionality of the latent code space.
        """
        super(ConvEncoder, self).__init__()
        c_hid = num_filters
        act_fn = nn.ReLU
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch of Images. Shape: [B,C,H,W]
        Outputs:
            z - Output of latent codes [B, z_dim]
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        z = self.net(x)
        return z

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class ConvDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 28,
                 z_dim: int = 20):
        """
        Convolutional Decoder network with linear and deconvolution layers and ReLU activations. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(ConvDecoder, self).__init__()
        c_hid = num_filters
        act_fn = nn.GELU
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            x - Reconstructed image of shape [B,C,H,W]
        """
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        """
        Discriminator network with linear layers and LeakyReLU activations.
        Inputs:
              z_dim - Dimensionality of the latent code space.
        """
        super(Discriminator, self).__init__()
        act_fn = nn.LeakyReLU(negative_slope=0.2)
        c_hid = 512
        self.net = nn.Sequential(
            nn.Linear(z_dim, c_hid),
            act_fn(),
            nn.Linear(c_hid, c_hid),
            act_fn(),
            nn.Linear(c_hid, 1),
        )

    def forward(self, z):
        """
        Inputs:
            z - Batch of latent codes. Shape: [B,z_dim]
        Outputs:
            preds - Predictions whether a specific latent code is fake (<0) or real (>0). 
                    No sigmoid should be applied on the output. Shape: [B,1]
        """
        preds = self.forward(z)
        return preds

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class AdversarialAE(nn.Module):
    def __init__(self, z_dim: int = 20, 
                 num_input_channels: int = 16, num_filters: int = 28):
        """
        Adversarial Autoencoder network with a Encoder, Decoder and Discriminator.
        Inputs:
              z_dim - Dimensionality of the latent code space. This is the number of neurons of the code layer
        """
        super(AdversarialAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = ConvEncoder(num_input_channels, num_filters, z_dim)
        self.decoder = ConvDecoder(num_input_channels, num_filters, z_dim)
        self.discriminator = Discriminator(z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            recon_x - Reconstructed image of shape [B,C,H,W]
            z - Batch of latent codes. Shape: [B,z_dim]
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def get_loss_autoencoder(self, x, x_hat, z_fake, lambda_=1):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
            x_hat - Reconstructed image of shape [B,C,H,W]
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]
            lambda_ - The reconstruction coefficient (between 0 and 1).

        Outputs:
            recon_loss - The MSE reconstruction loss between actual input and its reconstructed version.
            gen_loss - The Generator loss for fake latent codes extracted from input.
            ae_loss - The combined adversarial and reconstruction loss for AAE
                lambda_ * reconstruction loss + (1 - lambda_) * adversarial loss
        """
        L_rec = nn.MSELoss(x_hat, x)
        L_gen = self.get_loss_discriminator(z_fake)
        L_ae = (1 - lambda_) * L_rec + lambda_ * L_gen
        logging_dict = {"gen_loss": L_gen,
                        "recon_loss": L_rec,
                        "ae_loss": L_ae}
        return L_ae, logging_dict

    def get_loss_discriminator(self, z_fake):
        """
        Inputs:
            z_fake - Batch of latent codes for fake samples. Shape: [B,z_dim]

        Outputs:
            disc_loss - The discriminator loss for real and fake latent codes.
            logging_dict - A dictionary for logging the model performance by following keys:
                disc_loss - The discriminator loss for real and fake latent codes.
                loss_real - The discriminator loss for latent codes sampled from the standard Gaussian prior.
                loss_fake - The discriminator loss for latent codes extracted by encoder from input
                accuracy - The accuracy of the discriminator for both real and fake samples.
        """
        z_real = torch.randn((z_fake.shape))
        loss_fake = (self.discriminator(z_fake) < 0).float().sum()
        loss_real = (self.discriminator(z_real) >= 0).float().sum()
        disc_loss = loss_fake + loss_real
        logging_dict = {"disc_loss": disc_loss,
                        "loss_real": loss_real,
                        "loss_fake": loss_fake,
                        "accuracy": None}
        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        return disc_loss, logging_dict

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random or conditioned images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = None
        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return self.encoder.device


