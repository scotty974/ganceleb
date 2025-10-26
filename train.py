import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from utils.dataloader import CelebDataset
from model.generator.generator import Generator
from model.discriminator.discriminator import Discriminator

def main():
    # Hyperparamètres
    LATENT_DIM = 100
    IMAGE_SIZE = 64
    BATCH_SIZE = 128
    NUM_EPOCHS = 5
    LR_D = 0.0002  # Learning rate pour Discriminateur
    LR_G = 0.0002  # Learning rate pour Générateur
    BETA1 = 0.5
    NUM_WORKERS = 2

    # Label smoothing pour stabiliser l'entraînement
    REAL_LABEL = 0.9
    FAKE_LABEL = 0.0

    # Créer les dossiers de sortie
    os.makedirs('generated_images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Configuration du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Utilisation de: {device}')


    # Initialisation des poids
    def weights_init(m):
        if hasattr(m, "weight") and m.weight is not None:
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    # Transformations pour CelebA
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Charger le dataset CelebA
    # Téléchargez d'abord le dataset depuis: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    # Ou utilisez: torchvision.datasets.CelebA avec download=True
    dataset = CelebDataset(r"dataset\img_align_celeba\img_align_celeba", transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Initialiser les modèles
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Appliquer l'initialisation des poids
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Fonction de perte et optimiseurs
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))

    # Vecteur de bruit fixe pour la visualisation
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Listes pour tracer les courbes
    img_list = []
    G_losses = []
    D_losses = []

    print("Début de l'entraînement...")

    # Boucle d'entraînement
    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(dataloader):
            
            # (1) Mettre à jour le discriminateur
            netD.zero_grad()
            real_imgs = data.to(device)
            b_size = real_imgs.size(0)
            
            # Instance noise pour stabiliser l'entraînement (diminue avec les epochs)
            noise_strength = 0.2 * (1 - epoch / NUM_EPOCHS)
            real_imgs_noisy = real_imgs + torch.randn_like(real_imgs) * noise_strength
            
            # Label smoothing avec variation aléatoire
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
            label = label - torch.rand_like(label) * 0.1  # Entre 0.8 et 0.9
            
            # Forward pass avec images réelles
            output = netD(real_imgs_noisy).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # Forward pass avec images générées
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            fake_imgs = netG(noise)
            
            # Ajouter du bruit aux images fake aussi
            fake_imgs_noisy = fake_imgs + torch.randn_like(fake_imgs) * noise_strength
            
            label = torch.full((b_size,), FAKE_LABEL, dtype=torch.float, device=device)
            label = label + torch.rand_like(label) * 0.1  # Entre 0.0 et 0.1
            
            output = netD(fake_imgs_noisy.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # (2) Mettre à jour le générateur
            netG.zero_grad()
            label.fill_(REAL_LABEL)
            output = netD(fake_imgs).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Stocker les pertes
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Afficher les statistiques
            if i % 50 == 0:
                print(f'[{epoch+1}/{NUM_EPOCHS}][{i}/{len(dataloader)}] '
                    f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                    f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            # Sauvegarder des images générées périodiquement
            if i % 500 == 0 or (epoch == NUM_EPOCHS-1 and i == len(dataloader)-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(make_grid(fake, padding=2, normalize=True))
        
        # Sauvegarder des images générées à chaque epoch
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        save_image(fake, f'generated_images/fake_epoch_{epoch+1:03d}.png', 
                normalize=True, nrow=8)
        
        # Sauvegarder les modèles
        if (epoch + 1) % 10 == 0:
            torch.save(netG.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
            torch.save(netD.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')

    print("Entraînement terminé !")

    # Sauvegarder les modèles finaux
    torch.save(netG.state_dict(), 'generator_final.pth')
    torch.save(netD.state_dict(), 'discriminator_final.pth')
    print("Modèles finaux sauvegardés!")

    # Tracer les courbes de perte
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G", alpha=0.7)
    plt.plot(D_losses, label="D", alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses.png")
    plt.show()

    # Créer une animation de la progression
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save("gan_training.gif", writer="pillow")
    print("Animation sauvegardée!")

    # Générer des images finales
    with torch.no_grad():
        sample_noise = torch.randn(100, LATENT_DIM, 1, 1, device=device)
        generated = netG(sample_noise).detach().cpu()
    save_image(generated, 'generated_images/final_samples.png', normalize=True, nrow=10)


if __name__ == '__main__':
    main()