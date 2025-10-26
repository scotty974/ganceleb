import torch
import matplotlib.pyplot as plt
import torchvision
from model.generator.generator import Generator
import numpy as np

def visualize_generated_images(model_path='generator_final.pth', num_images=64, save_path='generated_samples.png'):
    """
    Génère et visualise des images avec le générateur entraîné
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Charger le générateur
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # Générer des images
    with torch.no_grad():
        noise = torch.randn(num_images, 100, 1, 1, device=device)
        fake_images = netG(noise).detach().cpu()
    
    # Créer une grille d'images
    grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True, nrow=8)
    
    # Afficher
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Images générées par le GAN")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ {num_images} images générées et sauvegardées dans '{save_path}'")
    return fake_images


def compare_real_vs_fake(data_dir, model_path='generator_final.pth', num_samples=32):
    """
    Compare des images réelles vs générées côte à côte
    """
    import torchvision.transforms as transforms
    from utils.dataloader import CelebDataset
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Charger des vraies images
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = CelebDataset(data_dir, transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    real_batch = next(iter(dataloader))
    
    # Charger le générateur
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # Générer des fausses images
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake_batch = netG(noise).detach().cpu()
    
    # Créer la comparaison
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Vraies images
    real_grid = torchvision.utils.make_grid(real_batch[:num_samples], padding=2, normalize=True, nrow=8)
    axes[0].imshow(np.transpose(real_grid, (1, 2, 0)))
    axes[0].set_title("Images RÉELLES (CelebA)", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Fausses images
    fake_grid = torchvision.utils.make_grid(fake_batch, padding=2, normalize=True, nrow=8)
    axes[1].imshow(np.transpose(fake_grid, (1, 2, 0)))
    axes[1].set_title("Images GÉNÉRÉES (GAN)", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_vs_fake_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Comparaison sauvegardée dans 'real_vs_fake_comparison.png'")


def analyze_diversity(model_path='generator_final.pth', num_batches=5, batch_size=64):
    """
    Vérifie la diversité des images générées (détection de mode collapse)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    all_images = []
    
    print("🔍 Génération de plusieurs batches pour tester la diversité...")
    with torch.no_grad():
        for i in range(num_batches):
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise).detach().cpu()
            all_images.append(fake)
    
    # Afficher quelques échantillons de chaque batch
    fig, axes = plt.subplots(num_batches, 8, figsize=(16, num_batches * 2))
    
    for batch_idx in range(num_batches):
        for img_idx in range(8):
            img = all_images[batch_idx][img_idx]
            img = (img + 1) / 2  # dénormaliser de [-1,1] à [0,1]
            axes[batch_idx, img_idx].imshow(np.transpose(img.numpy(), (1, 2, 0)))
            axes[batch_idx, img_idx].axis('off')
            
            if img_idx == 0:
                axes[batch_idx, img_idx].set_ylabel(f'Batch {batch_idx+1}', fontsize=10)
    
    plt.suptitle("Test de diversité - Échantillons de différents batches", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diversity_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Test de diversité sauvegardé dans 'diversity_test.png'")
    print("📊 Vérifiez visuellement si les visages sont variés ou similaires")


def interpolate_latent_space(model_path='generator_final.pth', steps=10):
    """
    Interpole dans l'espace latent pour voir les transitions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    
    # Deux points aléatoires dans l'espace latent
    z1 = torch.randn(1, 100, 1, 1, device=device)
    z2 = torch.randn(1, 100, 1, 1, device=device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = netG(z_interp).detach().cpu()
            interpolated_images.append(img)
    
    # Afficher l'interpolation
    fig, axes = plt.subplots(1, steps, figsize=(20, 2.5))
    
    for i, img in enumerate(interpolated_images):
        img = (img[0] + 1) / 2  # dénormaliser
        axes[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axes[i].axis('off')
        axes[i].set_title(f'{i/(steps-1):.1f}', fontsize=8)
    
    plt.suptitle("Interpolation dans l'espace latent (vérification de la continuité)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('latent_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Interpolation sauvegardée dans 'latent_interpolation.png'")


def main():
    """
    Lance toutes les visualisations
    """
    print("=" * 60)
    print("🎨 ANALYSE COMPLÈTE DES RÉSULTATS DU GAN")
    print("=" * 60)
    
    model_path = 'generator_final.pth'
    data_dir = r"dataset\img_align_celeba\img_align_celeba"
    
    try:
        print("\n1️⃣ Génération d'échantillons...")
        visualize_generated_images(model_path, num_images=64)
        
        print("\n2️⃣ Comparaison Réel vs Fake...")
        compare_real_vs_fake(data_dir, model_path, num_samples=32)
        
        print("\n3️⃣ Test de diversité...")
        analyze_diversity(model_path, num_batches=5, batch_size=64)
        
        print("\n4️⃣ Interpolation dans l'espace latent...")
        interpolate_latent_space(model_path, steps=10)
        
        print("\n" + "=" * 60)
        print("✅ ANALYSE TERMINÉE !")
        print("=" * 60)
        print("\n📁 Fichiers générés:")
        print("   - generated_samples.png")
        print("   - real_vs_fake_comparison.png")
        print("   - diversity_test.png")
        print("   - latent_interpolation.png")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        print("Assurez-vous que 'generator_final.pth' existe dans le dossier actuel.")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")


if __name__ == "__main__":
    main()