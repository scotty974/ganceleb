# Gan de Fabien ETHEVE  

### Architecture

```
C:.
├───checkpoints # les checkpoints de l'entrainement
├───dataset # le datasets d'entrainement
│   └───img_align_celeba
│       └───img_align_celeba
├───generated_images # les images générés de l'entrainement avec attention
├───generated_images_witou_attention # les résultats d'entrainemenet sans attention
├───model # les différents modeles 
│   ├───attention
│   ├───discriminator
│   └───generator
└───utils # les utilitaires comme le data loader

```
Nous avons aussi deux fichiers : 

train.py : qui va nous permettre de lancer l'entrainement du modele

test.py : qui va nous permettre de lancer un test avec le meilleur modele de l'entrainement

Pour le projet nous avons utilisé pytorch car il est plus facile d'utilisation que tensorflow sur le GPU.

## Courbe apprentissage sans attention
![alt text](image.png)
### et des exemples de générations : 
![alt text](image-1.png)
Et voici quelques images générées durant l'entrainement : 
Epoch 1 :
![alt text](image-2.png)

Epoch 3 : 

![alt text](image-3.png)

Epoch 5 : 

![alt text](image-4.png)

## Courbe apprentissage avec attention
![alt text](image-5.png)

## exemple de générations : 

![alt text](image-6.png)

Et voici quelques images générées durant l'entrainement : 

Epoch 1 : 

![alt text](image-7.png)

Epoch 3 : 

![alt text](image-8.png)

Epoch 5 :

![alt text](image-9.png)

