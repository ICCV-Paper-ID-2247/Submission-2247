# Implementation of ICCV Submission 2247

# Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks

![Figure 1](Mapping_Function.png)

This repository is an PyTorch implementation of the paper [Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks]

To counter adversarial attacks, we propose Prototype Conformity Loss to class-wise disentangle intermediate features of a deep network. From the figure, it can be observed that the main reason for the existence of such adversarial samples is the close proximity of learnt features in the latent feature space.

We provide scripts for reproducing the results from our paper.


## Clone the repository
Clone this repository into any place you want.
```bash
git clone https://github.com/ICCV-Paper-ID-2247/Submission-2247
cd Submission-2247
```
## Softmax (Cross-Entropy) Training
To expedite the process of forming clusters for our proposed loss, we initially train the model using cross-entropy loss.
 
``softmax_training.py`` -- ( For initial softmax training).

* The trained checkpoints will be saved in ``Models_Softmax`` folder.


## Prototype Conformity Loss
The deep features for the prototype conformity loss are extracted from different intermediate layers using auxiliary branches, which map the features to a lower dimensional output as shown in the following figure.

![](Block_Diag.png)



``pcl_training.py`` -- ( Joint supervision with cross-entropy and our loss).

* The trained checkpoints will be saved in ``Models_PCL`` folder.

## Adversarial Training
``pcl_training_adversarial_fgsm.py`` -- ( Adversarial Training using FGSM Attack).

``pcl_training_adversarial_pgd.py`` -- ( Adversarial Training using PGD Attack).



## Testing Model's Robustness against White-Box Attacks

``robustness.py`` -- (Evaluate trained model's robustness against various types of attacks with CE loss as the adversarial loss).


``robustness_full_loss.py`` -- (Evaluate trained model's robustness against various types of attacks with combined CE and PC loss as the adversarial loss).

## Comparison of Softmax Trained Model and Our Model
Retained classification accuracy of the model's under various types of adversarial attacks:

| Training Scheme |  No Attack  |  FGSM  |   BIM   |   MIM   |   PGD   |
| :-------        | :---------- | :----- |:------  |:------  |:------  |
|     Softmax     |    92.15    |  21.48 |   0.01  |   0.02  |   0.00  |
|      Ours       |    90.45    |  66.90 |  31.29  |  32.84  |  27.05  |
