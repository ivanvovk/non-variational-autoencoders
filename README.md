# non-variational-autoencoders

Authors: Grigorii Sotnikov, Vladimir Gogoryan, Dmitry Smorchkov, Ivan Vovk

**UDT**
https://arxiv.org/pdf/2001.05017.pdf
https://github.com/oripress/ContentDisentanglement


The problem of unsupervised domain translation is considered when a model given two non-overlapping sets of samples,
from each domain tries to approximate a mapping between between data points in different domains. And what is more there is no restriction for correspondence between objects, i.e. pairs from both domains are unordered. So a we are given samples $a \in S_A$ and $b \in S_b$, where $S_A$ and $S_B$ are two distributions from one of which we try to transfer attributes into another in a manner that samples b contain all the information that exists in samples a and some additional information.

In order to solve this problem a model of a specific structure is applied:
$f_{\theta_1}$ is a shared encoder used for both domain. It extracts information which is necessary to reconstruct images from both of them. While we apply $f_{\theta_2}$ to extract local information only about the attribute we are interested to transfer from one domain to another. Latent code of each object $x$ consists of two parts:  
$$
    f_{\theta_1, \theta_2}(x) =
    \begin{cases}
        \{f_{\theta_1}(x), f_{\theta_2}(x)\}, & \mbox{if } \mbox{x $\in S_b$}
        \\ \{f_{\theta_1}(x), 0_{E_2}\}, & \mbox{if } \mbox{x $\in S_b$}
    \end{cases}
$$
where $E_2$ represents the cardinality of the output of $f_{\theta_2}$.

Speaking formally we define 2 separate encoders responsible for the mappings $f_{\theta_1}: R^M \to R^{E_1}$ and  $f_{\theta_2}: R^M \to R^{E_2}$. Where M is the cardinality of the input $x$, $E_1$ represents the cardinality of the output of $f_{\theta_1}$, i.e. content of the object $x$ and  $E_2$ represents the cardinality of the output of $f_{\theta_2}$, i.e. features of the specific attribute of interest. If we know that this attribute is not presented in the input $x$ then we simply append $0_{E_2}$ to the content part of the latent code of $x$ ($f_{\theta_1}(x)$) instead of calculating $f{\theta_2}(x)$. In order to obtain a reconstruction of object $x$ we apply a decoder $g_{\phi}(x)$, which is a mapping $g_{\phi}: R^{E_1 + E_2} \to R^M$.
In this paper authors use a Discriminator network $d_{\omega}$ to encourage the encoder $f_{\theta_1}$ to make latent codes of samples from both datasets indistinguishable, then the only difference between them will be contained in the representation learned by $f_{\theta_2}$.
$$
    \min_{f_{\theta_1}, f_{\theta_2}, g_{\phi}} \max_{d_{\omega}} \{\mathcal{L}_A + \mathcal{L}_B - \lambda \mathcal{L}_D \},
    \\
    \mathcal{L}_A = \frac{1}{|S_A|} \sum_{a \in S_A} \|g_{\phi}(f_{\theta_1}(a), 0_{E_2}) - a\|^2
    \\
    \mathcal{L}_B = \frac{1}{|S_B|} \sum_{b \in S_B} \|g_{\phi}(f_{\theta_1}(b), f_{\theta_2}(b)) - b\|^2
$$
$$
    \mathcal{L}_D &= \frac{1}{|S_A|} \sum_{a \in S_A} l(d_{\omega}(f_{\theta_1}(a)),0) + \frac{1}{|S_B|} \sum_{b \in S_B}l(d_{\omega}(f_{\theta_1}(b)),1)
$$


Link to download aligned images in CelebA
https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM

Link to download list_attr_celeba.txt
https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs

Run a command:
python preprocess.py --root ./img_align_celeba --attributes ./list_attr_celeba.txt --dest ./glasses_train

After this command in directory /glasses_train will be located 4 files: trainA.txt, trainB.txt, testA.txt, testA.txt

Then by starting from Disentanglement.ipynb a mode can be trained from the config.yml

The other possible option is to use pretrained models which are located at https://drive.google.com/open?id=1sA6BXedG23_ZTuES_udeQlYXrssFq2BU
and proceed to Visualization.ipynb where beautiful visualizations can be plotted via written functions


**Files description**

    1) Disentanglement.ipynb - fit a FC on top of trained latent representations
    2) modules.py - parts of the model: E1, E2, Decoder, Disc
    3) utils.py - necessary functions for reproducible research
    4) preprocess.py - given folder img_align_celeba produces separate dataset partitioning
    5) Visualization.ipynb - a notebook where you can plot beautiful interpolations between samples in different settings. Source of visualization for the project.


**ACAI**
https://arxiv.org/pdf/1807.07543.pdf

One of the most common strategies how to mix latent codes is linear interpolation. It is defined in a way that $\hat{x}_{\alpha} = g_{\phi}(\alpha z_1 + (1 - \alpha)z_2)$ for some $\alpha \in [0, 1]$ and $z_1 = f_{\theta}(x_1)$, $z_2 = f_{\theta}(x_2)$ where $x_1$ and $x_2$ are original data points. As we do not know the exact appearance of $\hat{x}_{\alpha}$, we can just consider that it is semantically similar to the data points $x_1$ and $x_2$ or may be involves their interaction.

$$\mathcal{L}_d = \| d_{\omega}(\hat{x}_{\alpha}) - \alpha \|^2 + \| d_{\omega}(\gamma x + (1 - \gamma)g_{\phi}(f_{\theta}(x)) \|^2$$


Here stands out an idea to use a regularizer which forces reconstructions of interpolated points to look more realistic or even indistinguishable
from reconstructions of real images.To accomplish this issue a Critic network (Adversarial Regularizer), as is done in Generative Adversarial Networks introduced. It's inputs are interpolations of existing datapoints (i.e. $\hat{x}_{\alpha}$ as defined above). Critic network is responsible for predicting $\alpha$ given a reconstruction of mixed latent codes $\hat{x}_{\alpha}$. It can be viewed as prediction of the mixing coefficient that was used in the reconstruction process of $\hat{x}_{\alpha}$, where $\alpha \in [0, 0.5]$. On the other hand, the Autoencoder is trained to fool the critic to think that $\alpha$ is always zero (that $\hat{x}_{\alpha}$ is an original datapoint and not a reconstuction). This is achieved by adding a term to the Autoencoder’s loss function which encourages it to fool the critic. Then the following loss function to train a Critic network is proposed:

$$\mathcal{L}_d = \| d_{\omega}(\hat{x}_{\alpha}) - \alpha \|^2 + \| d_{\omega}(\gamma x + (1 - \gamma)g_{\phi}(f_{\theta}(x)) \|^2$$

$\mathcal{L}_d$ consists of two terms. The first one encourages the Critic network to guess the coefficient $\alpha$ of linear interpolation between latent codes. The second lets the Critic network to get familiar with the data distribution to have an ability to better distinguish between original data points and interpolated reconstructions. Also here we consider that during training $\|g_{\phi}(f_{\theta}(x) - x\|^2 \to 0$ according to the reconstruction loss.

$$\mathcal{L}_{f, g} = \| x -  g_{\phi}(f_{\theta}(x)) \|^2 + \lambda \cdot \| d_{\omega}(\hat{x}_{\alpha}) \|^2$$

$\mathcal{L}_{f, g}$ also consists of two terms. The first one is referred to a reconstruction loss which forces an output of Autoencoder to be as close as original image as possible. The second is Critic fooling loss which rewards Autoencoder for making the Critic network to predict irrelevant values of $\alpha$ (by making reconstructions indistinguishable from original datapoints). During inference time we do not need the Critic network.

**Files description**

    1) evaluation.py - fit a FC on top of trained latent representations
    2) modules.py - blocks of models
    3) train.py - fit ACAI and Baseline models with necessary losses
    4) utils.py - load datasets
    5) visualize.py - plot interpolated samples during training
    6) VisulizeInterpolations.ipynb - obtain interpolations for report and score models via FC layer
    7) ACAI_FINAL.ipynb - train baseline AE and ACAI from configs


Vovan
https://arxiv.org/pdf/1901.08479.pdf


**Augmentation procedure**

Another approach to increase the quality of the latent space is to change slightly the weights of encoder by augmentation procedure. First of all we fix the set $\mathcal{T}$ of augmentations $t$ - different types of perturbations, like an inversing values of one dimension, rotating the picture, adding of small color noise or others. Also we need to set up a probability distribution $\mathbf{P}_\mathcal{T}$, for example $\mathcal{U}(\mathcal{T})$. Next we take a batch of objects $X_{original}$ and consider its perturbated augmentated copy $X_{aug} = t(X_{original})$ obtained by randomly chosen $t \in \mathcal{T}$. Then we consider the images of these batches in the latent space. The correspondent distance between points $z$ from the image of original batch $Z_{original} = f_{\theta}(X_{original})$ and its pertrubated copy $Z_{aug} = f_{\theta}(X_{aug})$ could be decreased by training the weights of encoder on the cross-changed loss:
$$\mathcal{L}_{cross} =  \mathcal{L}(g_\phi(Z_{original}), X_{aug}) + \mathcal{L}(g_\phi(Z_{aug}), X_{original}).$$