# Interpretable Control Exploration and Counterfactual Explanation (ICE) on StyleGAN

![teaser](./ice/teaser.png)

**Which Style Makes Me Attractive? Interpretable Control Discovery and Counterfactual Explanation on StyleGAN**

Bo Li, Qiulin Wang, Jiquan Pei, Yu Yang, Xiangyang Ji

Abstract: _The semantically disentangled latent subspace in GAN provides rich interpretable controls in image generation. This paper includes two contributions on semantic latent subspace analysis in the scenario of face generation using StyleGAN2. 
First, we propose a novel approach to disentangle latent subspace semantics by exploiting existing face analysis models, e.g., face parsers and face landmark detectors. These models provide the flexibility to construct various criterions with very concrete and interpretable semantic meanings (e.g., change face shape or change skin color) to restrict latent subspace disentanglement. Rich latent space controls unknown previously can be discovered using the constructed criterions. 
Second, we propose a new perspective to explain the behavior of a CNN classifier by generating counterfactuals in the interpretable latent subspaces we discovered. This explanation helps reveal whether the classifier learns semantics as intended.
Experiments on various disentanglement criterions demonstrate the effectiveness of our approach. We believe this approach contributes to both areas of image manipulation and counterfactual explainability of CNNs._

----

The code is developed on [`NVlabs/stylegan2-ada-pytorch`](https://github.com/NVlabs/stylegan2-ada-pytorch) and put in the `ice` folder. Please play with the two ipython notebooks.

* `ice/discover_subspaces` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prclibo/ice/blob/ice/ice/discover_subspaces.ipynb)

    Solve subspaces by using face analysis models as criterions. Currently we only include several representative subspaces. The notebook requires to download some pre-trained models. You might have to spend some efforts to put everything at the right place. See the notebook comments for details. This notebook shows the code sketch to generate Figure 3 (as below) in the paper, i.e., the latent subspace for interpretable face manipulation.

![subspaces](./ice/subspaces.png)

* `ice/explain_counterfactually` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prclibo/ice/blob/ice/ice/explain_counterfactually.ipynb)
    
    Use the interpretable subspaces discovered by the above notebook to explain the classifier of attractiveness. This notebook shows the code sketch to generate Figure 4 (as below) in the paper, i.e., the interpretable counterfactuals to increase attractiveness score of a given classifier. Since we did not find good public pre-trained model. The attractiveness classifier is trained by ourselves using `d-li14/face-attribute-prediction`.

![coutnerfactuals](./ice/counterfactuals.png)
