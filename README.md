# Interpretable Control Exploration and Counterfactual Explanation (ICE)

This is the supplementary code of the paper. The code is developed on `NVlabs/stylegan2-ada-pytorch` and put in the `ice` folder. Please play with the two ipython notebooks:

* `ice/discover_subspaces`: Solve subspaces by using face analysis models as criterions. Currently we only include several representative subspaces. The notebook requires to download some pre-trained models. You might have to spend some efforts to put everything at the right place. See the notebook comments for details.

* `ice/explain_counterfactually`: Use the interpretable subspaces discovered by the above notebook to explain the classifier of attractiveness. Since we did not find good public pre-trained model. The attractiveness classifier is trained by ourselves using `d-li14/face-attribute-prediction` and oversized to upload 
