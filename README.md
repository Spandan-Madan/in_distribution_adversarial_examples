<div align="center">
<h3>Adversarial examples within the training distribution: A widespread challenge.</h3>
  <!--img src="docs/images/fig_1_github.png" alt="Teaser Figure"--!>
  <a href="https://arxiv.org/abs/2007.08032">Paper</a> •
  <a href="#overview">Overview</a> •
  <a href="#demos">Demos</a> •
  <a href="#codebase">Using the Codebase</a> •
  <a href="#data">Accessing the data</a> •
</div>

This repository contains the official implementation of our paper: *Adversarial examples within the training distribution: A widespread challenge.* Here you can find the code, demos and the data used for this project.

The paper can be accessed [here](https://arxiv.org/abs/2106.16198).

<div align="center">
<h3>Authors</h3>
  <a href="https://spandan-madan.github.io/about/">Spandan Madan</a> •
  <a href="https://cbmm.mit.edu/about/people/sasaki">Tomotake Sasaki</a> •
  <a href="https://vcg.seas.harvard.edu/people/hanspeter-pfister">Hanspeter Pfister</a> •
  <a href="https://cseweb.ucsd.edu/~tzli/">Tzu-Mao Li</a> •
  <a href="https://web.mit.edu/xboix/www/index.html">Xavier Boix</a>
</div>

# Overview
Despite a plethora of proposed theories, understanding why deep neural networks are susceptible to adversarial attacks remains an open question. A promising recent strand of research investigates adversarial attacks within the training data distribution, providing a more stringent and worrisome definition for these attacks. These theories posit that the key issue is that in high dimensional datasets, most data points are close to the ground-truth class boundaries. This has been shown in theory for some simple data distributions, but it is unclear if this theory is relevant in practice. Here, we demonstrate the existence of in-distribution adversarial examples for object recognition. This result provides evidence supporting theories attributing adversarial examples to the proximity of data to ground-truth class boundaries, and calls into question other theories which do not account for this more stringent definition of adversarial attacks. These experiments are enabled by our novel gradient-free, evolutionary strategies (ES) based approach for finding in-distribution adversarial examples in 3D rendered objects, which we call CMA-Search.

# Demos
We provide three demos to reproduce our main findings on in-distribution adversarial examples, one for each level of data complexity we investigate in our paper:-

1. Simplistic parametrically controlled data sampled from disjoint per-category uniform distributions: [LINK](https://github.com/Spandan-Madan/in_distribution_adversarial_examples/blob/main/demos/demo_uniform_data.ipynb)
 
2. Parametric and controlled images of objects using our graphics pipeline: [LINK](https://github.com/Spandan-Madan/in_distribution_adversarial_examples/blob/main/demos/demo_uniform_data.ipynb)
 
3. Natural image data from the ImageNet dataset: [LINK](https://github.com/Spandan-Madan/in_distribution_adversarial_examples/blob/main/demos/demo_uniform_data.ipynb)

# Codebase

Our work builds on three existing pipelines---[Redner](https://github.com/BachiLi/redner) for rendering images, [PyCMA] (https://github.com/CMA-ES/pycma) for running the CMA-ES search algorithm, and [Single view MPI](https://github.com/google-research/google-research/tree/master/single_view_mpi) for generating novel views with ImageNet. Each of these libraries were modified from their original versions to adapt them for searching in-distribution adversarial examples with *CMA-Search*, which we propose in our work. Thus, we provide the adapted versions with this codebase in the folders `redner`, `cma` and `single_view_mpi` respectively.

`rendering`: Contains scripts for our computer graphics pipeline to render ShapeNet objects under lighting and viewpoint variations.

`training_models`: Contains scripts and notebooks for training our visual recognition models.

`other_optimization_methods`: Contains scripts and notebooks for attacking trained models using *CMA-Search*.

`demos`: Contains demos covering every aspect of training models, and attacking them using *CMA-Search* mentioned above.

The best entry point to explore this code base would be to start from `demos` to understand the results, and explore how the rest of the code is used to reproduce the results in the paper.

# Data
