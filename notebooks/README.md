# Guide to notebooks

NOTE: The notebooks containing dynamic content will not render correctly from GitHub.
The use of [nbviewer](https://nbviewer.jupyter.org/) is recommended for better results.

- [Toffoli](./Toffoli): notebooks to train and analyse qubit networks to implement the Toffoli gate.
  - [toffoli_only_diagonal_from_reduced_expression.ipynb](./Toffoli/toffoli_only_diagonal_from_reduced_expression.ipynb): Code used to train the model for the Toffoli gate with only diagonal pairwise interactions, starting from the reduced expression obtained via the conditions in the paper. It also contains the code to generate the plots for the Toffoli in [the paper](https://arxiv.org/abs/1803.07119).
- [Fredkin](./Fredkin): notebooks to train and analyse qubit networks to implement the Fredkin gate. 
  - [fredkin_only_diagonal_from_reduced_expression.ipynb](./Fredkin/fredkin_only_diagonal_from_reduced_expression.ipynb): Code used to train the model for the Fredkin gate with only diagonal pairwise interactions, starting from the reduced expression obtained via the conditions in the paper. It also contains the code to generate the plots for the Fredkin in [the paper](https://arxiv.org/abs/1803.07119).
  - [fredkin_Banchietal.ipynb](./Fredkin/fredkin_Banchietal.ipynb): Notebook used to reproduce the results in [Banchi et al. 2015](https://www.nature.com/articles/npjqi201619) using the new method.
- [doublefredkin.ipynb](./doublefredkin.ipynb): Training of four-qubit networks to implement the four-qubit *double Fredkin* gate.
- [half-adder.ipynb](./half-adder.ipynb): Training of four-qubit networks to implement the three-qubit half-adder gate (as defined in [Barbosa 2006](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.73.052321)), using a single ancillary qubit.
- [toffredkin.ipynb](./toffredkin.ipynb): Training of four-qubit networks to implement the three-qubit *Toffredkin* gate, using a single ancillary qubit.
- [qft_3qubit+5ancillae.ipynb](./qft_3qubit+5ancillae.ipynb): Training of eight-qubit networks to implement the three-qubit QFT gate, using five ancillary qubits.

# Requirements
To run the cells in each notebook, the only requirement is to first run its first cell, which should import all the required packages.
Besides standard packages, two fundamental requirements to run the code are [`qutip`](http://qutip.org/) and [`theano`](http://deeplearning.net/software/theano/).
It is generally recommended to use [`conda`](https://conda.io/docs/) to install and organise all the necessary packages in an isolated environment (but any other method can be used if you know what you are doing).
Note that installing `qutip` and `theano` may be problematic on some operative system (see e.g. [here](http://www.cgranade.com/blog/2016/08/22/qutip-on-wsl.html) for instruction on how to install `qutip` on Windows).

IMPORTANT: During the training you may get weird errors from Theano. If this happens, it may be caused by [a bug in the way Theano computes the gradient of the matrix exponential](https://github.com/Theano/Theano/issues/6379).
An easy way to fix this problem is to manually apply [this patch](https://github.com/lucainnocenti/Theano/commit/235cdfcd3b97b559652135b96c5d4b46765ba490) to your local Theano installation (one way to do it is to download the [development version](https://github.com/Theano/Theano) of Theano using `git`, then apply the above patch to the code, and finally install Theano from the local repo using `pip`).