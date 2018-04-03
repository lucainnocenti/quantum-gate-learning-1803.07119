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

IMPORTANT: During the training you may get weird errors from Theano. If this happens, it may be caused by [a bug in the way Theano computed the gradient of the matrix exponential](https://github.com/Theano/Theano/issues/6379).
The easiest way to fix this problem is probably to manually apply [this patch](https://github.com/lucainnocenti/Theano/commit/235cdfcd3b97b559652135b96c5d4b46765ba490) to your local Theano installation (one way to do it is to download the [development version](https://github.com/Theano/Theano) of Theano using `git`, then apply the above patch to the code, and finally install Theano from the local repo using `pip`).

# Results stored in net files
## 3-qubit gates without ancillae, all interactions

All of the following nets have been successfully trained with the following code (appropriately changing the `target_gate` parameter):

```
net = QubitNetwork(
    num_qubits=3,
    system_qubits=3,
    interactions='all'
)
qn.sgd_optimization(
    net=net,
    learning_rate=2,
    n_epochs=1000,
    batch_size=10,
    target_gate=gate,
    training_dataset_size=50,
    test_dataset_size=100,
    decay_rate=.01
)
```
where `gate` is generated using standard `qutip` functions (e.g. `qutip.toffoli()`, `qutip.fredkin()`, and so on).

| Target gate | Obtained fidelity |
| ---- | -------- |
| [Toffoli (CC-X)][toff3qb_1] | 1. |
| [Toffoli (CC-X)][toff3qb_2] | 0.9999 |
| [Fredkin (C-SWAP)][fredkin3qb_1] | 0.99998 |
| [Fredkin (C-SWAP)][fredkin3qb_2] | 0.99999 |
| [CC-Z][ccz3qb] | 1. |
| [CC-S][ccs3qb] | 1. |
| [CC-Hadamard][ccH3qb] | 1. |

The above target gates are generated with the following `qutip` functions:

```
toffoli = qutip.toffoli()

fredkin = qutip.fredkin()

ccZ = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
       qutip.tensor(qutip.projection(2, 1, 1), qutip.cphase(np.pi)))

ccS = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
       qutip.tensor(qutip.projection(2, 1, 1), qutip.cphase(np.pi / 2)))

ccHadamard = (qutip.tensor(qutip.projection(2, 0, 0), qutip.qeye(2), qutip.qeye(2)) +
              qutip.tensor(qutip.projection(2, 1, 1), qutip.qip.gates.controlled_gate(qutip.hadamard_transform())))
```

[toff3qb_1]: ../data/nets/toffoli_3q_all_1fid.pickle
[toff3qb_2]: ../data/nets/toffoli_3q_all_0.9999fid.pickle
[fredkin3qb_1]: ../data/nets/fredkin_3q_all_0.9999fid.pickle
[fredkin3qb_2]: ../data/nets/fredkin_3q_all_0.99999fid.pickle
[ccz3qb]: ../data/nets/ccZ_3q_all_1fid.pickle
[ccS3qb]: ../data/nets/ccS_3q_all_1fid.pickle
[ccH3qb]: ../data/nets/ccH_3q_all_1fid.pickle

## 3-qubit Toffoli gates, using topology obtained after conditions

Generating code, also found in [toffoli_only_diagonal_from_reduced_expression.ipynb](toffoli_only_diagonal_from_reduced_expression.ipynb) ([view on nbviewer](https://nbviewer.jupyter.org/github/lucainnocenti/quantum-gate-learning/blob/0879f34de99ffc55dcc344c198e7c1e3c64c699a/notebooks/toffoli_only_diagonal_from_reduced_expression.ipynb)):

    def J(*args):
    return sympy.Symbol('J' + ''.join(str(arg) for arg in args))
    def pauli(*args):
    return pauli_product(*args)
    toffoli_diagonal = J(3, 0, 0) * pauli(3, 0, 0)
    toffoli_diagonal += J(0, 3, 0) * pauli(0, 3, 0)
    toffoli_diagonal += J(0, 0, 1) * pauli(0, 0, 1)
    toffoli_diagonal += J(3, 0, 3) * (pauli(0, 0, 3) + pauli(3, 0, 3))
    toffoli_diagonal += J(0, 3, 3) * (pauli(0, 0, 3) + pauli(0, 3, 3))
    toffoli_diagonal += (J(1, 0, 1) * pauli(1, 0, 0) + J(0, 1, 1) * pauli(0, 1, 0)) * (pauli(0, 0, 0) + pauli(0, 0, 1))
    toffoli_diagonal += J(2, 2, 0) * (pauli(1, 1, 0) + pauli(2, 2, 0))
    toffoli_diagonal += J(3, 3, 0) * pauli(3, 3, 0)
    toffoli_diagonal

    net = QubitNetworkModel(sympy_expr=toffoli_diagonal, initial_values=1)
    optimizer = Optimizer(
        net=net,
        learning_rate=1.,
        n_epochs=500,
        batch_size=2,
        target_gate=qutip.toffoli(),
        training_dataset_size=200,
        test_dataset_size=100,
        decay_rate=.005,
        sgd_method='momentum'
    )
    optimizer.run()
Corresponding parameters historie*s* [here](https://nbviewer.jupyter.org/github/lucainnocenti/quantum-gate-learning/tree/0879f34de99ffc55dcc344c198e7c1e3c64c699a/data/parameters_histories/).

### Toffoli gate with only diagonal interactions
Starting with the initial "Ansatz" provided by Abdullah, we easily find a solution for the Toffoli using only interactions of type `xx`, `yy`, `zz`, `x`, `y`, `z`.

## 3 qubits + 1 ancilla networks, regular topology, only z selfinteractions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Fredkin][fredkin3qb+1a_1] | 0.996 |
| [Fredkin][fredkin3qb+1a_2] | 0.998 |
| [Fredkin][fredkin3qb+1a_3] | 0.99999 |
| [Fredkin][fredkin3qb+1a_4] | 0.999999 |
| [Toffoli][toffoli3qb+1a] | 0.989 |

[fredkin3qb+1a_1]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.996fid.pickle
[fredkin3qb+1a_2]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.998fid.pickle
[fredkin3qb+1a_3]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.99999fid.pickle
[fredkin3qb+1a_4]: ../data/nets/fredkin_3q+1a_allpairs_onlyz_0.999999fid.pickle
[toffoli3qb+1a]: ../data/nets/toffoli_3q+1a_all_0.989fid.pickle


## 3 qubits + 1 ancilla networks, regular topology, all interactions

| Target gate | Obtained fidelity |
| ----------- | ----------------- |
| [Toffredkin][toffredkin3qb+1a] | 0.99998 (possibly improvable) |

[toffredkin3qb+1a]: ../data/nets/toffredkin_3q+1a_0.9999fid.pickle
