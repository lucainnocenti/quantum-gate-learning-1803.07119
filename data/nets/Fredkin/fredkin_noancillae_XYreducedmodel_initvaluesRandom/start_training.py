"""
This file is intended to be used as a script to start a training for a
specific gate.

Only the parameters at the beginning of `main` need to be modified.
"""
import glob
import os
import socket
import datetime

import numpy as np
import scipy
import sympy
import pandas as pd
import logging

import qutip
import qutip.qip.algorithms.qft

import qubit_network as qn
import qubit_network.Optimizer
import qubit_network.model

OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_FILES_NAME = 'training_no_'
HOSTNAME = socket.gethostname()
PLACEHOLDER_FILE = 'training_on_' + HOSTNAME
PLACEHOLDER_FILE_FULLDIR = os.path.join(OUTPUT_DIR, PLACEHOLDER_FILE)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    FORMAT = "[%(asctime)s %(filename)18s:%(lineno)3s - %(funcName)25s()] %(message)s"
    formatter = logging.Formatter(FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    logfilename = 'log_'
    logfilename += datetime.datetime.now().strftime('%Y%m%dh%Hm%M')
    logfilename += '.txt'
    fileHandler = logging.FileHandler(os.path.join(OUTPUT_DIR, logfilename))
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


def make_fredkin_model():
    import qubit_network.analytical_conditions as ac

    def X(i, num_qubits=3):
        zeros = [0] * num_qubits
        zeros[i - 1] = 1
        return ac.pauli_product(*zeros)
    def Y(i, num_qubits=3):
        zeros = [0] * num_qubits
        zeros[i - 1] = 2
        return ac.pauli_product(*zeros)
    def Z(i, num_qubits=3):
        zeros = [0] * num_qubits
        zeros[i - 1] = 3
        return ac.pauli_product(*zeros)
    def XY(i, j, num_qubits=3):
        first_term = X(i, num_qubits) * X(j, num_qubits)
        second_term = Y(i, num_qubits) * Y(j, num_qubits)
        return first_term + second_term

    parametrized_hamiltonian = (
        ac.J(3, 0, 0) * Z(1) +
        ac.J(0, 1, 0) * (X(2) + X(3)) +
        ac.J(0, 2, 0) * (Y(2) + Y(3)) + 
        ac.J(0, 3, 0) * (Z(2) + Z(3)) +
        ac.J(0, 1, 1) * XY(2, 3) +
        ac.J(1, 1, 0) * (XY(1, 2) + XY(1, 3))
    )
    return parametrized_hamiltonian


def main():
    # GENERAL SETTINGS
    num_system_qubits = 3
    num_ancillae = 0
    num_qubits = num_system_qubits + num_ancillae
    n_attempts = 20
    logging.info('We are going for 20 attempts.')
    # SET TARGET GATE AND INTERACTIONS
    target_gate = qutip.fredkin()
    # interactions = 'all'
    fredkin_model = make_fredkin_model()
    # HYPERPARAMETERS
    training_dataset_size = 200
    test_dataset_size = 100
    n_epochs = 300
    batch_size = 2
    sgd_method = 'adadelta'
    learning_rate = 1
    decay_rate = 0.1
    initial_values = None
    logging.info('Random initial values')
    # TAKE CARE NOT TO OVERWRITE PREVIOUSLY SAVED FILES
    prefix = os.path.join(OUTPUT_DIR, OUTPUT_FILES_NAME)
    ext = '.pickle'
    files = glob.glob(prefix + '*')
    pre_idx = 0
    if files:
        pre_idx = max([int(f[len(prefix):-len(ext)]) for f in files])
    # STARTING MESSAGES
    for idx in range(n_attempts):
        logging.info('Starting training no.{}'.format(str(idx + 1)))

        model = qn.model.QubitNetworkGateModel(
            sympy_expr=fredkin_model,
            initial_values=initial_values
        )
        optimizer = qn.Optimizer.Optimizer(
            net=model,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            target_gate=target_gate,
            training_dataset_size=training_dataset_size,
            test_dataset_size=test_dataset_size,
            sgd_method=sgd_method,
            headless=True
        )

        optimizer.run()
        
        newpath = os.path.join(
            OUTPUT_DIR,
            OUTPUT_FILES_NAME + str(1 + pre_idx + idx) + '.pickle'
        )

        optimizer.save_results(newpath)
        logging.info('Fidelity obtained: {}'.format(model.fidelity_test()))


if __name__ == '__main__':
    # make placeholder file
    if os.path.isfile(PLACEHOLDER_FILE_FULLDIR):
        raise ValueError('Training locked due to possible other training ongoing.'
                         '\nDelete "{}" to unlock.'.format(PLACEHOLDER_FILE_FULLDIR))
    with open(PLACEHOLDER_FILE_FULLDIR, 'w') as f:
        f.write('Training on ' + HOSTNAME)
    setup_logging()

    logging.info('Starting script')
    main()
    logging.info('Training finished. Deleting placeholder file.')
    os.remove(PLACEHOLDER_FILE_FULLDIR)

