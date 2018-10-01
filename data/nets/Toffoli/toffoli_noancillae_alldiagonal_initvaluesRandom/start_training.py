"""
This file is intended to be used as a script to start a training for a
specific gate.

Only the parameters at the beginning of `main` need to be modified.
"""
import glob
import os
import socket
import datetime
import itertools

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


def make_alldiagonal_model(num_qubits):
    import qubit_network.analytical_conditions as ac

    indices = ac.pairwise_diagonal_interactions_indices(num_qubits)
    return ac.indices_to_hamiltonian(indices)


def main():
    # GENERAL SETTINGS
    num_system_qubits = 3
    num_ancillae = 0
    num_qubits = num_system_qubits + num_ancillae
    # the attempts will be equally (when possible) distributed among these
    # initial values. Each initial value will be used the same amount of 
    # times.
    attempted_initvalues = ['random', 0, 1, 2, 3, 4]
    num_attempts_per_initvalue = 10
    num_attempts = num_attempts_per_initvalue * len(attempted_initvalues)
    logging.info('We are going for a total of {}'
                 'attempts.'.format(num_attempts))
    logging.info('The following initial values will be used:'
                 '{}.'.format(attempted_initvalues))
    # SET TARGET GATE AND INTERACTIONS
    target_gate = qutip.toffoli()
    # interactions = 'all'
    toffoli_model = make_alldiagonal_model(num_qubits)
    # HYPERPARAMETERS
    training_dataset_size = 200
    test_dataset_size = 100
    n_epochs = 100
    batch_size = 2
    sgd_method = 'momentum'
    learning_rate = 1
    decay_rate = 0.1
    # TAKE CARE NOT TO OVERWRITE PREVIOUSLY SAVED FILES
    prefix = os.path.join(OUTPUT_DIR, OUTPUT_FILES_NAME)
    ext = '.pickle'
    files = glob.glob(prefix + '*')
    pre_idx = 0
    if files:
        pre_idx = max([int(f[len(prefix):-len(ext)]) for f in files])
    # STARTING MESSAGES
    for idx in range(num_attempts):
        initial_values = attempted_initvalues[idx // num_attempts_per_initvalue]
        logging.info('Starting training no.{}'.format(str(idx + 1)))
        logging.info('Initial values: {}.'.format(initial_values))

        model = qn.model.QubitNetworkGateModel(
            sympy_expr=toffoli_model,
            initial_values=initial_values,
            num_system_qubits=num_system_qubits
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
        logging.info('Fidelity obtained: {}'.format(model.average_fidelity()))


if __name__ == '__main__':
    # make placeholder file
    if os.path.isfile(PLACEHOLDER_FILE_FULLDIR):
        raise ValueError('Training locked due to possible other training ongo'
                         'ing.\nDelete "{}" to unlock.'.format(
                             PLACEHOLDER_FILE_FULLDIR))
    with open(PLACEHOLDER_FILE_FULLDIR, 'w') as f:
        f.write('Training on ' + HOSTNAME)
    setup_logging()

    logging.info('Starting script')
    main()
    logging.info('Training finished. Deleting placeholder file.')
    os.remove(PLACEHOLDER_FILE_FULLDIR)

