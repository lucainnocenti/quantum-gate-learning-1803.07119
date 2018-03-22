import os, sys
import numpy as np
import scipy
import qutip
import theano
import theano.tensor as T

src_dir = os.path.join(os.getcwd(), os.pardir)
sys.path.append(src_dir)

import qubit_network
from qubit_network.net_analysis_tools import NetsDataFolder


netsData = NetsDataFolder('../../data/nets/')
netsData.filter('qft*')[1].data
