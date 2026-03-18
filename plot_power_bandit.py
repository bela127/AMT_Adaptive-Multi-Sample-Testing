from os import makedirs

import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":

    tests = np.load(f"./bandit_res/test_test.mode-bandit_sel.mode-bandit_coins-20_fake-1_pdiff-15.0000_chance-50.0000_samplemax-4000_initialsize-10_reps-1000.npy")
    #tests = np.load(f"./bandit_res/test_test.mode-bandit_sel.mode-bandit_coins-20_fake-0_pdiff-00.0000_chance-50.0000_samplemax-4000_initialsize-10_reps-1000.npy")


    reject = np.zeros(shape=(tests.shape[0],), dtype=bool)
    test_decition = np.zeros(shape=tests.shape, dtype=bool)
    for i in range(tests.shape[1]):
        reject = np.logical_or(reject, tests[:,i])
        test_decition[:,i] = reject

    power = np.sum(test_decition, axis=0)/tests.shape[0]

    plt.plot(np.arange(0, tests.shape[1])/2, power)
    plt.title("Power Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Power")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig("./bandit_res/power_curve.png")
    plt.show()
    