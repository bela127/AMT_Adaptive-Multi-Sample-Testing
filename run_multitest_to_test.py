import numpy as np

from atm.configuration import Config

def combined_test(tests):
    test = np.any(tests, axis=1)
    test = np.transpose(test,axes=(0,2,1))
    return test

def load_data(conf: Config, load_path: str):
    name = conf.get_test_name()
    tests = np.load(f"{load_path}/multi.teststat_{name}.npy")
    print(name)
    return tests

def calc_power(test):
    reject_count = np.count_nonzero(test, axis=1)
    power = reject_count/test.shape[1]
    return power

def run(conf: Config, load_path: str = "./multi_stat_res", save_path: str = "stat_res"):
    tests = load_data(conf, load_path)
    test = combined_test(tests)
    power = calc_power(test)

    np.save(f"{save_path}/reject_{conf.get_test_name()}.npy", test)
    np.save(f"{save_path}/power_{conf.get_test_name()}.npy", power)
    return test, power


if __name__ == "__main__":
    for n in [5, 10, 15, 20]:
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
            for test_mode in ["betabinom.comb"]:
                for m in [1]:#[0,1]:

                    conf = Config(
                        n = n,
                        m = m,
                        sample_size = 2000,
                        initial_size = 10,
                        reps = 5000,
                        selection_mode = sel_mode,
                        test_mode = test_mode,
                    )

                    run(conf, load_path="./multi_stat_res/H1", save_path="./test_res/H1")

