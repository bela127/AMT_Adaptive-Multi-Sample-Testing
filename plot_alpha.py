from os import makedirs

import numpy as np
from matplotlib import pyplot as plt

from atm.configuration import Config


def plot_power(confs: list[Config], title, label, save, save_path: str = "./alpha_plots", load_path = "./test_res") -> None:

    conf = confs[0]

    makedirs(save_path, exist_ok=True)
    index = np.arange(conf.sample_size - conf.initial_size + 1)
    sizes = index*2+conf.initial_size+2
    
    for i, sig in enumerate(confs[0].significance):
        plt.figure(figsize=(10, 6))

        for conf in confs:
            name = conf.get_test_name()
            power = np.load(f"{load_path}/power_{name}.npy")
            print(name)
            
            plt.plot(sizes,power[i], label=label(conf))
            
        plt.title(title(conf, sig))
        plt.xlabel("Sample Size")
        plt.ylabel(rf"$\alpha$ error")
        plt.ylim(0, sig + 0.4*sig)
        plt.legend()
        plt.grid()
        
        plt.savefig(f"{save_path}/alpha.curve_{save(conf,sig)}.svg", dpi=300, bbox_inches='tight', format="svg")
        
        #plt.show()
        plt.close()

if __name__ == "__main__":
    for n in [5, 10, 15, 20]:
        for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:
            confs = []
            for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
                conf = Config(
                    n = n,
                    m = 0,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 2500,
                    common_p = 0.5,
                    p_diff = 0.00,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                confs.append(conf)

            def title(conf: Config, sig):
                return rf"$\alpha$ error for {conf.n} coins with {conf.test_mode} test at $\alpha={sig}$"
            def label(conf: Config):
                return f"{conf.selection_mode} selection"
            def save(conf: Config, sig):
                return f"test.mode-{conf.test_mode}_coins-{conf.n}_sig-{sig}"
            
            plot_power(confs, title=title, label=label, save=save, load_path="./test_res/H0_test", save_path = "./alpha_plots/sel.mode")

    for n in [5, 10, 15, 20]:
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
            confs = []
            for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:# 
                conf = Config(
                    n = n,
                    m = 0,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 2500,
                    common_p = 0.5,
                    p_diff = 0.00,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                confs.append(conf)

            def title(conf: Config, sig):
                return rf"$\alpha$ error for {conf.n} coins with selection {conf.selection_mode} at $\alpha={sig}$"
            def label(conf: Config):
                return f"{conf.test_mode} test"
            def save(conf: Config, sig):
                return f"sel.mode-{conf.selection_mode}_coins-{conf.n}_sig-{sig}"
            
            plot_power(confs, title=title, label=label, save=save, load_path="./test_res/H0_test", save_path = "./alpha_plots/test.mode")
    
    for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:
        for sel_mode in ["ts.5", "ts", "equal", "beta", "means", "mean.slow"]:# 
            confs = []
            for n in [5, 10, 15, 20]:
                conf = Config(
                    n = n,
                    m = 0,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 2500,
                    common_p = 0.5,
                    p_diff = 0.00,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                confs.append(conf)

            def title(conf: Config, sig):
                return rf"$\alpha$ error for {conf.test_mode} test with selection {conf.selection_mode} at $\alpha={sig}$"
            def label(conf: Config):
                return rf"$n = {conf.n}$"
            def save(conf: Config, sig):
                return f"sel.mode-{conf.selection_mode}_test.mode-{conf.test_mode}_sig-{sig}"
            
            plot_power(confs, title=title, label=label, save=save, load_path="./test_res/H0_test", save_path = "./alpha_plots/coins")
        
    for n in [5, 10, 15, 20]:
        confs = []
        for test_mode in ["mean", "chi2","kw", "beta", "betabinom.comb"]:
            for sel_mode in ["equal", "beta"]:# 
                conf = Config(
                    n = n,
                    m = 0,
                    sample_size = 2000,
                    initial_size = 10,
                    reps = 2500,
                    common_p = 0.5,
                    p_diff = 0.00,
                    selection_mode = sel_mode,
                    test_mode = test_mode,
                    coin_weights = "posdif",
                )
                confs.append(conf)

            def title(conf: Config, sig):
                return rf"$\alpha$ error for {conf.n} coins, equal vs beta selection at $\alpha={sig}$"
            def label(conf: Config):
                return f"{conf.selection_mode} selection, {conf.test_mode} test"
            def save(conf: Config, sig):
                return f"test.mode-equal.vs.beta_coins-{conf.n}_sig-{sig}"
            
            plot_power(confs, title=title, label=label, save=save, load_path="./test_res/H0_test", save_path = "./alpha_plots/equal.vs.beta")
