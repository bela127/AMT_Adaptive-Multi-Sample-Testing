import numpy as np
from dataclasses import dataclass

@dataclass
class Config():
    n:int = 20 # number of coins = 5, 10, 15, 20
    m:int = 0 #m<n number of plated coins
    sample_size:int = 2000 #number of trials per coin
    initial_size:int = 10 #initial_size < sample_size
    reps:int = 5000 #number of test repetitions
    common_p:float = 0.5
    p_diff:float = 0.05 #Difference in p for plated coins = 0.25 0.1 .05 .01 .005
    selection_mode:str = "ts" # selection mode = "adapt", "rand", "equal", "opt", "adapt.par", "adapt.slow"
    test_mode:str = "beta"
    coin_weights:str = "posdif"
    significance:float = 0.05

    def __post_init__(self):
        if self.m == 0: self.p_diff = 0
        if self.p_diff == 0: self.m = 0
    
    def get_sel_name(self):
        return f"mode-{self.selection_mode}_coins-{self.n}_fake-{self.m}_pdiff-{self.p_diff*100:07.4f}_chance-{self.common_p*100:07.4f}_samplemax-{self.sample_size}_initialsize-{self.initial_size}_reps-{self.reps}"

    def get_test_name(self):
        return f"test.mode-{self.test_mode}_sel.mode-{self.selection_mode}_coins-{self.n}_fake-{self.m}_pdiff-{self.p_diff*100:07.4f}_chance-{self.common_p*100:07.4f}_samplemax-{self.sample_size}_initialsize-{self.initial_size}_reps-{self.reps}"

    def load_from_name(self, file_name):
        params=file_name.split(sep="_")
        key_value = {}
        for param in params[1:]:
            key, value = param.split(sep="-")
            key_value[key] = value

        try:
            self.test_mode = key_value["test.mode"]
            self.selection_mode = key_value["sel.mode"]
        except:
            self.selection_mode = key_value["mode"]

        self.n = int(key_value["coins"]) # number of coins = 10, 15, 20, 25, 40, 70, 100
        self.m = int(key_value["fake"]) #m<n number of plated coins
        self.sample_size = int(key_value["samplemax"]) #number of trials per coin
        self.initial_size = int(key_value["initialsize"]) #initial_size < sample_size
        self.reps = int(key_value["reps"]) #number of test repetitions
        self.common_p = float(key_value["chance"])/100
        self.p_diff = float(key_value["pdiff"])/100


def calc_coin_weights_posdif(conf: Config):
    same_ps = np.ones(shape=(conf.n,))*conf.common_p
    all_diff_ps = np.ones(shape=(conf.n,))*conf.common_p+conf.p_diff
    m_diff_ps = np.concatenate((same_ps[:conf.n-conf.m], all_diff_ps[:conf.m]))

    ps = m_diff_ps
    return ps

coin_weights = {
        "posdif": calc_coin_weights_posdif,
    }
