import numpy as np
from dataclasses import dataclass

@dataclass
class Config():
    n:int = 20 # number of coins = 5, 10, 15, 20
    m:int = 0 #m<n number of plated coins
    sample_size:int = 2000 #number of trials per coin
    initial_size:int = 10 #initial_size < sample_size
    reps:int = 10000 #number of test repetitions
    common_p:float = 0.5
    p_diff:float = 0.05 #Difference in p for plated coins = 0.25 0.1 .05 .01 .005
    selection_mode:str = "ts" # selection mode = "adapt", "rand", "equal", "opt", "adapt.par", "adapt.slow"
    test_mode:str = "beta"
    coin_weights:str = "posdif"
    significance:tuple = (0.05,0.025,0.01)
    dataset:str = ""

    def __post_init__(self):
        self._check_m_p_diff("p_diff")
        self._check_m_p_diff("m")

    def __setattr__(self, prop, val):
        super().__setattr__(prop, val)
        if prop == "p_diff" or prop == "m":
            self._check_m_p_diff(prop)
    
    def _check_m_p_diff(self, prop):
        if prop == "p_diff":
            if self.p_diff == 0 and self.m != 0: self.m = 0
        if prop == "m":
            if self.m == 0 and self.p_diff != 0: self.p_diff = 0
    
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

    def clone(self):
        return Config(
            n=self.n,
            m=self.m,
            sample_size=self.sample_size,
            initial_size=self.initial_size,
            reps=self.reps,
            common_p=self.common_p,
            p_diff=self.p_diff,
            selection_mode=self.selection_mode,
            test_mode=self.test_mode,
            coin_weights=self.coin_weights
        )


def calc_coin_weights_posdif(conf: Config):
    same_ps = np.ones(shape=(conf.n,))*conf.common_p
    all_diff_ps = np.ones(shape=(conf.n,))*conf.common_p+conf.p_diff
    m_diff_ps = np.concatenate((same_ps[:conf.n-conf.m], all_diff_ps[:conf.m]))

    ps = m_diff_ps
    return ps

coin_weights = {
        "posdif": calc_coin_weights_posdif,
    }
