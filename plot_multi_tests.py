import numpy as np
import matplotlib.pyplot as plt
from os import makedirs


file_name = "multi.teststat_test.mode-betabinom.comb_sel.mode-beta_coins-5_fake-0_pdiff-00.0000_chance-50.0000_samplemax-2000_initialsize-10_reps-5000"
alpha = 0.05


params=file_name.split(sep="_")
key_value = {}
for param in params[1:]:
    key, value = param.split(sep="-")
    key_value[key] = value

try:
    test_mode = key_value["test.mode"]
    selection_mode = key_value["sel.mode"]
except:
    selection_mode = key_value["mode"]

test_mode = "betabinom.comb"

n = int(key_value["coins"]) # number of coins = 10, 15, 20, 25, 40, 70, 100
m = int(key_value["fake"]) #m<n number of plated coins
sample_size = int(key_value["samplemax"]) #number of trials per coin
initial_size = int(key_value["initialsize"]) #initial_size < sample_size
reps = int(key_value["reps"]) #number of test repetitions
common_p = float(key_value["chance"])/100
p_diff = float(key_value["pdiff"])/100

tests = np.load(f"./multi_stat_res/H0/{file_name}.npy")
#tests = tests[...,:200]

index = np.arange(tests.shape[2])
sizes = index*2+initial_size+2

reject_count = np.count_nonzero(tests, axis=3)
power = reject_count/tests.shape[3]


fig, ax = plt.subplots()

ax.plot(sizes[None,...].T,power[0].T, linestyle = "dotted",label=f"emp sig={alpha}")

plt.xlabel("sample size")
plt.ylabel("power")
if m==0: plt.ylim(0,0.025)
else: plt.ylim(0,1)
plt.legend()
makedirs("./coin_plot/", exist_ok=True)
plt.savefig(f"./coin_plot/power_single__teststat_test.mode-{test_mode}_sel.mode-{selection_mode}_coins-{n}_fake-{m}_pdiff-{p_diff*100:07.4f}_chance-{common_p*100:07.4f}_samplemax-{sample_size}_initialsize-{initial_size}_reps-{reps}.png")
plt.show()