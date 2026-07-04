from os import makedirs
import time
import numpy as np

from amt.configuration import Config
from amt.tests import Test

if __name__ == "__main__":
    test_suite = Test()
    
    # --- CONFIGURABLE APPROACH FILTER ---
    # To run everything: list(test_suite.test_modes.keys())
    # To run a single approach: ["one.vs.rest.beta.mixture"]
    test_modes = list(test_suite.test_modes.keys())

    # Environment Parameters
    n_values = np.array([5, 10, 15, 20, 30])
    bench_max_iterations = 200  
    bench_reps = 25             
    
    save_path = "./time_benchmark"
    makedirs(save_path, exist_ok=True)

    # Core configuration array metadata must remain globally present for the plotting layer
    np.save(f"{save_path}/benchmark_n_values.npy", n_values)

    print("=========================================")
    print(" INCREMENTAL APPROACH TIME BENCHMARK     ")
    print("=========================================\n")

    for test_mode in test_modes:
        print(f"\n>>> Profiling Approach: {test_mode} <<<")
        
        # Prepare 1D array profile across our designated n combinations
        approach_runtimes = np.empty(len(n_values), dtype=np.float64)

        for n_idx, n in enumerate(n_values):
            # Generate monotonic increasing history for this specific coin dimension size
            delta_heads = np.random.randint(0, 3, size=(n, bench_max_iterations, bench_reps))
            delta_tails = np.random.randint(0, 3, size=(n, bench_max_iterations, bench_reps))
            
            initial_heads = np.random.randint(5, 10, size=(n, bench_reps))
            initial_tails = np.random.randint(5, 10, size=(n, bench_reps))
            
            cum_heads = initial_heads[:, None, :] + np.cumsum(delta_heads, axis=1)
            cum_tails = initial_tails[:, None, :] + np.cumsum(delta_tails, axis=1)
            
            timeline_contingency = np.stack([cum_heads, cum_tails], axis=0)

            dummy_conf = Config(
                n=int(n), m=1, sample_size=bench_max_iterations, initial_size=10, 
                reps=bench_reps, common_p=0.5, p_diff=0.075, hyp=1,
                selection_mode="beta.med", test_mode=test_mode
            )
            
            start_time = time.perf_counter()
            test_engine = Test(conf=dummy_conf)
            
            # Inner core evaluation profiling loop
            for idx_i in range(bench_max_iterations):
                for idx_r in range(bench_reps):
                    _ = test_engine.test(timeline_contingency[:, :, idx_i, idx_r], conf=dummy_conf)
                    
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            total_evaluations = bench_max_iterations * bench_reps
            normalized_cost = (elapsed / total_evaluations) * 1000
            
            print(f" [n={n:2d}] Cost/1k ops: {normalized_cost:.4f}s")
            approach_runtimes[n_idx] = normalized_cost

        # Save this single approach array dynamically (overwrites if re-run, appends if new)
        np.save(f"{save_path}/bench_{test_mode}.npy", approach_runtimes)
        print(f" Saved standalone: {save_path}/bench_{test_mode}.npy")

    print("\nBenchmark pass finished successfully.")