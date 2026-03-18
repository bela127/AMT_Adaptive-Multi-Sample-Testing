from amt.configuration import Config
from amt.permutation_test_runner import Experiment

datasets = ['contingency_data-arabica.clean_value-Moisture Percentage_by-Harvest Year', 'contingency_data-arabica.clean_value-Acidity_by-Harvest Year', 'contingency_data-arabica.clean_value-Category Two Defects_by-Harvest Year', 'contingency_data-arabica.clean_value-Flavor_by-Color', 'contingency_data-arabica.clean_value-Acidity_by-Color', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Color', 'contingency_data-arabica.clean_value-Flavor_by-Country of Origin', 'contingency_data-arabica.clean_value-Category Two Defects_by-Country of Origin', 'contingency_data-arabica.clean_value-Aftertaste_by-Country of Origin']
ms = [4,4,4,9,9,9,14,14,14]
ps = [56.1088,47.3296,31.0599, 57.4741, 54.1648, 53.8864, 45.5353, 44.5184, 35.5670]
pdiffs = [26.3889,15.0738,22.0058, 55.5556, 32.1429, 25.0000, 76.1905, 47.6190, 52.3810]

if __name__ == "__main__":
    for dataset, mh1, p, pdiff in zip(datasets, ms, ps, pdiffs):
        for mode in ["equal", "beta.med"]:#  "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for test_mode in ["chi2","beta"]:
                null_conf = Config(
                    sample_size = 1000,
                    m=0,
                    initial_size = 10,
                    reps = 5000,
                    selection_mode = mode,
                    test_mode = test_mode,
                    dataset= dataset,
                    n=mh1,
                    common_p=p/100,
                    p_diff=0
                )

                conf = Config(
                    sample_size = 1000,
                    m=mh1,
                    initial_size = 10,
                    reps = 5000,
                    selection_mode = mode,
                    test_mode = test_mode,
                    dataset= dataset,
                    n=mh1,
                    common_p=p/100,
                    p_diff=pdiff/100
                )

                exp = Experiment(conf=conf, null_conf=null_conf, save_path = f"./test_res/real/{dataset}", load_path = f"./stat_res/real/{dataset}", null_path = f"./stat_res/real_h0/{dataset}")
                exp.run_parallel()
                exp.save()