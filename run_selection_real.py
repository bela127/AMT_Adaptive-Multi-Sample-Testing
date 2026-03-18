import numpy as np

from amt.configuration import Config
from amt.sel_runner import ExperimentReal

datasets = ['contingency_data-arabica.clean_value-Aroma_by-Processing Method', 'contingency_data-arabica.clean_value-Aroma_by-Country of Origin', 'contingency_data-arabica.clean_value-Aroma_by-Harvest Year', 'contingency_data-arabica.clean_value-Aroma_by-Color', 'contingency_data-arabica.clean_value-Flavor_by-Processing Method', 'contingency_data-arabica.clean_value-Flavor_by-Country of Origin', 'contingency_data-arabica.clean_value-Flavor_by-Harvest Year', 'contingency_data-arabica.clean_value-Flavor_by-Color', 'contingency_data-arabica.clean_value-Aftertaste_by-Processing Method', 'contingency_data-arabica.clean_value-Aftertaste_by-Country of Origin', 'contingency_data-arabica.clean_value-Aftertaste_by-Harvest Year', 'contingency_data-arabica.clean_value-Aftertaste_by-Color', 'contingency_data-arabica.clean_value-Acidity_by-Processing Method', 'contingency_data-arabica.clean_value-Acidity_by-Country of Origin', 'contingency_data-arabica.clean_value-Acidity_by-Harvest Year', 'contingency_data-arabica.clean_value-Acidity_by-Color', 'contingency_data-arabica.clean_value-Body_by-Processing Method', 'contingency_data-arabica.clean_value-Body_by-Country of Origin', 'contingency_data-arabica.clean_value-Body_by-Harvest Year', 'contingency_data-arabica.clean_value-Body_by-Color', 'contingency_data-arabica.clean_value-Balance_by-Processing Method', 'contingency_data-arabica.clean_value-Balance_by-Country of Origin', 'contingency_data-arabica.clean_value-Balance_by-Harvest Year', 'contingency_data-arabica.clean_value-Balance_by-Color', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Processing Method', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Country of Origin', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Harvest Year', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Color', 'contingency_data-arabica.clean_value-Category Two Defects_by-Processing Method', 'contingency_data-arabica.clean_value-Category Two Defects_by-Country of Origin', 'contingency_data-arabica.clean_value-Category Two Defects_by-Harvest Year', 'contingency_data-arabica.clean_value-Category Two Defects_by-Color']

datasets = ['contingency_data-arabica.clean_value-Moisture Percentage_by-Harvest Year', 'contingency_data-arabica.clean_value-Acidity_by-Harvest Year', 'contingency_data-arabica.clean_value-Category Two Defects_by-Harvest Year', 'contingency_data-arabica.clean_value-Flavor_by-Color', 'contingency_data-arabica.clean_value-Acidity_by-Color', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Color', 'contingency_data-arabica.clean_value-Flavor_by-Country of Origin', 'contingency_data-arabica.clean_value-Category Two Defects_by-Country of Origin', 'contingency_data-arabica.clean_value-Aftertaste_by-Country of Origin']

if __name__ == "__main__":
    for dataset in datasets:
        for mode in ["mean.slow", "equal", "beta.med"]:#  "ts.5", "ts", "equal", "beta", "means", "mean.slow"
            for m in [0]: # m=0 means H0, m=1 means H1 where m=1 internally sets m to the number of samples N contained in the dataset
                conf = Config(
                    sample_size = 1000,
                    m=m,
                    initial_size = 10,
                    reps = 5000,
                    selection_mode = mode,
                    dataset= dataset,
                )


                exp = ExperimentReal(conf=conf, save_path=f"./coin_res/real_h0/{dataset}")
                exp.run_parallel()
                exp.save()
