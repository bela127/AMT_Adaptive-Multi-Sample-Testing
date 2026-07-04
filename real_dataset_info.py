import numpy as np
datasets = ['contingency_data-arabica.clean_value-Aroma_by-Processing Method', 'contingency_data-arabica.clean_value-Aroma_by-Country of Origin', 'contingency_data-arabica.clean_value-Aroma_by-Harvest Year', 'contingency_data-arabica.clean_value-Aroma_by-Color', 'contingency_data-arabica.clean_value-Flavor_by-Processing Method', 'contingency_data-arabica.clean_value-Flavor_by-Country of Origin', 'contingency_data-arabica.clean_value-Flavor_by-Harvest Year', 'contingency_data-arabica.clean_value-Flavor_by-Color', 'contingency_data-arabica.clean_value-Aftertaste_by-Processing Method', 'contingency_data-arabica.clean_value-Aftertaste_by-Country of Origin', 'contingency_data-arabica.clean_value-Aftertaste_by-Harvest Year', 'contingency_data-arabica.clean_value-Aftertaste_by-Color', 'contingency_data-arabica.clean_value-Acidity_by-Processing Method', 'contingency_data-arabica.clean_value-Acidity_by-Country of Origin', 'contingency_data-arabica.clean_value-Acidity_by-Harvest Year', 'contingency_data-arabica.clean_value-Acidity_by-Color', 'contingency_data-arabica.clean_value-Body_by-Processing Method', 'contingency_data-arabica.clean_value-Body_by-Country of Origin', 'contingency_data-arabica.clean_value-Body_by-Harvest Year', 'contingency_data-arabica.clean_value-Body_by-Color', 'contingency_data-arabica.clean_value-Balance_by-Processing Method', 'contingency_data-arabica.clean_value-Balance_by-Country of Origin', 'contingency_data-arabica.clean_value-Balance_by-Harvest Year', 'contingency_data-arabica.clean_value-Balance_by-Color', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Processing Method', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Country of Origin', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Harvest Year', 'contingency_data-arabica.clean_value-Moisture Percentage_by-Color', 'contingency_data-arabica.clean_value-Category Two Defects_by-Processing Method', 'contingency_data-arabica.clean_value-Category Two Defects_by-Country of Origin', 'contingency_data-arabica.clean_value-Category Two Defects_by-Harvest Year', 'contingency_data-arabica.clean_value-Category Two Defects_by-Color']

data_path = "./real_datasets"

def load_data(dataset):
        contingency = np.load(f"{data_path}/{dataset}.npy")
        return contingency

if __name__ == "__main__":
    for dataset in datasets:
        contingency = load_data(dataset)
        mean = contingency[0] / np.sum(contingency, axis=0)
        n = mean.shape[0]
        common_p = np.mean(mean)
        p_diff = mean.max() - mean.min()
        print(f"dataset: {dataset}, n: {n}, common_p: {common_p}, p_diff: {p_diff}")
