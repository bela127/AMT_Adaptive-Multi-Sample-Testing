import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from atm.configuration import Config
from atm.sel_runner import ExperimentReal

if __name__ == "__main__":
    filtered_data = []
    for dataset in ["arabica.clean", "occupancy.train"]:
        pdata = pd.read_csv(f"./dataset/{dataset}.csv")

        if dataset == "arabica.clean":
            pdata = pdata.drop(columns=["Company", "Variety", "Clean Cup", "Defects", "Sweetness", "Producer", "Status", "Grading Date", "Unnamed: 0", "ID", "Farm Name", "Altitude", "Region", "Total Cup Points", "Lot Number", "Mill", "ICO Number", "Expiration", "Certification Contact", "Certification Address", "Overall", "Certification Body", "Owner", "In-Country Partner", "Bag Weight", "Number of Bags"])
        elif dataset == "occupancy.train":
            pdata = pdata.drop(columns=["date"])


        print(pdata.columns)
        print([(c, pdata[c].unique().shape[0], pdata[c].dtype) for c in pdata.columns])
        print(*[(c, pdata[c].unique().shape[0], pdata[c].unique()[:10]) for c in pdata.columns],sep=",\n")

        print(pdata.describe())

        pdata.hist()
        plt.show()

        for c in pdata.columns:
            if pdata[c].dtype == "O":
                print(c, "is categorical")
                pd.DataFrame(pdata[c].value_counts()).plot.bar(title=c)
                plt.show()
        
        #test samples
        if dataset == "arabica.clean":
            for c in ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", "Moisture Percentage", "Category Two Defects"]:
                pdata[f"{c}_bin"] = pdata[c] > pdata[c].mean()

                for g in ["Processing Method", "Country of Origin", "Harvest Year", "Color"]:

                    group = pdata.groupby(g)
                    subset = group[f"{c}_bin"]
                    group_size = subset.size()
                    print(group_size)

                    def filter(group):
                        size = len(group)
                        keep_size = size > 3
                        mean = group[f"{c}_bin"].mean()
                        keep_1 = mean < 0.95
                        keep_0 = mean > 0.05
                        keep = keep_size and keep_1 and keep_0
                        return keep


                    filtered = group.filter(filter)
                    group = filtered.groupby(g)
                    subset = group[f"{c}_bin"]
                    group_size = subset.size()
                    print(group_size)

                    group_size.plot.bar(title=f"Counts for {c} by {g}")
                    plt.show()

                    subset.mean().plot.bar(title=f"{c} by {g}")
                    plt.show()

                    counts = subset.value_counts()

                    ones = counts[:,True].to_numpy()
                    zeros = counts[:,False].to_numpy()
                    contingency = np.concatenate((ones[None,...], zeros[None,...]), axis=0)
                    np.save(f"./dataset/filtered/contingency_data-{dataset}_value-{c}_by-{g}.npy", contingency)
                    filtered_data.append(f"contingency_data-{dataset}_value-{c}_by-{g}")
    
    print(filtered_data)
    


        
        
