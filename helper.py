import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
palette = sns.color_palette("cubehelix", 5)
print(palette.as_hex())
sns.palplot(palette)
plt.show()