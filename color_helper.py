import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
palette = sns.color_palette("cubehelix", 6)
print(palette.as_hex())
sns.palplot(palette)
plt.show()