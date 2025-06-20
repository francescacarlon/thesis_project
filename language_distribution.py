import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator

# List of native languages (cleaned) for L and CS background
languages_L = [
    "Ukrainian", "Ukrainian", "Greek", "Portuguese",
    "Italian", "Turkish", "Spanish", "Catalan", "Italian", "Italian", "Portuguese",
    "Persian", "English", "German", "Greek", "German",
    "English", "Catalan", "Spanish", "Catalan"
]

languages_CS = [
    "Hindi", "Chinese","German", "Vietnamese", "English", "French", "German", "Persian", 
    "Bengali", "Arabic", "Marathi", "English", "Tamil", "Tamil", "Hindi", "English", 
    "Konkani", "Nepali", "Hindi", "English", "Hindi", "Guarajati", "Kutchi"
]

counts = Counter(languages_CS)
sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 6))
# plt.bar(sorted_counts.keys(), sorted_counts.values(), color='mediumpurple')
plt.bar(sorted_counts.keys(), sorted_counts.values(), color='teal')
plt.title('Distribution of Native Languages')
plt.xlabel('Language')
plt.ylabel('Number of Speakers')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 🔹 Force y-axis to use only integer values
ax = plt.gca()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("language_distribution.png")
plt.show()
