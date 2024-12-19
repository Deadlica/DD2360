import matplotlib.pyplot as plt

x = [4, 8, 12, 16, 20, 24, 28, 32]
y = [0.015787, 0.017734, 0.015787, 0.015756, 0.014816, 0.014864, 0.015008, 0.016279]

plt.figure(figsize=(8, 6))
plt.bar(x, y, color="blue")
plt.xlabel("Number of segments", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)
plt.title("Execution Time on Vector of Size 100000000", fontsize=14)
plt.xticks(x, fontsize=10)
plt.tight_layout()
plt.savefig("segments.png")
