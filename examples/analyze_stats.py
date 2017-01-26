from os import listdir, path

target_dir = "C:\\workspace\\data\\learning_data"

folds_tumor = {}
folds_total = {}

for k in range(0, 3):
	folds_tumor[k] = 0
	folds_total[k] = 0

with open(path.join(target_dir, "stats.txt"), "r") as file_handle:
	for line in file_handle.readlines():
		if "Total" in line:
			continue
		parts = line.split(" ")
		fold = int(parts[0].split("-")[1])
		folds_tumor[fold] += int(parts[1])
		folds_total[fold] += int(parts[2])

print(folds_tumor)
print(folds_total)