import csv

TEST_RESULT_PATH = "../ensemble_scripts/dets_numpy/test/results/result_ensemble_fsr.csv"
THRESH = 0.0

with open(TEST_RESULT_PATH, "r") as f:
    reader = csv.reader(f, delimiter=",")
    filtered_results = []
    for i, line in enumerate(reader):
        if float(line[2]) > THRESH:
            line[0] = line[0].zfill(8)
            filtered_results.append(line)


print("Writing filtered results to csv")
with open("../ensemble_scripts/dets_numpy/test/results/filtered_result_ensemble_fsr_" + str(THRESH) + ".csv", 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerows(filtered_results)