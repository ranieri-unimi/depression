import pandas as pd
import numpy as np
import argparse

# To calculate the metrics of the classification algorithm
def erde_evaluation(goldenTruth_path, algorithmResult_path, o):
    # Transform files into tables
    try:
        data_golden = pd.read_csv(
            goldenTruth_path, sep="\t", header=None, names=["subj_id", "true_risk"]
        )
        data_result = pd.read_csv(
            algorithmResult_path,
            sep=" ",
            header=None,
            names=["subj_id", "risk_decision", "delay"],
        )

        # Merge tables (data) on common field 'subj_id' to compare the true risk and the decision risk
        merged_data = data_golden.merge(data_result, on="subj_id", how="left")

        # Add column to store the idividual ERDE of each user
        merged_data.insert(loc=len(merged_data.columns), column="erde", value=1.0)

        # Variables
        risk_d = merged_data["risk_decision"]
        t_risk = merged_data["true_risk"]
        k = merged_data["delay"]
        erde = merged_data["erde"]

        # Count of how many true positives there are
        true_pos = len(merged_data[t_risk == 1])

        # Count of how many positive cases the system decided there were
        pos_decisions = len(merged_data[risk_d == 1])

        # Count of how many of them are actually true positive cases
        pos_hits = len(merged_data[(t_risk == 1) & (risk_d == 1)])

        # Total count of users
        total_users = len(merged_data)

        # ERDE calculus
        for i in range(total_users):
            if risk_d[i] == 1 and t_risk[i] == 0:
                erde.ix[i] = float(true_pos) / total_users
            elif risk_d[i] == 0 and t_risk[i] == 1:
                erde.ix[i] = 1.0
            elif risk_d[i] == 1 and t_risk[i] == 1:
                erde.ix[i] = 1.0 - (1.0 / (1.0 + np.exp(k[i] - o)))
            elif risk_d[i] == 0 and t_risk[i] == 0:
                erde.ix[i] = 0.0

        # Calculus of F1, Precision, Recall and global ERDE
        precision = float(pos_hits) / pos_decisions
        recall = float(pos_hits) / true_pos
        F1 = 2 * (precision * recall) / (precision + recall)
        erde_global = erde.mean() * 100

        indiv_erde = merged_data.ix[:, ["subj_id", "erde"]]
        print(indiv_erde.to_string())
        print("Global ERDE (with o = %d): %.2f" % (o, erde_global), "%")
        print("F1: %.2f" % F1)
        print("Precision: %.2f" % precision)
        print("Recall: %.2f" % recall)

    except:
        print("Some file or directory doesn't exist or an error has occurred")


pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument(
    "-gpath",
    help="(Obligatory) path to golden truth file.",
    required=True,
    nargs=1,
    dest="gpath",
)
parser.add_argument(
    "-ppath",
    help="(Obligatory) path to prediction file from a system.",
    required=True,
    nargs=1,
    dest="ppath",
)
parser.add_argument(
    "-o", help="(Obligatory) o parameter.", required=True, nargs=1, dest="o"
)

args = parser.parse_args()

gpath = args.gpath[0]
ppath = args.ppath[0]
o = int(args.o[0])


erde_evaluation(gpath, ppath, o)
