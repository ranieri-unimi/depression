import pandas as pd
import numpy as np


def erde_evaluation(goldenTruth_path, algorithmResult_path, o):
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
    P_true = len(merged_data[t_risk == 1])

    # Count of how many positive cases the system decided there were
    P_hat = len(merged_data[risk_d == 1])

    # Count of how many of them are actually true positive cases
    TP = len(merged_data[(t_risk == 1) & (risk_d == 1)])

    # Total count of users
    N = len(merged_data)

    # ERDE calculus
    for i in range(N):
        if risk_d[i] == 1 and t_risk[i] == 0:
            erde.iat[i] = float(P_true) / N
        elif risk_d[i] == 0 and t_risk[i] == 1:
            erde.iat[i] = 1.0
        elif risk_d[i] == 1 and t_risk[i] == 1:
            erde.iat[i] = 1.0 - (1.0 / (1.0 + np.exp(k[i] - o)))
        elif risk_d[i] == 0 and t_risk[i] == 0:
            erde.iat[i] = 0.0

    # Calculus of F1, Precision, Recall and global ERDE
    precision = float(TP) / P_hat
    recall = float(TP) / P_true
    F1 = 2 * (precision * recall) / (precision + recall)
    erde_global = erde.mean() * 100

    indiv_erde = merged_data.loc[:, ["subj_id", "erde"]]
    print(indiv_erde.to_string())
    print("Global ERDE (with o = %d): %.2f" % (o, erde_global), "%")
    print("F1: %.2f" % F1)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)


def erde_mem(predictions, labels, delays, order=50):

    yy = list(zip(predictions, labels))

    P_TRUE = sum(labels)
    P_HAT = sum(predictions)

    TP = yy.count((1, 1))
    N = len(yy)

    # https://tec.citius.usc.es/ir/pdf/CLEF16_paper.pdf
    erde = list()
    for i in range(N):
        y_hat, y_true = yy[i]
        match (y_hat, y_true):
            case (0, 0):
                loss = 0
            case (1, 0):
                loss = P_TRUE / N
            case (0, 1):
                loss = 1
            case (1, 1):
                penalty = np.exp(delays[i] - order)
                loss = 1 - (1 / (1 + penalty))

        erde.append(loss)

    # Calculus of F1, Precision, Recall and global ERDE
    precision = TP / P_HAT
    recall = TP / P_TRUE
    F1 = 2 * (precision * recall) / (precision + recall)
    erde_global = sum(erde)/len(erde)

    print(f"Global ERDE({order}): {erde_global}")
    print(f"F1: {F1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


def metrics(predictions, labels):

    yy = list(zip(predictions, labels))

    P_TRUE = sum(labels)
    P_HAT = sum(predictions)

    TP = yy.count((1, 1))
    N = len(yy)

    precision = TP / P_HAT
    recall = TP / P_TRUE
    F1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"F1: {F1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")