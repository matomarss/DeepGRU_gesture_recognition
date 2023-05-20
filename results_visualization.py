import os

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib
import seaborn as sns
from results_parsing import load_from_json


def load_pca_test_results():
    final = {}

    direc = "results"
    filename_no_pca = "NO_PCA"
    filename_pca = "PCA"
    data = load_from_json(direc, filename_pca)
    final[18] = data.get("all")
    for n_comp in data.keys():
        if n_comp == "all":
            continue
        final[int(n_comp)] = data.get(n_comp)

    data = load_from_json(direc, filename_no_pca)
    final[-1] = data.get("NO_PCA")

    final = dict(sorted(final.items()))

    return final


def create_pca_test_graphs():
    data = load_pca_test_results()
    for prep in ["None", "center_norm"]:
        x = []
        y_stand = []
        y_minmax = []
        plt.clf()
        if prep == "None":
            plt.title("Bez vlastného predspracovania")
        elif prep == "center_norm":
            plt.title("S center-norm predspracovaním")
        for point in data.keys():
            if point == -1:
                pass
            else:
                x.append(int(point))
                records = data.get(point)
                for rec in records:
                    if rec.get("preprocessing") == prep:
                        if rec.get("scaler") == "StandardScaler()":
                            y_stand.append(rec.get("validation_accuracy")/100)
                        elif rec.get("scaler") == "MinMaxScaler()":
                            y_minmax.append(rec.get("validation_accuracy")/100)

        plt.scatter(x, y_stand, color="orange", label="S použitím PCA a štandardného škálovania")
        #plt.plot(x, y_stand, color="green", label="With PCA and standard scaler")
        plt.scatter(x, y_minmax, color="green", label="S použitím PCA a min-max škálovania")
        #plt.plot(x, y_minmax, color="orange", label="With PCA and minmax scaler")

        records = data.get(-1)
        for rec in records:
            if rec.get("preprocessing") == prep:
                if rec.get("scaler") == "StandardScaler()":
                    plt.axhline(y=rec.get("validation_accuracy")/100, color='orange', linestyle='-', label="S použitím štandardného škálovania bez PCA")
                elif rec.get("scaler") == "MinMaxScaler()":
                    plt.axhline(y=rec.get("validation_accuracy")/100, color='green', linestyle='-', label="S použitím min-max škálovania bez PCA")
                elif rec.get("scaler") == "None":
                    plt.axhline(y=rec.get("validation_accuracy")/100, color='black', linestyle='-', label="Bez PCA a škálovania")

        plt.xticks(x, x)
        plt.tick_params(axis='x', labelsize=6)
        plt.xlabel("Počet ponechaných komponentov po PCA")
        plt.ylabel("Validačná presnosť")
        plt.legend()
        #matplotlib.use("pgf")
        #plt.show()

        graph_direc = "graphs"
        if not os.path.exists(graph_direc):
            os.makedirs(graph_direc)

        filename = f"deepGRU_major_graph_{prep}.jpg"
        plt.savefig(os.path.join(graph_direc, filename), dpi=200, format='jpg')


def main():
    create_pca_test_graphs()


if __name__ == '__main__':
    main()