import os
import json


def dump_to_json(path, filename, data):
    path = os.path.join(path, filename) + ".json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_from_json(path, filename):
    path = os.path.join(path, filename) + ".json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def parse(path, pca_scan):
    print("Parsing....")

    parameter_line = "Evaluated preprocessing is"
    accuracy_line = "Average accuracy"

    json_dic = {}
    lc = 0
    with open(path, "r") as f:
        while line := f.readline():
            lc += 1
            print("Line reading step n.{}".format(lc))
            if parameter_line in line:
                "pca: \"PCA(n_components=8)\""
                if "center-norm: \"False\"" in line:
                    preprocessing = "None"
                elif "center-norm: \"True\"" in line:
                    preprocessing = "center_norm"
                else:
                    raise Exception("Invalid format detected")

                if "scaler: \"StandardScaler()\"" in line:
                    scaler = "StandardScaler()"
                elif "scaler: \"MinMaxScaler()\"" in line:
                    scaler = "MinMaxScaler()"
                elif "scaler: \"None\"" in line:
                    scaler = "None"
                else:
                    raise Exception("Invalid format detected")

                if pca_scan is True:
                    if "pca: \"None\"" in line:
                        raise Exception("Invalid format detected")
                    elif "pca: \"PCA(n_components=2)\"" in line:
                        pca = "2"
                    elif "pca: \"PCA(n_components=4)\"" in line:
                        pca = "4"
                    elif "pca: \"PCA(n_components=5)\"" in line:
                        pca = "5"
                    elif "pca: \"PCA(n_components=8)\"" in line:
                        pca = "8"
                    elif "pca: \"PCA(n_components=11)\"" in line:
                        pca = "11"
                    elif "pca: \"PCA(n_components=12)\"" in line:
                        pca = "12"
                    elif "pca: \"PCA(n_components=18)\"" in line:
                        pca = "all"
                    else:
                        raise Exception("Invalid format detected")
                else:
                    if "pca: \"None\"" in line:
                        pca = "NO_PCA"
                    else:
                        raise Exception("Invalid format detected")

                line = f.readline()
                if accuracy_line not in line:
                    raise Exception("Invalid format detected")
                else:
                    line = line.split(": ")
                    accuracy = float(line[1])

                data = {"preprocessing": preprocessing, "scaler": scaler, "validation_accuracy": accuracy}
                if pca not in json_dic.keys():
                    json_dic[pca] = []
                json_dic[pca].append(data)

    if pca_scan:
        filename = "PCA"
    else:
        filename = "NO_PCA"
    dump_to_json("C:/Users/matom/OneDrive/Počítač/skola3/gestures_recognition/neural_nets/DeepGRU_gesture_recognition/results", filename, json_dic)


def main():
    path = "C:/Users/matom/OneDrive/Počítač/skola3/gestures_recognition/neural_nets/DeepGRU_gesture_recognition/logs"
    filename = "modifi.txt"
    path = os.path.join(path, filename)

    parse(path, True)

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()