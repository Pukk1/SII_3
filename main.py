import math
import random

import matplotlib.pyplot as plt

PARAMS_NUMBER = 6


def read_data():
    T = []
    selected_columns = random.sample(list(range(1, 32)), PARAMS_NUMBER)
    with open('lab3.csv') as file:
        for line in file.readlines():
            dataset_seperated_columns = line.split(';')
            T.append(
                (
                    [int(dataset_seperated_columns[c]) for c in selected_columns],
                    int(dataset_seperated_columns[32]),
                )
            )
    return T


def select_all_classes(T):
    classes = []
    for _, C in T:
        classes.append(C)
    return classes


def freq(C, T):
    filtered = []
    for i in T:
        if i[1] == C:
            filtered.append(i)
    return len(filtered)


def split_by_X(T, X):
    T_parts = {}
    for params, clazz in T:
        parameter = params[X]
        if parameter not in T_parts:
            T_parts[parameter] = []
        T_parts[parameter].append((params, clazz))
    return T_parts


def info(T):
    return -sum([freq(C, T) / len(T) * math.log2(freq(C, T) / len(T)) for C in select_all_classes(T)])


def info_x(T, X):
    return sum([len(Ti) / len(T) * info(Ti) for Ti in split_by_X(T, X).values()])


def split_info_x(T, X):
    return -sum([len(Ti) / len(T) * math.log2(len(Ti) / len(T)) for Ti in split_by_X(T, X).values()])


def gain_ratio(T, X):
    divider = split_info_x(T, X)
    if divider == 0:
        return -math.inf
    return (info(T) - info_x(T, X)) / divider


def select_best_x(T):
    best_X = 0
    for X in range(PARAMS_NUMBER):
        if gain_ratio(T, X) > gain_ratio(T, best_X):
            best_X = X
    return best_X


def build_tree_node(T):
    X = select_best_x(T)
    T_parts = split_by_X(T, X)
    if len(T_parts.keys()) == 1:
        return select_all_classes(T)[0]
    return {X: {clazz: build_tree_node(T_parts[clazz]) for clazz in T_parts.keys()}}


def check(tree, params):
    X = list(tree.keys())[0]
    C = tree[X][params[X]]
    if type(C) == int:
        return C
    else:
        return check(C, params)


def check_all(T):
    results = []
    for params, real_clazz in T:
        check_res = check(tree, params)
        print(f'By check: {check_res} and true is: {real_clazz}')
        results.append((check_res, real_clazz))
    return results


def apr(results, separator=3):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for checked_class, real_class in results:
        if (checked_class >= separator) == (real_class >= 3):
            if checked_class >= separator:
                tp += 1
            else:
                tn += 1
        else:
            if checked_class >= separator:
                fp += 1
            else:
                fn += 1
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall


def print_AUC_ROC(results):
    results.sort(key=lambda item: -item[0])
    x = [0]
    y = [0]
    for i in range(0, len(results)):
        if results[i][1] >= PARAMS_NUMBER:
            x.append(x[-1])
            y.append(y[-1] + 1)
        else:
            x.append(x[-1] + 1)
            y.append(y[-1])
    plt.plot(x, y)
    # plt.plot([x[0], x[len(x) - 1]], [y[0], y[len(y) - 1]])
    plt.title("AUC-ROC")
    plt.show()


def print_AUC_PR(results):
    x = []
    y = []
    last_recall = None
    for separator in range(1, 7):
        _, precision, recall = apr(results, separator)
        if last_recall is not None:
            x.append(last_recall)
            y.append(precision)
        x.append(recall)
        y.append(precision)
        last_recall = recall
    plt.plot(x, y)
    plt.title("AUC-PR")
    plt.show()


T = read_data()
tree = build_tree_node(T)
results = check_all(T)
accuracy, precision, recall = apr(results)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print_AUC_ROC(results)
print_AUC_PR(results)
