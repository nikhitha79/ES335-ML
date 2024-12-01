import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tree.base import DecisionTree

def create_data(N, M, task_type):
    P = 10  # number of unique values in feature column in dataset
    if task_type == "DIDO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif task_type == "DIRO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    elif task_type == "RIDO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif task_type == "RIRO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    return X, y

def measure_learning_time(tree, X, y):
    start_time = datetime.now()
    tree.fit(X, y)
    end_time = datetime.now()
    return (end_time - start_time).total_seconds()

def measure_predicting_time(tree, X):
    start_time = datetime.now()
    tree.predict(X)
    end_time = datetime.now()
    return (end_time - start_time).total_seconds()

def fun(N=[3,], M=[5,], method="DT", task=["DIDO", ], case=None):
    output = np.zeros((len(task), len(N), len(M), 4))
    for i in task:
        for j in N:
            for k in M:
                # generate data
                X, y = create_data(j, k, i)
                # my DT implementation
                if method == "DT":
                    tree = DecisionTree()
                # sklearn DT implementation
                else:
                    if i in {"DIDO", "RIDO"}:
                        tree = DecisionTreeClassifier(criterion="entropy")
                    elif i in {"DIRO", "RIRO"}:
                        tree = DecisionTreeRegressor()
                # finding time for learning and predicting
                learning_time = measure_learning_time(tree, X, y)
                predicting_time = measure_predicting_time(tree, X)
                output[task.index(i), N.index(j), M.index(k)] = np.array([j, k, learning_time, predicting_time])
    plot_learning_and_predicting(output, N, M, task, method, case)
    return output

save_folder = "plots"
os.makedirs(save_folder, exist_ok=True)
def plot_learning_and_predicting(output, N, M, task, method, case):
    for measure in ["learning", "predicting"]:
        plt.figure()

        if len(N) > 1:
            for t in task:
                plt.plot(output[task.index(t), :, 0, 0], output[task.index(t), :, 0, 2 if measure == "learning" else 3], label=t)

            plt.title(f"{method} : {measure} plot (Varied N)")
            plt.xlabel("N")
            plt.ylabel("Time (secs)")
            plot_name = f"{method}_{measure}_{'Varied N'}.png" 

        else:  # if M is varied
            for t in task:
                plt.plot(output[task.index(t), 0, :, 1], output[task.index(t), 0, :, 2 if measure == "learning" else 3], label=t)

            plt.title(f"{method} : {measure} plot (Varied M)")
            plt.xlabel("M")
            plt.ylabel("Time (secs)")
            plot_name = f"{method}_{measure}_{'Varied M'}.png" 

        plt.legend()
        
        plot_path = os.path.join(save_folder, plot_name)
        plt.savefig(plot_path)
        plt.show()


print("case 1 : Decision tree & varying No. of samples")
print("case 2 : Sklearn & varying No. of samples")
print("case 3 : Decision tree & varying No. of features")
print("case 4 : Sklearn & varying No. of features")

case = int(input("Enter the case (1, 2, 3, or 4): "))
if case == 1:
    fun(N=list(range(10, 50, 5)), task=["DIDO", "DIRO", "RIDO", "RIRO"], method="DT", case=case)
elif case == 2:
    fun(N=list(range(10, 50, 5)), task=["DIDO", "DIRO", "RIDO", "RIRO"], method="Sklearn", case=case)
elif case == 3:
    fun(M=list(range(3, 10, 2)), task=["DIDO", "DIRO", "RIDO", "RIRO"], method="DT", case=case)
elif case == 4:
    fun(M=list(range(3, 10, 2)), task=["DIDO", "DIRO", "RIDO", "RIRO"], method="Sklearn", case=case)
else:
    print("Invalid case.")

