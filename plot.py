
import matplotlib.pyplot as plt

# Plotter function
def plot_history(history, train: str, val: str):
    train_x = history.history[train]
    val_x = history.history[val]

    plt.plot(train_x)
    plt.plot(val_x)
    plt.legend([f"train_{train}", val])
    plt.xlabel("Epochs")
    plt.ylabel(train.upper())
    plt.show()