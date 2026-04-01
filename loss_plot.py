import csv

import matplotlib.pyplot as plt


def moving_average(data, window_size=5):
    return [
        sum(data[max(0, i - window_size) : i + 1])
        / (i + 1 if i < window_size else window_size)
        for i in range(len(data))
    ]


def main(use_mva=True, path="loss.csv"):
    epochs = []
    losses = []

    # -- load csv
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))

    # -- plot
    plt.figure()

    if use_mva:
        smoothed = moving_average(losses, window_size=5)
        plt.plot(epochs, losses, alpha=0.4, label="Raw Loss")
        plt.plot(epochs, smoothed, linewidth=2, label="Smoothed Loss")
        plt.legend()
        plt.title("Training Loss (Smoothed)")
    else:
        plt.plot(epochs, losses, marker="o")
        plt.title("Training Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # True = with smoothing, False = raw only
    main(use_mva=True, path="sup/loss.csv")
