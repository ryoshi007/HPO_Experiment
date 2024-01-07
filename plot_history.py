import matplotlib.pyplot as plt


def plot_score_vs_iterations(history):
    iterations = [entry[0] for entry in history]
    scores = [entry[2] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, scores, marker='o')
    plt.title('Score vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()


def plot_diff_vs_iterations(history):
    iterations = [entry[0] for entry in history]
    diffs = [entry[4] for entry in history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, diffs, marker='o', color='red')
    plt.title('Difference in Score vs Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Difference in Score')
    plt.grid(True)
    plt.show()