import matplotlib.pyplot as plt

def secure_sort(x_enc, y_enc):
    n = len(x_enc)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            compare = x_enc[j] <= x_enc[j + 1]   # 1 if left <= right
            left = x_enc[j]
            right = x_enc[j + 1]
            left_y = y_enc[j]
            right_y = y_enc[j + 1]

            new_left  = left * compare + right * (1 - compare)
            new_right = right * compare + left  * (1 - compare)

            #new_left_y =

            x_enc[j]     = new_left
            x_enc[j + 1] = new_right

    return x_enc

def plot_roc(fpr, tpr, name="ROC"):
    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, label=f"{name}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    #plt.grid(True)
    plt.show()
