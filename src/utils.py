
import matplotlib.pyplot as plt

def secure_sort(x_enc):
    result = x_enc
    n = len(result)

    for i in range(n - 1):
        for j in range(0, n - i - 1):
            compare = result[j] <= result[j + 1]   # 1 if left <= right
            left = result[j]
            right = result[j + 1]

            new_left  = left * compare + right * (1 - compare)
            new_right = right * compare + left  * (1 - compare)

            result[j]     = new_left
            result[j + 1] = new_right

    return result

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
