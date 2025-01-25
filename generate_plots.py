import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    data_standard = list(np.array(read_list_from_log("results_benchmark/2025_01_24T15_19_10_standard.log")) / 10000000)
    data_baldwinian = list(np.array(read_list_from_log("results_benchmark/2025_01_24T16_29_48_baldwinian.log")) / 10000000)
    data_lamarckian = list(np.array(read_list_from_log("results_benchmark/2025_01_24T16_30_47_lamarckian.log")) / 10000000)

    # Ensure all lists have the same length
    data_baldwinian = pad_list_to_length(data_baldwinian, len(data_standard))
    data_lamarckian = pad_list_to_length(data_lamarckian, len(data_standard))

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    # Plot for Standard
    axs[0, 0].plot(data_standard)
    axs[0, 0].set_title('Standard')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Fittest Individual')
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Plot for Baldwinian
    axs[0, 1].plot(data_baldwinian)
    axs[0, 1].set_title('Baldwinian')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Fittest Individual')
    axs[0, 1].axhline(y=4.4, color='r', linestyle='--', label='Best Known')
    axs[0, 1].legend()
    axs[0, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Plot for Lamarckian
    axs[1, 0].plot(data_lamarckian)
    axs[1, 0].set_title('Lamarckian')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('Fittest Individual')
    axs[1, 0].axhline(y=4.4, color='r', linestyle='--', label='Best Known')
    axs[1, 0].legend()
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Combined plot
    axs[1, 1].plot(data_standard, label='Standard')
    axs[1, 1].plot(data_baldwinian, label='Baldwinian')
    axs[1, 1].plot(data_lamarckian, label='Lamarckian')
    axs[1, 1].set_title('Combined')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Fittest Individual')
    axs[1, 1].axhline(y=4.4, color='r', linestyle='--', label='Best Known')
    axs[1, 1].legend()
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    # Save plot to disk
    plt.savefig(f"results_benchmark/comparison.png")

    plt.show()

    # Best result
    data_best = list(np.array(read_list_from_log("best_result/best_result.log")) / 10000000)

    # Separate plot for Best result
    fig_best, ax_best = plt.subplots(figsize=(10, 6))
    ax_best.plot(data_best)
    ax_best.set_title('Best')
    ax_best.set_xlabel('Generation')
    ax_best.set_ylabel('Fittest Individual')
    ax_best.axhline(y=4.475, color='r', linestyle='--', label='Best Known')
    ax_best.legend()
    ax_best.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    # Save plot to disk
    plt.savefig(f"results_benchmark/best_result_plot.png")
    plt.show()


def pad_list_to_length(data_list, target_length):
    if len(data_list) < target_length:
        last_value = data_list[-1]
        data_list.extend([last_value] * (target_length - len(data_list)))
    return data_list


def read_list_from_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming the list is on the second line
        data_list = ast.literal_eval(lines[1].strip())
    return data_list


if __name__ == '__main__':
    main()
