import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def turn_txt_into_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def show_mean_std(file_path, title: str):
    list_data = turn_txt_into_list(file_path)
    list_data = np.array(list_data, dtype=float)
    print(title)
    print(f"Mean: {round(list_data.mean(), 2)}, std: {round(list_data.std(), 2)}")
    return list_data

def show_results():
    # get the mean and std of the list
    xavier_CR = show_mean_std("output_result/output_Xavier_sys62_data0_CR.txt", "Xavier CR")
    xavier_GCD = show_mean_std("output_result/output_Xavier_sys62_data0_GCD.txt", "Xavier GCD")
    print()
    gaussian_CR = show_mean_std("output_result/output_Gaussian_sys62_data0_CR.txt", "Gaussian CR")
    gaussian_GCD = show_mean_std("output_result/output_Gaussian_sys62_data0_GCD.txt", "Gaussian GCD")
    print()
    data_order_xavier_CR = show_mean_std("output_result/output_Xavier_data666_CR.txt", "Data Order Xavier CR")    
    data_order_xavier_GCD = show_mean_std("output_result/output_Xavier_data666_GCD.txt", "Data Order Xavier GCD")    

    # plot the histogram
    histogram(xavier_CR, "output_result/histogram_xavier_CR.png", "CR", "Xavier")
    histogram(xavier_GCD, "output_result/histogram_xavier_GCD.png", "GCD", "Xavier")
    histogram(gaussian_CR, "output_result/histogram_gaussian_CR.png", "CR", "Gaussian")
    histogram(gaussian_GCD, "output_result/histogram_gaussian_GCD.png", "GCD", "Gaussian")
    histogram(xavier_CR, "output_result/histogram_data_xavier_CR.png", "CR", "Seed 62")
    histogram(data_order_xavier_CR, "output_result/histogram_data_order_xavier_CR.png", "CR", "Seed 666")
    histogram(xavier_GCD, "output_result/histogram_data_xavier_GCD.png", "GCD", "Seed 62")
    histogram(data_order_xavier_GCD, "output_result/histogram_data_order_xavier_GCD.png", "GCD", "Seed 666")

# def QQPlot(data1: np.ndarray, save_path: str):
#     fig = sm.qqplot(data1, line ='45')  
#     plt.savefig(save_path)

def histogram(data: np.ndarray, save_path: str, task: str, init: str):
    mean = data.mean()
    plt.hist(data, bins = 20, color = 'blue', edgecolor = 'black')
    # draw mean line
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
    plt.title('Histogram of ' + task + ' - ' + init)
    plt.xlabel('Iterations')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def main():
    show_results()
    
if __name__ == "__main__":
    main()