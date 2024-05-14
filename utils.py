import numpy as np

def turn_txt_into_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]
    
if __name__ == "__main__":
    # get the mean and std of the list
    xavier_cr = turn_txt_into_list("output_result/output_Xavier_sys62_data0_CR.txt")
    xavier_cr = np.array(xavier_cr, dtype=float)
    print(f"Xavier CR mean: {round(xavier_cr.mean(), 2)}, std: {round(xavier_cr.std(), 2)}")

    xavier_gcd = turn_txt_into_list("output_result/output_Xavier_sys62_data0_GCD.txt")
    print(f"Xavier GCD mean: {round(np.array(xavier_gcd, dtype=float).mean(), 2)}, std: {round(np.array(xavier_gcd, dtype=float).std(), 2)}")

    gaussian_cr = turn_txt_into_list("output_result/output_Gaussian_sys62_data0_CR.txt")
    print(f"Gaussian CR mean: {round(np.array(gaussian_cr, dtype=float).mean(), 2)}, std: {round(np.array(gaussian_cr, dtype=float).std(), 2)}")

    gaussian_gcd = turn_txt_into_list("output_result/output_Gaussian_sys62_data0_GCD.txt")
    print(f"Gaussian GCD mean: {round(np.array(gaussian_gcd, dtype=float).mean(), 2)}, std: {round(np.array(gaussian_gcd, dtype=float).std(), 2)}")
    
    data_order_xavier_cr = turn_txt_into_list("output_result/output_Xavier_data666_CR.txt")
    print(f"Data Order Xavier CR mean: {round(np.array(data_order_xavier_cr, dtype=float).mean(), 2)}, std: {round(np.array(data_order_xavier_cr, dtype=float).std(), 2)}")
    
    data_order_xavier_gcd = turn_txt_into_list("output_result/output_Xavier_data666_GCD.txt")
    print(f"Data Order Xavier GCD mean: {round(np.array(data_order_xavier_gcd, dtype=float).mean(), 2)}, std: {round(np.array(data_order_xavier_gcd, dtype=float).std(), 2)}")