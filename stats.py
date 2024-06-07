import numpy as np
from scipy.stats import t as t_table, f as f_table
from scipy.stats import ttest_ind
from utils import turn_txt_into_list

def my_two_sample_t_test(data_list1: np.ndarray, data_list2: np.ndarray, siginificance_level: float, is_two_tailed: bool = False, D0 = 0):
    # mean
    mean1 = np.mean(data_list1)
    mean2 = np.mean(data_list2)
    # variance
    var1 = np.var(data_list1, ddof=1)
    var2 = np.var(data_list2, ddof=1)
    
    # t-test
    n1 = len(data_list1)
    n2 = len(data_list2)
    # f-test first before decide df
    significance_level_f = 0.05
    is_two_tailed_f = True
    F_stat, F_cdf, diffVar = F_test(data_list1, data_list2, significance_level_f, is_two_tailed_f)
    if (diffVar):
        c = (var1 / n1) / \
            (var1 / n1 + var2 / n2)
        df = (n1 - 1) * (n2 - 1) / \
            ((1 - c) ** 2 * (n1 - 1) + c ** 2 * (n2 - 1))
        # df < (n1 + n2 - 2) if var_A != var_B
        SE = np.sqrt(var1 / n1 + var2 / n2)
    else:
        df = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df)
        SE = sp * np.sqrt(1 / n1 + 1 / n2)
    
    # p-value
    t_stat = (mean1 - mean2 - D0) / SE
    if is_two_tailed:
        if t_stat < 0:
            p_value = 2 * t_table.cdf(t_stat, df)
        else:
            p_value = 2 * (1 - t_table.cdf(t_stat, df))
    else:
        if t_stat < 0:
            p_value = t_table.cdf(abs(t_stat), df)
        else:
            p_value = 1 - t_table.cdf(abs(t_stat), df)

    # t-critical & margin error
    t_critical = t_table.ppf(1 - siginificance_level / 2, df) if is_two_tailed else t_table.ppf(1 - siginificance_level, df)
    margin_err = t_critical * SE # margin error > 0

    lower = (mean1 - mean2 - D0) - margin_err
    upper = (mean1 - mean2 - D0) + margin_err

    return t_stat, p_value, p_value < siginificance_level, lower, upper

def F_test(data_list1: np.ndarray, data_list2: np.ndarray, siginificant_level: float, is_two_tailed: bool):
    # variance
    var1 = np.var(data_list1, ddof=1)
    var2 = np.var(data_list2, ddof=1)
    # F-test
    F_stat = var1 / var2
    df1 = len(data_list1) - 1
    df2 = len(data_list2) - 1

    if is_two_tailed:
        lower, upper = f_table.ppf([siginificant_level / 2, 1 - siginificant_level / 2], df1, df2)
        return F_stat, f_table.cdf(F_stat, df1, df2), F_stat < lower or F_stat > upper
    else:
        upper = f_table.ppf(1 - siginificant_level, df1, df2)
        print("Upper:", upper)
        return F_stat, 1 - f_table.cdf(F_stat, df1, df2), F_stat > upper

def two_sample_t_test(data1: np.ndarray, data2: np.ndarray, siginificance_level: float, alternative: str = 'two-sided'):
    # f-test
    f_stat, p_value, diffVar = F_test(data1, data2, 0.05, True)
    if (diffVar):
        # t-test
        print("Variances are different")
        t_stat, p_value = ttest_ind(data1, data2, equal_var=False, alternative=alternative)
    else:
        t_stat, p_value = ttest_ind(data1, data2, equal_var=True, alternative=alternative)

    return t_stat, p_value, p_value < siginificance_level

def CR():
    gaussian_CR = turn_txt_into_list("output_result/output_Gaussian_sys62_data0_CR.txt")
    print("Mean: ", np.mean(gaussian_CR), "Std: ", np.std(gaussian_CR))
    xavier_CR = turn_txt_into_list("output_result/output_Xavier_sys62_data0_CR.txt")
    print("Mean: ", np.mean(xavier_CR), "Std: ", np.std(xavier_CR))

    f_stat, p_value, is_reject = F_test(gaussian_CR, xavier_CR, 0.05, False)
    print("F-statistic:", f_stat, "p-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject = two_sample_t_test(gaussian_CR, xavier_CR, 0.05)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject, lower, upper = my_two_sample_t_test(gaussian_CR, xavier_CR, 0.05, False)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject, "Lower:", lower, "Upper:", upper)

def GCD():
    gaussian_GCD = turn_txt_into_list("output_result/output_Gaussian_sys62_data0_GCD.txt")
    print("Mean: ", np.mean(gaussian_GCD), "Std: ", np.std(gaussian_GCD))
    xavier_GCD = turn_txt_into_list("output_result/output_Xavier_sys62_data0_GCD.txt")
    print("Mean: ", np.mean(xavier_GCD), "Std: ", np.std(xavier_GCD))

    f_stat, p_value, is_reject = F_test(gaussian_GCD, xavier_GCD, 0.05, False)
    print("F-statistic:", f_stat, "p-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject = two_sample_t_test(gaussian_GCD, xavier_GCD, 0.05)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject, lower, upper = my_two_sample_t_test(gaussian_GCD, xavier_GCD, 0.05, False)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject, "Lower:", lower, "Upper:", upper)

def DataOrder_CR():
    xavier_CR = turn_txt_into_list("output_result/output_Xavier_sys62_data0_CR.txt")
    print("Mean: ", np.mean(xavier_CR), "Std: ", np.std(xavier_CR))
    data_order_xavier_CR = turn_txt_into_list("output_result/output_Xavier_data666_CR.txt")
    print("Mean: ", np.mean(data_order_xavier_CR), "Std: ", np.std(data_order_xavier_CR))

    f_stat, p_value, is_reject = F_test(xavier_CR, data_order_xavier_CR, 0.05, False)
    print("F-statistic:", f_stat, "p-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject = two_sample_t_test(xavier_CR, data_order_xavier_CR, 0.05)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject, lower, upper = my_two_sample_t_test(xavier_CR, data_order_xavier_CR, 0.05, False)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject, "Lower:", lower, "Upper:", upper)

def DataOrder_GCD():
    xavier_GCD = turn_txt_into_list("output_result/output_Xavier_sys62_data0_GCD.txt")
    print("Mean: ", np.mean(xavier_GCD), "Std: ", np.std(xavier_GCD))
    data_order_xavier_GCD = turn_txt_into_list("output_result/output_Xavier_data666_GCD.txt")
    print("Mean: ", np.mean(data_order_xavier_GCD), "Std: ", np.std(data_order_xavier_GCD))

    f_stat, p_value, is_reject = F_test(xavier_GCD, data_order_xavier_GCD, 0.05, False)
    print("F-statistic:", f_stat, "p-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject = two_sample_t_test(xavier_GCD, data_order_xavier_GCD, 0.05)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject)
    t_stat, p_value, is_reject, lower, upper = my_two_sample_t_test(xavier_GCD, data_order_xavier_GCD, 0.05, False)
    print("T-statistic:", t_stat, "P-value:", p_value, "Is reject:", is_reject, "Lower:", lower, "Upper:", upper)

def q2_2():
    xavier_CR = [4500, 3800, 7350, 3900, 3300, 3200, 4200, 3200, 4800, 2850]
    general_order_CR = turn_txt_into_list("output_Q2/output_Xavier_ChickenRabbit.txt")
    t, p, reject = two_sample_t_test(xavier_CR, general_order_CR, 0.05, alternative='less')
    print("T-statistic:", t, "P-value:", p, "Is reject:", reject)

    xavier_GCD = [800, 850, 1200, 1000, 650, 1050, 600, 1150, 700, 1350]
    general_order_CR = turn_txt_into_list("output_Q2/output_Xavier_GCD.txt")
    t, p, reject = two_sample_t_test(xavier_GCD, general_order_CR, 0.05, alternative='less')
    print("T-statistic:", t, "P-value:", p, "Is reject:", reject)

    xavier_GCD_2 = [1150, 1000, 550, 650, 1000, 1000, 650, 1400, 650, 850]
    t, p, reject = two_sample_t_test(xavier_GCD_2, general_order_CR, 0.05, alternative='less')
    print("T-statistic:", t, "P-value:", p, "Is reject:", reject)

def main():
    # CR()
    # GCD()
    # DataOrder_CR()
    # DataOrder_GCD()
    q2_2()

if __name__ == "__main__":
    main()