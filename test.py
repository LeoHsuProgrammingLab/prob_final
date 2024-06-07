import torch
import numpy as np
from run import set_seed_all

def test():
    for i in range(10):
        set_seed_all(62)
        print(torch.nn.init.normal_(torch.empty(10, 1)))

def mean():
    CR = [4500, 3800, 7350, 3900, 3300, 3200, 4200, 3200, 4800, 2850]
    print(np.mean(CR), np.std(CR))
    GCD1 = [800, 850, 1200, 1000, 650, 1050, 600, 1150, 700, 1350]
    print(np.mean(GCD1), np.std(GCD1))
    GCD2 = [1150, 1000, 550, 650, 1000, 1000, 650, 1400, 650, 850]
    print(np.mean(GCD2), np.std(GCD2))

def test2(data: list):
    return np.mean(data), np.std(data)

if __name__ == '__main__':
    # test()
    # mean()
    data_CR = [4500, 4150, 3150, 4300, 3150, 3400, 2800, 3600, 4100, 3750, 3150, 3300, 5100, 4100, 6000, 4600, 3350, 3250, 3750, 3750, 6700, 4300, 4950, 4000, 4900, 6150, 3300, 5450, 4150, 3650, 4200, 4150, 4550, 3850, 3850, 3350, 4900, 3750, 4050, 3350, 3850, 4600, 4850, 4250, 8200, 5300, 6600, 4450, 6850, 3400, 3650, 3000, 4250, 3500, 3350, 4000, 4600, 4400, 4000, 4350, 4100, 3600, 3850, 4150, 3850, 3150, 3950, 3800, 3400, 5500, 4000, 3300, 3700, 4650, 4900, 3750, 3500, 4650, 4200, 3000, 4450, 4000, 3150, 4300, 5450, 4400, 2550, 3300, 3900, 3800, 3400, 4200, 4150, 3150, 5650, 7750, 2800, 4750, 4200, 3200]
    print(test2(data_CR))
    data_GCD = []
    with open('output_Q2/output_Xavier_GCD.txt', 'r') as f:
        data = [int(line.strip()) for line in f.readlines()]
        data_GCD.append(data)
    print(test2(data_GCD))

