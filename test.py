import torch
from run import set_seed_all

def test():
    for i in range(10):
        set_seed_all(62)
        print(torch.nn.init.normal_(torch.empty(10, 1)))

if __name__ == '__main__':
    test()