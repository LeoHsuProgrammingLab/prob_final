import torch
import math

class GCDOrder:
    @staticmethod
    def coprime_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        prime_list = [item for item in perm_list if item[-1] == 1 and item[-2] == 0]
        non_prime_list = [item for item in perm_list if item[-1] != 1 or item[-2] != 0]
        return torch.tensor(prime_list + non_prime_list, dtype=torch.long)

    @staticmethod
    def zero_digits_first(perm: torch.tensor, zero_nums = 2) -> torch.tensor:
        perm_list = perm.tolist()
        zero_list = [item for item in perm_list if sum(num == 0 for num in item) > zero_nums]
        non_zero_list = [item for item in perm_list if sum(num == 0 for num in item) <= zero_nums]
        return torch.tensor(zero_list + non_zero_list, dtype=torch.long)
    
    @staticmethod
    def less_hard_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        steps_list = []
        for item in perm_list:
            a = (item[0]*10 + item[1])
            b = (item[2]*10 + item[3])
            c = (item[4]*10 + item[5])
            if a > b:
                a, b = b, a
            steps = GCDOrder.gcd_steps(a, b)
            # size_score = math.log(a) + math.log(b)
            prime_factor_score = GCDOrder.prime_factor_score(a) + GCDOrder.prime_factor_score(b)
            score = prime_factor_score
            steps_list.append([item, score])
        
        sort_list = sorted(steps_list, key=lambda x: x[1])
        return torch.tensor([x[0] for x in sort_list], dtype=torch.long)

    @staticmethod
    def gcd_steps(a, b):
        steps = 0
        while b != 0:
            a, b = b, a % b
            steps += 1
        return steps
    
    @staticmethod
    def prime_factor_score(n):
        def prime_factors(n):
            """Return a list of prime factors of the given integer n."""
            factors = []
            # Check for the number of 2s that divide n
            while n % 2 == 0:
                factors.append(2)
                n = n // 2

            # n must be odd at this point so a skip of 2 (i.e., 3, 5, 7, ...)
            for i in range(3, int(math.sqrt(n)) + 1, 2):
                # While i divides n, add i and divide n
                while n % i == 0:
                    factors.append(i)
                    n = n // i

            # This condition is to check if n is a prime number
            # greater than 2
            if n > 2:
                factors.append(n)
            
            return factors
        
        score = 0
        # Assuming a function that returns the list of prime factors
        factors = prime_factors(n)
        for factor in factors:
            score += math.log(factor)
        return score
    
    @staticmethod
    def small_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        item_list = []
        for item in perm_list:
            a = (item[0]*10 + item[1])
            b = (item[2]*10 + item[3])
            c = (item[4]*10 + item[5])
            print(a, b, c)  
            item_list.append([item, a, b, c])

        sort_list = sorted(item_list, key=lambda x: (x[1], x[3]))
        return torch.tensor([x[0] for x in sort_list], dtype=torch.long)
    
    @staticmethod
    def sum_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        sorted_list = sorted(perm_list, key=lambda x: sum(x))

        return torch.tensor(sorted_list, dtype=torch.long)
    
    @staticmethod
    def lexicographic_order(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        sorted_list = sorted(perm_list)
        return torch.tensor(sorted_list, dtype=torch.long)

if __name__ == "__main__":
    gcd_perm = torch.tensor(
        [
            [1, 5, 0, 9, 0, 3],
            [6, 2, 3, 1, 3, 1],
            [2, 4, 0, 8, 0, 8],
        ],
        dtype=torch.long
    )

    GCDOrder.less_gcd_step(gcd_perm)
    