import torch

class CROrder:
    @staticmethod
    def even_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        even_data = torch.tensor([item for item in perm_list if sum(item) % 2 == 0], dtype=torch.long)
        odd_data = torch.tensor([item for item in perm_list if sum(item) % 2 != 0], dtype=torch.long)
        return torch.cat((even_data, odd_data), 0)

    @staticmethod
    def odd_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        even_data = torch.tensor([item for item in perm_list if sum(item) % 2 == 0], dtype=torch.long)
        odd_data = torch.tensor([item for item in perm_list if sum(item) % 2 != 0], dtype=torch.long)
        result = torch.cat((odd_data, even_data), 0)
        return result

    @staticmethod
    def partial_even_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        even_data = torch.tensor([item for item in perm_list if sum(item[:-3]) % 2 == 0], dtype=torch.long)
        odd_data = torch.tensor([item for item in perm_list if sum(item[:-3]) % 2 != 0], dtype=torch.long)
        return torch.cat((even_data, odd_data), 0)

    @staticmethod
    def partial_odd_first(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        even_data = torch.tensor([item for item in perm_list if sum(item[:-3]) % 2 == 0], dtype=torch.long)
        odd_data = torch.tensor([item for item in perm_list if sum(item[:-3]) % 2 != 0], dtype=torch.long)
        return torch.cat((odd_data, even_data), 0)

    @staticmethod
    def trinity(perm: torch.tensor) -> torch.tensor:
        perm_list = perm.tolist()
        first_data = torch.tensor([item for item in perm_list if sum(item) % 3 == 0], dtype=torch.long)
        second_data = torch.tensor([item for item in perm_list if sum(item) % 3 == 1], dtype=torch.long)
        third_data = torch.tensor([item for item in perm_list if sum(item) % 3 == 2], dtype=torch.long)
        return torch.cat((first_data, second_data, third_data), 0)

    @staticmethod
    def chicken_first(perm: torch.tensor, asc = False) -> torch.tensor:
        perm_list = perm.tolist()
        sort_list = sorted(perm_list, key=lambda x: int(f"{x[6]}{x[7]}"), reverse=not asc)
        return torch.tensor(sort_list)

    @staticmethod
    def rabbit_first(perm: torch.tensor, asc = False) -> torch.tensor:
        perm_list = perm.tolist()
        sort_list = sorted(perm_list, key=lambda x: int(f"{x[8]}{x[9]}"), reverse=not asc)
        return torch.tensor(sort_list)

    @staticmethod
    def animal_num_first(perm: torch.tensor, asc = False) -> torch.tensor:
        perm_list = perm.tolist()
        sort_list = sorted(perm_list, key=lambda x: int(f"{x[0]}{x[1]}{x[2]}"), reverse=not asc)
        return torch.tensor(sort_list)

    @staticmethod
    def feet_num_first(perm: torch.tensor, asc = False) -> torch.tensor:
        perm_list = perm.tolist()
        sort_list = sorted(perm_list, key=lambda x: int(f"{x[3]}{x[4]}{x[5]}"), reverse=not asc)
        return torch.tensor(sort_list) 

if __name__ == "__main__":
    perm = torch.tensor([[0, 0, 3, 0, 0, 8, 0, 2, 0, 1], [0, 0, 2, 0, 0, 5, 0, 1, 0, 4], [0, 0, 2, 0, 0, 5, 0, 1, 0, 4]], dtype=torch.long)
    perm_list = perm.tolist()
    for item in perm_list:
        print(int(f"{item[6]}{item[7]}"))
        print(sum(item))
        print(sum(item) % 2)  