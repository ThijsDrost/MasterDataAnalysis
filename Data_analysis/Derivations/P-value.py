import numpy as np

n = int(1e6)
repeats = 100

x1 = 0.4
std1 = 0.03
x2 = 0.57
std2 = 0.07

nums = np.zeros(repeats)
for i in range(100):
    vals1 = np.random.normal(x1, std1, n)
    vals2 = np.random.normal(x2, std2, n)

    num = np.sum(vals1 >= vals2)
    nums[i] = num/n

print(f'{np.mean(nums):.6f} +/- {np.std(nums):.6f}')
