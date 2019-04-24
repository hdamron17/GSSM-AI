from random import random


def shoot(num_shots):
    #calculates shot inside circle
    count = 0
    for loc in range(num_shots):
        x = random() * 2 - 1 #in range [-1, 1)
        y = random() * 2 - 1 #in range [-1, 1)
        dist = x**2 + y**2 #square rooted unnecessary because < 1 stays < 1
        if dist < 1:
            count += 1
    return count

def pi(num_shots):
    hits = shoot(num_shots)
    return 4 * hits / num_shots

def main():
    nums = (100, 1000, 10000, 20000, 30000, 40000, 50000, 100000, 1000000, 10000000, 100000000, 1000000000)
    for num in nums:
        print("%d : %f" % (num, pi(num)))

if __name__ == "__main__":
    main()
