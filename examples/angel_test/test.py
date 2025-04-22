def main():
    dict = {"uno":1, "dos":2}
    a = dict.keys()
    # print(a)

    b = list(range(10))
    # print(b[2:])

    # import random
    # b = [1,2,3,4,5,6,7,8,9]

    # # random.seed(4)
    # random.Random(4).shuffle(b)
    # print("With seed:", b)

    # # random.seed()
    # random.shuffle(b)
    # print("With no seed:", b)

    import numpy as np
    epsilon = 1e-10
    original_h = np.random.rand(10, 10)*1e-9
    print(original_h)
    mask_small = np.abs(original_h) < epsilon
    print(mask_small)


if __name__ == "__main__":
    main()