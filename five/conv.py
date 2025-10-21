from five.hamil import eig_even_odd
import numpy as np
import random

# choose parameter M based on the convergence of energy levels
Ec = 200

# order Ej1 Ej2 Ej3 EjEint12 Eint23 Eint13


def gen_points():
    # generate points in parameter space to be used for convergence
    min_Ec = 40
    max_Ec = 1000
    min_Ej = 6000
    max_Ej = 20000
    min_Eint = -100
    max_Eint = 100
    Ejs = np.linspace(min_Ej, max_Ej, 14)
    num_random = 100
    # we have 16 parameters in 5T
    points = np.zeros(shape=(len(Ejs) + num_random, 16))
    i = 0
    for E in Ejs:
        points[i, :] = [
            Ec,
            Ec,
            Ec,
            Ec,
            Ec,
            E,
            E,
            E,
            E,
            E,
            max_Eint,
            max_Eint,
            max_Eint,
            max_Eint,
            max_Eint,
            max_Eint,
        ]
        i += 1
    for _ in range(num_random):
        Ecs = [random.randint(min_Ec, max_Ec) for _ in range(5)]
        Ejs = [random.randint(min_Ej, max_Ej) for _ in range(5)]
        Eints = [
            round(random.random() * (max_Eint - min_Eint) + min_Eint, 2)
            for _ in range(6)
        ]
        points[i, :] = Ecs + Ejs + Eints
        i += 1
    np.save("random_points_conv", points)


MM = np.arange(12, 22, 2)
C = 50


def collect():
    points = np.load("random_points_conv.npy")
    print(points)
    num_levels = (
        100  # number of levels above ground to include 1 2 3 excitation subspace
    )
    relative_errs = np.zeros((len(points), len(MM), num_levels))

    for p_i, p in enumerate(points):
        print(p)
        print(p_i)
        levels = np.zeros((len(MM), num_levels))
        for j, M in enumerate(MM):
            vals = eig_even_odd(*p, only_energy=True, M=M, C=C)
            levels[j, :] = (
                vals[1 : num_levels + 1] - vals[0]
            )  # take levels relative to ground
        final = levels[-1, :]
        relerr = (levels - final) / final  # all relative to their final
        relative_errs[p_i, :, :] = relerr

    maxed = np.max(
        np.abs(relative_errs), axis=(0, 2)
    )  # max over parameter point and energy level
    np.save("converce_5T_maxed", maxed)


if __name__ == "__main__":
    gen_points()
    collect()
