import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def initialize():
    a = np.random.randint(-3, 3)
    b = np.random.randint(2, 4)

    return [a, b]


def candidate(xt, sigma):
    while True:
        a = np.random.normal(xt, sigma)
        if (-3 <= a[0] <= 3) & (2 <= a[1] <= 4):
            return a



def formula(x, T):
    ret = np.exp((1/T)*(-0.01*(np.sin(x[0])*np.exp([(1-np.cos(x[1]))**2]) + np.cos(x[1])*np.exp([(1-np.sin(x[0]))**2])+(x[0]-x[1])**2)))
    return ret[0]


def acceptance_func(x, xt, T):
    ret = min(1, formula(x, T)/formula(xt, T))
    return ret


def bias(a):
    x = np.random.uniform(0, 1)
    if (1-a) >= x:
        return 0
    return 1


def z_function(x, y):
    ret = np.exp(-0.01*(np.sin(x)*np.exp([(1-np.cos(y))**2]) + np.cos(y)*np.exp([(1-np.sin(x))**2])+(x-y)**2))
    return ret[0]


# Initalize Xt
starter = initialize()
# Hyper-parameters
max_iterations = 1000
sigma = 1
algorithm = 1
# Leave T value
T = 1
C = 2
# Alter initial_T for hyper-parameter of SA
initial_T = 100
algorithm_name = ["Metropolis-Hastings", "Simulated Annealing"]
while algorithm < 3:
    # First run will be for Metro hastings Algorithm
    Xt = starter
    print(Xt)
    list_z = []
    list_x = []
    list_y = []
    cumulative_list = []
    list_z.append(formula(Xt, T))
    list_x.append(Xt[0])
    list_y.append(Xt[1])
    accepted_count = 0
    iterations = 0
    cumulative_list.append(0)
    for i in range(max_iterations):

        if T == 0:
            break
        if algorithm > 1:
            T = (C * np.log(iterations + initial_T)**-1)
        # Get candidate
        X = candidate(Xt, sigma)
        # Run acceptance probability function
        alpha = acceptance_func(X, Xt, T)
        # accept or reject?
        u = np.random.uniform(0, 1)

        if alpha <= u:
            accepted_count += 1
            Xt = X
            list_z.append(formula(Xt, T=1))
            list_x.append(Xt[0])
            list_y.append(Xt[1])
            cumulative_list.append(cumulative_list[i] + 1)
        else:
            cumulative_list.append(cumulative_list[i] - 1)

        iterations += 1
    print(accepted_count)
    print("Acceptance rate: ", (accepted_count / iterations) * 100, "%")
    print("Accepted points :", accepted_count)

    # Create wire/meshgrid graph #
    # randomise values to be used
    x = np.linspace(-3, 3, 90)
    y = np.linspace(2, 4, 30)
    X, Y = np.meshgrid(x, y)
    Z = z_function(X, Y)
    # Make graph
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='grey', alpha =0.5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    title = "Scatterplot graph of " + algorithm_name[algorithm-1]
    ax.set_title(title)

    # overlap previous graph with scatter plot from accepted points
    ax.scatter3D(list_x, list_y, list_z, c=list_z,  cmap='hsv')
    plt.show()

    plt.hist2d(list_x, list_y, bins = 200, range=((-3, 3), (2, 4)))
    plt.xlabel('x1')
    plt.ylabel('x2')
    title = "Histogram graph of " + algorithm_name[algorithm-1]
    plt.title(title)
    plt.show()
    # plot Cumulative graph
    plt.plot(cumulative_list)
    plt.xlabel("Iterations")
    plt.ylabel("Number of accepted points")
    plt.grid()
    title = "Cumulative graph of accepted points in " + algorithm_name[algorithm-1]
    plt.title(title)
    plt.show()

    # Begin Simulated Annealing
    algorithm += 1

    T = initial_T


