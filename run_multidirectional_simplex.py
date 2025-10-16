import numpy as np
import matplotlib.pyplot as plt
from _multidirectional_simplex import MultidirectionalSimplex
from _multidirectional_simplex import MyProblem

import time

def f(x):
    # time.sleep(0.5)
    f = (x[0] - 2) ** 2 + (x[1] + 3) ** 2 + np.sin(3 * x[0]) * np.cos(3 * x[1])
    return f

if __name__ == '__main__':
    ts = time.time()
    # ---- Run optimization ----
    problem = MyProblem(error_func=f, n_var=2)
    algorithm = MultidirectionalSimplex(problem=problem,
                                        x0= np.array([0, 0]),
                                        initial_simplex=np.array([[0.0, 0.0], [0.25, 0.0], [-0.25, 0.25]]),
                                        bounds=[[-5, -5], [5, 5]],
                                        n_dims_per_parallel_computing=2,
                                        n_jobs=1)
    algorithm.initialize()
    for i in range(10):
        algorithm.next_look_ahead_n_iterations(n_iters=3)
        print("Time:", time.time() - ts, "| Error:", algorithm.history['v0'][-1].F)

    # for i in range(100):
    #     algorithm.next()   # step forward explicitly
    #     print("Time:", time.time() - ts, "| Error:", algorithm.history['v0'][-1].F)

    print("Best Solution:", algorithm.history['v0'][-1].X, "f(x)=", algorithm.history['v0'][-1].F)

    print('Executed Time:', time.time() - ts)
    all_X = []
    all_F = []
    [all_X.extend([vertex.X for vertex in simplex]) for simplex in algorithm.history['sim']]
    [all_F.extend([vertex.F for vertex in simplex]) for simplex in algorithm.history['sim']]
    best_X_traj = np.array([vertex.X for vertex in algorithm.history['v0']])
    best_F_traj = np.array([vertex.F for vertex in algorithm.history['v0']])

    # Create contour of the function
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = f(np.vstack((X.flatten(), Y.flatten()))).reshape(X.shape)

    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="f(x,y)")

    # All evaluated points (red dots)
    # plt.scatter(np.array(all_X)[:, 0], np.array(all_X)[:, 1], c="red", s=20, label="Evaluations", alpha=0.4)

    # Best solution trajectory (white line with markers)
    plt.plot(best_X_traj[:, 0], best_X_traj[:, 1], '-o', c="white", markersize=6,
             markeredgecolor="black", label="Best trajectory")

    plt.title("Multidirectional Simplex Search Optimization Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()