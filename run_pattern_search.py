import numpy as np
import matplotlib.pyplot as plt
from _patternsearch import PatternSearch as Pattern
from _patternsearch import MyProblem

import time

def f(x):
    # time.sleep(0.5)
    f = (x[0] - 2) ** 2 + (x[1] + 3) ** 2 + np.sin(3 * x[0]) * np.cos(3 * x[1])
    return f

if __name__ == '__main__':
    ts = time.time()
    # ---- Run optimization ----
    problem = MyProblem(error_func=f, n_var=2, xl=np.array([-5, -5]), xu=np.array([5, 5]))
    algorithm = Pattern(problem=problem, x0= np.array([0, 0]), seed=1,
                        n_dims_per_parallel_computing=2, n_jobs=1, step_size=1.0)
    algorithm.initialize()

    for i in range(20):
        algorithm.next()   # step forward explicitly
        print("Time:", time.time() - ts, "| Error:", algorithm.history['f'][-1])

    print("Best Solution:", algorithm.history['x'][-1], "f(x)=", algorithm.history['f'][-1])

    print('Executed Time:', time.time() - ts)
    all_X = np.array(algorithm.problem.history['x'])
    all_F = np.array(algorithm.problem.history['f']).flatten()
    best_X_traj = np.array(algorithm.history['x'])
    best_F_traj = np.array(algorithm.history['f']).flatten()

    # Create contour of the function
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = f(np.vstack((X.flatten(), Y.flatten()))).reshape(X.shape)

    plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="f(x,y)")

    # All evaluated points (red dots)
    plt.scatter(all_X[:, 0], all_X[:, 1], c="red", s=20, label="Evaluations", alpha=0.4)

    # Best solution trajectory (white line with markers)
    plt.plot(best_X_traj[:, 0], best_X_traj[:, 1], '-o', c="white", markersize=6,
             markeredgecolor="black", label="Best trajectory")

    plt.title("Pattern Search Optimization Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()