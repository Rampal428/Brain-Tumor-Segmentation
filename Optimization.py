
import numpy as np
import tensorflow as tf
from Segmentation import residual_autoencoder
from tensorflow.keras.optimizers import Adam


def objective_function(solution, X_train, Y_train, X_val, Y_val):

    filters = int(np.clip(
        round(solution[0] / 16) * 16, 16, 64
    ))

    batch_size = int(np.clip(
        round(solution[1] / 8) * 8, 8, 64
    ))

    learning_rate = float(np.clip(
        solution[2], 0.0001, 0.001
    ))

    tf.keras.backend.clear_session()

    model = residual_autoencoder(
        (128, 128, 3),
        filters
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=10, 
        batch_size=batch_size,
        verbose=0
    )

    predictions = model.predict(
        X_val,
        batch_size=batch_size,
        verbose=0
    )

    # Objective function: Validation MSE
    fitness = np.mean(
        np.square(Y_val - predictions)
    )

    del model

    return float(fitness), filters, batch_size, learning_rate


# ============================================================
# 2. SECRETARY BIRD OPTIMIZATION ALGORITHM
# ============================================================

class SBOA:

    def __init__(self, pop_size=30, max_iter=50):

        self.pop_size = pop_size
        self.max_iter = max_iter

        # Number of optimization dimensions
        self.dim = 3

        # Lower and upper bounds
        self.lb = np.array([16, 8, 0.0001])
        self.ub = np.array([64, 64, 0.001])

        self.best_position = None
        self.best_fitness = np.inf

        self.convergence = []
        self.history = []

    # ========================================================
    # 3. SBOA OPTIMIZATION
    # ========================================================

    def optimize(self, X_train, Y_train, X_val, Y_val):

        # Population initialization
        population = np.random.uniform(
            self.lb,
            self.ub,
            (self.pop_size, self.dim)
        )

        fitness = np.zeros(self.pop_size)


        for i in range(self.pop_size):

            fitness[i], f, b, lr = objective_function(
                population[i],
                X_train, Y_train,
                X_val, Y_val
            )

            self.history.append([
                i + 1, f, b, lr, fitness[i]
            ])

       
        # Initial global best
        best_idx = np.argmin(fitness)

        self.best_position = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]


        for t in range(self.max_iter):


            for i in range(self.pop_size):

                X = population[i].copy()

                # ---------------- EXPLORATION ----------------

                if np.random.rand() < 0.5:

                    beta = 2 * np.random.rand() - 1

                    new_position = (
                        X
                        + beta * (self.best_position - X)
                        + np.random.rand(self.dim)
                        * (self.ub - self.lb) * 0.1
                    )

                # ---------------- EXPLOITATION ----------------

                else:

                    new_position = (
                        X
                        + np.random.rand(self.dim)
                        * (self.best_position - X)
                    )

                # Boundary handling
                new_position = np.clip(
                    new_position,
                    self.lb,
                    self.ub
                )

                # Evaluate new candidate
                new_fitness, f, b, lr = objective_function(
                    new_position,
                    X_train, Y_train,
                    X_val, Y_val
                )

                # Greedy selection
                if new_fitness < fitness[i]:

                    population[i] = new_position.copy()
                    fitness[i] = new_fitness

                # Update global best
                if new_fitness < self.best_fitness:

                    self.best_position = new_position.copy()
                    self.best_fitness = new_fitness

                evaluation = (
                    self.pop_size
                    + t * self.pop_size
                    + i + 1
                )

                self.history.append([
                    evaluation, f, b, lr, new_fitness
                ])

            # Store convergence
            self.convergence.append(
                self.best_fitness
            )


        return self.best_position, self.best_fitness
