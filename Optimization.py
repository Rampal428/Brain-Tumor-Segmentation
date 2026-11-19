

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tensorflow.keras.optimizers import Optimizer
import tensorflow as tf
class SBOA(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                  amsgrad=False, weight_decay=None, clipnorm=None, clipvalue=None,
                  global_clipnorm=None, use_ema=False, ema_momentum=0.99,
                  ema_overwrite_frequency=None, jit_compile=True, name="TCSPO", **kwargs):
        super().__init__(name=name, weight_decay=weight_decay, clipnorm=clipnorm,
                          clipvalue=clipvalue, global_clipnorm=global_clipnorm, use_ema=use_ema,
                          ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency,
                          jit_compile=jit_compile, **kwargs)
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.archive = [] 
   
    
    def objective_function(self,x):
        return np.sum(x**2)

   # ------------------------------
   # Secretary Bird Optimization Algorithm (SBOA)
    def SBOA_(self,gradient):
        dim=30
        lb=-100
        ub=100
        pop_size=30
        max_iter=5
        # === Step 1: Initialization ===
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.apply_along_axis(self.objective_function, 1, population)
        best_idx = np.argmin(fitness)
        Xbest = population[best_idx].copy()
        Fbest = fitness[best_idx]

        # === Main Loop ===
        for t in range(1, max_iter + 1):
            for i in range(pop_size):
                X = population[i].copy()
                r1 = np.random.rand()

                # === Exploration Phase ===
                if t < max_iter / 3:
                    # Eq. 4–5: Random walk + attraction to best
                    r2 = np.random.rand(dim)
                    beta = 2 * np.random.rand() - 1  # in [-1, 1]
                    new_pos = X + beta * (Xbest - X) + r2 * (ub - lb)
                elif t < 2 * max_iter / 3:
                    # Eq. 7–8: Logistic or sinusoidal movement
                    a = 2 * (1 - t / max_iter)
                    r3 = np.random.rand(dim)
                    new_pos = X + a * np.sin(r3 * np.pi) * (Xbest - X)
                else:
                    # Eq. 9–10: Levy-like or attraction model
                    r4 = np.random.randn(dim)
                    new_pos = X + r4 * (Xbest - X) / (np.linalg.norm(X - Xbest) + 1e-10)

                # Bound check
                new_pos = np.clip(new_pos, lb, ub)

                # === Exploitation Phase ===
                r = np.random.rand()
                if r < 0.5:
                    # C1 component in Eq. 14
                    C1 = np.random.rand(dim)
                    step = C1 * (Xbest - new_pos)
                else:
                    # C2 component in Eq. 14
                    C2 = np.random.rand(dim)
                    step = C2 * (new_pos - Xbest)

                # Eq. 15: Final update
                new_pos += step
                new_pos = np.clip(new_pos, lb, ub)

                # Evaluate and accept if better
                f_new = self.objective_function(new_pos)
                if f_new < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = f_new

                    # Update global best
                    if f_new < Fbest:
                        Xbest = new_pos.copy()
                        Fbest = f_new

           

        return Xbest, Fbest
        
    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        self.SBOA_(gradient)
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]
        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        if isinstance(gradient, tf.IndexedSlices):
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
    
    



