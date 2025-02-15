import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.cos(x)  # Despeje de x en la ecuación

def g_derivative(x):
    return -np.sin(x)  # Derivada de g(x) para analizar convergencia

def fixed_point_iteration(g, x0, tol=1e-5, max_iter=100):
    iterations = [x0]
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        iterations.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return x_new, iterations

# Parámetros iniciales
x0 = 0.5
tolerance = 1e-5

# Aplicamos el método
root, iter_values = fixed_point_iteration(g, x0, tolerance)

# Graficamos la convergencia
x_vals = np.linspace(0, 1, 100)
y_vals = g(x_vals)

deriv_vals = np.abs(g_derivative(x_vals))

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='g(x) = cos(x)')
plt.plot(x_vals, x_vals, '--', label='y = x')
plt.scatter(iter_values, g(np.array(iter_values)), color='red', label='Iteraciones')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.title('Convergencia del método de punto fijo')
plt.legend()
plt.grid()
plt.show()

# Evaluación de |g'(x)| < 1
derivative_check = all(deriv_vals < 1)

# Imprimir resultados
print(f"Raíz aproximada encontrada: {root}")
print(f"Número de iteraciones: {len(iter_values)}")
print(f"¿Se cumple |g'(x)| < 1 en el intervalo? {'Sí' if derivative_check else 'No'}")
