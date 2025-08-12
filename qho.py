import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
m = 1.0
omega = 1.0
hbar = 1.0
beta = 10.0
N = 100
dtau = beta / N

# Potential function (change here to try other potentials)
def V(x):
    # Harmonic oscillator potential
    return 0.5 * m * omega**2 * x**2
    # Example for anharmonic oscillator:
    # lambda_ = 1.0
    # return 0.5 * m * omega**2 * x**2 + lambda_ * x**4

# Local action for a single point in path (only terms involving x[i])
def local_action(path, i):
    ip1 = (i + 1) % N
    im1 = (i - 1) % N

    kinetic = (m / (2*dtau)) * ((path[ip1] - path[i])**2 + (path[i] - path[im1])**2)
    potential = dtau * V(path[i])
    return kinetic + potential

# Metropolis update for one time slice i
def metropolis_update(path, i, step_size):
    old_x = path[i]
    new_x = old_x + np.random.uniform(-step_size, step_size)

    old_S = local_action(path, i)
    path[i] = new_x
    new_S = local_action(path, i)

    dS = new_S - old_S

    if dS < 0 or np.random.rand() < np.exp(-dS / hbar):
        return True
    else:
        path[i] = old_x
        return False

# Energy estimator using discrete path
def energy_estimate(path):
    # Kinetic energy estimator (using finite difference)
    kinetic = np.sum((path - np.roll(path, 1))**2) * (m / (2 * dtau**2)) / N
    potential = np.mean(V(path))
    return kinetic + potential

# MC simulation parameters
n_steps = 60000
equil_steps = 10000
step_size = 1.0

path = np.zeros(N)
energy_samples = []
accept_count = 0
total_updates = 0

# Store some paths for visualization (e.g., every 500 steps after equilibration)
stored_paths = []

for step in range(n_steps):
    for i in range(N):
        total_updates += 1
        if metropolis_update(path, i, step_size):
            accept_count += 1

    # After equilibration, sample energy and occasionally store path
    if step >= equil_steps:
        e = energy_estimate(path)
        energy_samples.append(e)

        if step % 500 == 0:
            stored_paths.append(path.copy())

acceptance_ratio = accept_count / total_updates
print(f"Acceptance ratio: {acceptance_ratio:.3f}")

energy_samples = np.array(energy_samples)
mean_energy = np.mean(energy_samples)
std_energy = np.std(energy_samples) / np.sqrt(len(energy_samples))

print(f"Estimated ground state energy: {mean_energy:.5f} ± {std_energy:.5f}")
print(f"Exact ground state energy: {0.5 * hbar * omega}")

# Plot energy convergence
plt.figure(figsize=(10, 5))
plt.plot(energy_samples, lw=0.5)
plt.xlabel("Sample number")
plt.ylabel("Energy estimate")
plt.title("Energy estimate vs Monte Carlo samples")
plt.grid(True)
plt.show()

# Visualize sampled paths
plt.figure(figsize=(10, 6))
for i, p in enumerate(stored_paths[:6]):
    plt.plot(np.linspace(0, beta, N), p, label=f"Sampled path {i+1}")
plt.xlabel("Imaginary time τ")
plt.ylabel("x(τ)")
plt.title("Sampled paths from Path Integral Monte Carlo")
plt.legend()
plt.grid(True)
plt.show()
