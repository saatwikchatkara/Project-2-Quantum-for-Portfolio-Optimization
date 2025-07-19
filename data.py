# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from collections import Counter # Import Counter here for broader scope

# Import PennyLane and its NumPy-like interface
import pennylane as qml
from pennylane import numpy as pnp

# --- Data Fetching and Pre-processing ---
# This block fetches historical financial data for Vanguard ETFs,
# calculates daily returns, and prepares them for factor analysis.

vanguard_tickers = [
    "VOO",  # Vanguard S&P 500 ETF (S&P 500)
    "VTI",  # Vanguard Total Stock Market ETF
    "BND",  # Vanguard Total Bond Market ETF
    "VXUS", # Vanguard Total International Stock ETF
    "VGT",  # Vanguard Information Technology ETF
]

start_date = "2018-01-01"
end_date = "2023-12-31"

print("--- Starting Data Preparation ---")
print(f"Fetching historical data for Vanguard ETFs from {start_date} to {end_date}...\n")

# Fetch raw data (Open, High, Low, Close, Adj Close, Volume)
data_raw = yf.download(vanguard_tickers, start=start_date, end=end_date, auto_adjust=False)
# Select only the Adjusted Close prices for portfolio analysis
data = data_raw['Adj Close']
# Calculate daily returns and drop any rows with NaN (usually the first row)
returns = data.pct_change().dropna()

print("\nData Preparation Complete. First 5 rows of daily returns:")
print(returns.head())
print("\n" + "="*70 + "\n")

# --- Factor Analysis using PCA ---
# This section performs Principal Component Analysis (PCA) to extract statistical factors
# and reconstruct a more robust covariance matrix for portfolio optimization.

num_assets = len(vanguard_tickers)
# Define the number of factors (principal components) to extract.
# This number should be less than the number of assets.
# Experiment with this value; typically factors explaining 80-90% of variance are chosen.
num_factors = 2 # For 5 assets, 2-3 factors is a reasonable starting point.

print(f"--- Performing Factor Analysis with PCA ({num_factors} components) ---\n")
pca = PCA(n_components=num_factors)
pca.fit(returns) # Fit PCA to the daily returns

# Factor Loadings (B): Represents how each asset's return is sensitive to each factor.
# It's the transpose of the principal components.
factor_loadings = pd.DataFrame(pca.components_.T,
                               index=returns.columns,
                               columns=[f'Factor_{i+1}' for i in range(num_factors)])
print("Factor Loadings (B):")
print(factor_loadings)
print(f"\nTotal variance explained by {num_factors} factors: {pca.explained_variance_ratio_.sum():.4f}")
print("\n" + "="*70 + "\n")

# Factor Returns (F): The time series of the extracted factors.
factor_returns = pd.DataFrame(pca.transform(returns),
                              index=returns.index,
                              columns=[f'Factor_{i+1}' for i in range(num_factors)])
# print("Factor Returns (first 5 rows):")
# print(factor_returns.head())
# print("\n" + "="*70 + "\n")

# Factor Covariance Matrix (Sigma_F): The covariance matrix of the factor returns.
factor_covariance_matrix = factor_returns.cov()
# print("Factor Covariance Matrix (Sigma_F):")
# print(factor_covariance_matrix)
# print("\n" + "="*70 + "\n")

# Specific Risk (D): The idiosyncratic (asset-specific) variance not explained by the factors.
# This is estimated by taking the diagonal of the original covariance matrix
# and subtracting the diagonal of the covariance explained by the factors.
original_covariance = returns.cov()
# Reconstruct the covariance explained by factors: B * Lambda * B^T, where Lambda is diagonal of explained variances
factor_explained_covariance_diag = np.diag(factor_loadings.values @ np.diag(pca.explained_variance_) @ factor_loadings.values.T)
specific_variance_diag = np.diag(original_covariance) - factor_explained_covariance_diag
specific_variance_diag[specific_variance_diag < 0] = 0 # Ensure non-negative specific variances
specific_risk_matrix_D = np.diag(specific_variance_diag)

print("Specific Risk Diagonal Matrix (D):")
print(pd.DataFrame(specific_risk_matrix_D, index=returns.columns, columns=returns.columns))
print("\n" + "="*70 + "\n")

# Reconstruct the Asset Covariance Matrix using the Factor Model: Sigma = B * Sigma_F * B_matrix.T + D
B_matrix = factor_loadings.values
Sigma_F_matrix = factor_covariance_matrix.values

factor_model_covariance = B_matrix @ Sigma_F_matrix @ B_matrix.T + specific_risk_matrix_D
factor_model_covariance = pd.DataFrame(factor_model_covariance, index=returns.columns, columns=returns.columns)

print("Factor Model Reconstructed Covariance Matrix (Sigma):")
print(factor_model_covariance)
print("\n" + "="*70 + "\n")

# Expected Returns (mu): Mean of the daily returns.
expected_returns = returns.mean()
print("Expected Daily Returns (mu vector):")
print(expected_returns)
print("\n" + "="*70 + "\n")

# --- Classical Portfolio Optimization (Minimum Variance Example) ---
# This section demonstrates a classical optimization to provide a benchmark
# for comparison with the quantum result.

def calculate_portfolio_metrics(weights, expected_returns, cov_matrix, annualization_factor=252):
    """Calculates portfolio expected return, variance, and standard deviation."""
    portfolio_return = np.sum(weights * expected_returns) * annualization_factor
    portfolio_variance = weights.T @ cov_matrix @ weights * annualization_factor
    portfolio_std_dev = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std_dev, portfolio_variance

# Objective function for classical optimization (minimize portfolio variance)
def classical_portfolio_variance_objective(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Constraints: weights sum to 1, and weights are non-negative
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets)) # Weights between 0 and 1

# Initial guess for weights (equal weighting)
initial_weights = np.array(num_assets * [1. / num_assets])

# Perform classical optimization for minimum variance portfolio
classical_min_var_result = minimize(classical_portfolio_variance_objective, initial_weights,
                                    args=(factor_model_covariance.values,),
                                    method='SLSQP', bounds=bounds, constraints=constraints)

classical_min_var_weights = pd.Series(classical_min_var_result.x, index=returns.columns)
print("Classical Minimum Variance Portfolio Weights (using factor model covariance):")
print(classical_min_var_weights)

classical_min_var_return, classical_min_var_std_dev, _ = calculate_portfolio_metrics(
    classical_min_var_weights.values, expected_returns.values, factor_model_covariance.values
)
classical_min_var_sharpe = classical_min_var_return / classical_min_var_std_dev if classical_min_var_std_dev != 0 else np.nan

print(f"\nClassical Minimum Variance Portfolio Annualized Return: {classical_min_var_return:.4f}")
print(f"Classical Minimum Variance Portfolio Annualized Risk (Std Dev): {classical_min_var_std_dev:.4f}")
print(f"Classical Minimum Variance Portfolio Sharpe Ratio (Risk-Free Rate = 0): {classical_min_var_sharpe:.4f}")
print("\n" + "="*70 + "\n")

# --- PennyLane Quantum Portfolio Optimization (using QAOA) ---

# Define the number of qubits (each qubit represents an asset selection: 0 or 1)
num_qubits = num_assets

# --- 1. Formulate the QUBO Problem ---
# The objective function to minimize for portfolio optimization is typically:
# C(x) = x^T * Sigma * x - q * mu^T * x + lambda * (sum(x_i) - K)^2
# Where:
# x_i are binary variables (0 or 1) representing asset selection
# Sigma is the covariance matrix (factor_model_covariance)
# mu is the expected returns vector
# q (q_risk_aversion) is a risk aversion parameter (higher q means more emphasis on return)
# lambda (lambda_penalty) is a penalty coefficient for violating the constraint
# K (K_target_assets) is the desired number of assets to select

# Convert covariance and expected returns to PennyLane NumPy arrays for easier use
sigma_pnp = pnp.array(factor_model_covariance.values, requires_grad=False)
mu_pnp = pnp.array(expected_returns.values, requires_grad=False)

# --- Define the QAOA Cost Hamiltonian based on QUBO ---
def build_qaoa_cost_hamiltonian(Q_matrix, num_qubits):
    """
    Builds the PennyLane Cost Hamiltonian from the QUBO Q_matrix.
    Mapping from binary x_i to Pauli Z_i is x_i = (I - Z_i) / 2.
    """
    coeffs = []
    ops = []

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j: # Diagonal terms Q_ii * x_i = Q_ii * (I - Z_i) / 2
                coeffs.append(Q_matrix[i, i] / 2)
                ops.append(qml.Identity(i))
                coeffs.append(-Q_matrix[i, i] / 2)
                ops.append(qml.PauliZ(i))
            elif i < j: # Off-diagonal terms Q_ij * x_i * x_j = Q_ij * 1/4 * (I - Z_i - Z_j + Z_i Z_j)
                # Q_matrix is symmetric, so Q_ij * x_i x_j + Q_ji * x_j x_i becomes 2 * Q_ij * x_i x_j
                # The Q_matrix already contains (Q_ij + Q_ji) for off-diagonals, so just Q_ij/4
                coeffs.append(Q_matrix[i, j] / 4)
                ops.append(qml.Identity(i) @ qml.Identity(j)) # Constant term
                coeffs.append(-Q_matrix[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.Identity(j))
                coeffs.append(-Q_matrix[i, j] / 4)
                ops.append(qml.Identity(i) @ qml.PauliZ(j))
                coeffs.append(Q_matrix[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    return qml.Hamiltonian(coeffs, ops).simplify()

# Define the quantum device (simulator for now)
# ADDED shots=10000 here to enable sampling on lightning.qubit
dev = qml.device("lightning.qubit", wires=num_qubits, shots=10000)

# Define the QAOA circuit structure
@qml.qnode(dev)
def qaoa_circuit(params, h_cost, mixer_h, num_qubits, p_layers_circuit): # p_layers_circuit passed as arg
    """
    QAOA circuit for portfolio optimization.
    Args:
        params (array[float]): Angle parameters for gamma and beta (gamma[0], beta[0], gamma[1], beta[1]...)
        h_cost (qml.Hamiltonian): The cost Hamiltonian.
        mixer_h (qml.Hamiltonian): The mixer Hamiltonian (usually sum of Pauli X).
        num_qubits (int): Number of qubits (assets).
        p_layers_circuit (int): Number of QAOA layers for this circuit instance.
    """
    # Unpack gamma and beta from the single params array
    gamma = params[0]
    beta = params[1]

    # Apply initial layer of Hadamard gates to create an equal superposition state
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # Apply p layers of QAOA operations
    for layer in range(p_layers_circuit): # Use p_layers_circuit here
        # Manually apply exp for each term in H_cost for Trotterization
        # Iterate through the terms (coefficients and operators) of the cost Hamiltonian
        for coeff, op in zip(h_cost.coeffs, h_cost.ops):
            if isinstance(op, qml.Identity):
                continue # Identity terms don't evolve the state

            # Handle single PauliZ terms (e.g., Q_ii * x_i)
            elif isinstance(op, qml.PauliZ) and len(op.wires) == 1:
                # qml.RZ(angle, wires) applies e^(-i * angle/2 * PauliZ)
                # We need e^(-i * coeff * gamma[layer] * PauliZ)
                # So, angle = 2 * coeff * gamma[layer]
                qml.RZ(2 * coeff * gamma[layer], wires=op.wires[0])

            # Handle two-qubit PauliZ product terms (e.g., Q_ij * x_i * x_j)
            # Corrected: Use qml.ops.op_math.Prod for product operator type check
            elif isinstance(op, qml.ops.op_math.Prod) and len(op.wires) == 2:
                # Ensure it's specifically a product of two PauliZ operators
                if all(isinstance(factor, qml.PauliZ) for factor in op.operands):
                    qml.IsingZZ(2 * coeff * gamma[layer], wires=op.wires)
                else:
                    qml.exp(op, coeff * gamma[layer])
            else:
                # print(f"Warning: Unexpected operator type in H_cost: {op}. Attempting qml.exp.") # Suppressed for cleaner output
                qml.exp(op, coeff * gamma[layer])

        # Apply mixer Hamiltonian evolution (e^(-i * beta * H_mixer))
        for i in range(num_qubits):
            qml.RX(2 * beta[layer], wires=i)

    # Return the expectation value of the cost Hamiltonian for optimization
    return qml.expval(h_cost)

# Define the mixer Hamiltonian (standard for QAOA, sum of Pauli X gates)
mixer_h = qml.Hamiltonian([1.0] * num_qubits, [qml.PauliX(i) for i in range(num_qubits)])

# --- QAOA Sampling Circuit (for final interpretation) ---
@qml.qnode(dev) # Use the same 'dev' as optimization, now with shots enabled
def qaoa_sampling_circuit(params, h_cost, mixer_h, num_qubits, p_layers_circuit):
    """
    QAOA circuit for sampling the final state.
    """
    # Unpack gamma and beta from the single params array
    gamma = params[0]
    beta = params[1]

    # Apply initial layer of Hadamard gates to create superposition
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    # Apply p layers of QAOA
    for layer in range(p_layers_circuit): # Use p_layers_circuit here
        # Manually apply exp for each term in H_cost for Trotterization
        for coeff, op in zip(h_cost.coeffs, h_cost.ops):
            if isinstance(op, qml.Identity):
                continue
            elif isinstance(op, qml.PauliZ) and len(op.wires) == 1:
                qml.RZ(2 * coeff * gamma[layer], wires=op.wires[0])
            elif isinstance(op, qml.ops.op_math.Prod) and len(op.wires) == 2:
                if all(isinstance(factor, qml.PauliZ) for factor in op.operands):
                    qml.IsingZZ(2 * coeff * gamma[layer], wires=op.wires)
                else:
                    qml.exp(op, coeff * gamma[layer])
            else:
                qml.exp(op, coeff * gamma[layer])

        # Apply mixer Hamiltonian evolution (e^(-i * beta * H_mixer))
        for i in range(num_qubits):
            qml.RX(2 * beta[layer], wires=i)

    # Return measurement results (bitstrings)
    return qml.sample(wires=range(num_qubits))


# --- Automated Hyperparameter Tuning Function ---
def tune_qaoa_parameters(q_risk_aversion_values, lambda_penalty_values, p_layers_values, stepsize_values,
                         num_qaoa_runs_per_hp, optimization_steps_per_run,
                         expected_returns, factor_model_covariance, num_assets, K_target_assets):
    
    tuning_results = []

    # Iterate through all combinations of hyperparameters
    for q_risk_aversion in q_risk_aversion_values:
        for lambda_penalty in lambda_penalty_values:
            for p_layers in p_layers_values:
                for stepsize in stepsize_values:
                    print(f"\n--- Tuning Run: q={q_risk_aversion}, lambda={lambda_penalty}, p={p_layers}, step={stepsize} ---")

                    # --- Construct QUBO Matrix for current hyperparameters ---
                    Q_matrix = pnp.zeros((num_assets, num_assets))
                    sigma_pnp = pnp.array(factor_model_covariance.values, requires_grad=False)
                    mu_pnp = pnp.array(expected_returns.values, requires_grad=False)

                    for i in range(num_assets):
                        for j in range(num_assets):
                            if i == j:
                                Q_matrix[i, i] = sigma_pnp[i, i] - q_risk_aversion * mu_pnp[i]
                            else:
                                Q_matrix[i, j] = sigma_pnp[i, j]

                    # Add the penalty for the cardinality constraint
                    # (sum(x_i) - K)^2 = sum(x_i^2) + sum(x_i x_j) * 2 - 2K * sum(x_i) + K^2
                    # Since x_i^2 = x_i for binary variables: sum(x_i) + sum(x_i x_j)*2 - 2K*sum(x_i) + K^2
                    # = (1-2K)sum(x_i) + 2*sum(x_i x_j) + K^2
                    # Constant K^2 term can be ignored for optimization, but affects absolute cost value.
                    # Q_ii gets lambda * (1-2K)
                    # Q_ij gets lambda * 2 for i!=j
                    for i in range(num_assets):
                        Q_matrix[i, i] += lambda_penalty * (1 - 2 * K_target_assets)
                        for j in range(i + 1, num_assets):
                            Q_matrix[i, j] += lambda_penalty * 2
                            Q_matrix[j, i] += lambda_penalty * 2 # Ensure symmetry

                    H_cost = build_qaoa_cost_hamiltonian(Q_matrix, num_assets)
                    
                    # --- Run multiple QAOA optimizations for robustness ---
                    best_cost_for_hp_set = np.inf
                    best_params_for_hp_set = None

                    for run_idx in range(num_qaoa_runs_per_hp):
                        # Initialize parameters randomly for each run
                        params = pnp.random.uniform(low=0, high=2*np.pi, size=(2, p_layers), requires_grad=True)
                        optimizer = qml.AdamOptimizer(stepsize=stepsize)
                        
                        cost_history = []
                        for i in range(optimization_steps_per_run):
                            params, cost = optimizer.step_and_cost(qaoa_circuit, params, h_cost=H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=p_layers)
                            cost_history.append(cost)
                        
                        current_run_final_cost = cost_history[-1]
                        if current_run_final_cost < best_cost_for_hp_set:
                            best_cost_for_hp_set = current_run_final_cost
                            best_params_for_hp_set = params
                    
                    # --- Sample with best params for this HP set ---
                    # The dev used by qaoa_sampling_circuit now has shots enabled globally
                    samples = qaoa_sampling_circuit(best_params_for_hp_set, h_cost=H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=p_layers)
                    sample_strings = ["".join(str(int(b)) for b in sample) for sample in samples]
                    counts = Counter(sample_strings)
                    
                    # Evaluate ALL unique bitstrings that meet cardinality constraint
                    best_sharpe_for_hp_set = -np.inf
                    best_bitstring_for_hp_set = None
                    best_weights_for_hp_set = np.zeros(num_assets)
                    best_return_for_hp_set = 0
                    best_std_dev_for_hp_set = 0

                    for bitstring in counts.keys(): # Iterate over all unique bitstrings
                        current_selected_assets_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
                        current_num_selected_assets = len(current_selected_assets_indices)

                        if current_num_selected_assets != K_target_assets or current_num_selected_assets == 0:
                            continue # Skip if it doesn't meet cardinality constraint or is empty

                        current_weights = np.zeros(num_assets)
                        weight_per_selected_asset = 1.0 / current_num_selected_assets
                        for idx in current_selected_assets_indices:
                            current_weights[idx] = weight_per_selected_asset
                        
                        current_return, current_std_dev, _ = calculate_portfolio_metrics(
                            current_weights, expected_returns.values, factor_model_covariance.values
                        )
                        current_sharpe = current_return / current_std_dev if current_std_dev != 0 else np.nan

                        if current_sharpe > best_sharpe_for_hp_set:
                            best_sharpe_for_hp_set = current_sharpe
                            best_bitstring_for_hp_set = bitstring
                            best_weights_for_hp_set = current_weights
                            best_return_for_hp_set = current_return
                            best_std_dev_for_hp_set = current_std_dev
                    
                    tuning_results.append({
                        'q_risk_aversion': q_risk_aversion,
                        'lambda_penalty': lambda_penalty,
                        'p_layers': p_layers,
                        'stepsize': stepsize,
                        'final_cost': best_cost_for_hp_set,
                        'best_bitstring': best_bitstring_for_hp_set,
                        'best_sharpe': best_sharpe_for_hp_set,
                        'best_return': best_return_for_hp_set,
                        'best_risk': best_std_dev_for_hp_set,
                        'best_weights': best_weights_for_hp_set.tolist() # Store as list for DataFrame
                    })
    return pd.DataFrame(tuning_results)


# --- Define Hyperparameter Search Space ---
q_risk_aversion_values = [0.1, 0.5, 1.0]
lambda_penalty_values = [5.0, 10.0]
p_layers_values = [1, 2] # Number of QAOA layers
stepsize_values = [0.01, 0.05] # Optimizer learning rate

num_qaoa_runs_per_hp_set = 3 # Number of independent QAOA runs for each HP combination
optimization_steps_per_run = 50 # Number of optimization steps per QAOA run

# Set target number of assets (K)
K_target_assets = 2 # Example: select 2 out of 5 assets

print("\n" + "="*70 + "\n")
print("--- Starting Automated QAOA Hyperparameter Tuning ---")
print(f"Evaluating {len(q_risk_aversion_values) * len(lambda_penalty_values) * len(p_layers_values) * len(stepsize_values)} hyperparameter combinations.")
print(f"Each combination will run {num_qaoa_runs_per_hp_set} QAOA optimizations for {optimization_steps_per_run} steps.")
print(f"Target number of selected assets (K): {K_target_assets}")
print("\n" + "="*70 + "\n")

tuning_df = tune_qaoa_parameters(q_risk_aversion_values, lambda_penalty_values, p_layers_values, stepsize_values,
                                 num_qaoa_runs_per_hp_set, optimization_steps_per_run,
                                 expected_returns, factor_model_covariance, num_assets, K_target_assets)

print("\n--- Hyperparameter Tuning Results ---")
print(tuning_df)
print("\n" + "="*70 + "\n")

# --- Select the Overall Best QAOA Portfolio from Tuning Results ---
# Filter out rows where best_sharpe is -inf (meaning no valid bitstrings were found for that HP set)
valid_tuning_results = tuning_df[tuning_df['best_sharpe'] != -np.inf]

if not valid_tuning_results.empty:
    best_tuned_qaoa_result = valid_tuning_results.loc[valid_tuning_results['best_sharpe'].idxmax()]

    best_qaoa_return = best_tuned_qaoa_result['best_return']
    best_qaoa_std_dev = best_tuned_qaoa_result['best_risk']
    best_qaoa_sharpe = best_tuned_qaoa_result['best_sharpe']
    best_qaoa_bitstring = best_tuned_qaoa_result['best_bitstring']
    best_qaoa_weights = np.array(best_tuned_qaoa_result['best_weights'])

    print("\n--- Overall Best QAOA Portfolio from Tuning ---")
    print(f"Optimal Hyperparameters: q_risk_aversion={best_tuned_qaoa_result['q_risk_aversion']}, "
          f"lambda_penalty={best_tuned_qaoa_result['lambda_penalty']}, "
          f"p_layers={best_tuned_qaoa_result['p_layers']}, "
          f"stepsize={best_tuned_qaoa_result['stepsize']}")
    print(f"Best QAOA Portfolio (by Sharpe Ratio) Bitstring: {best_qaoa_bitstring}")
    selected_assets_final = [vanguard_tickers[i] for i, bit in enumerate(best_qaoa_bitstring) if bit == '1']
    print(f"Selected Assets: {selected_assets_final}")
    print(f"Number of selected assets: {len(selected_assets_final)}")

    # --- Optimal Portfolio Weights (QAOA Result - Table) ---
    optimal_portfolio_df = pd.DataFrame({
        'Asset': vanguard_tickers,
        'Weight': best_qaoa_weights
    })
    print("\n--- Optimal Portfolio Weights (QAOA Result - Table) ---")
    print(optimal_portfolio_df)
    print("\n" + "="*70 + "\n")

    # --- Create a bar chart for visual representation of QAOA weights ---
    plt.figure(figsize=(10, 6))
    plt.bar(optimal_portfolio_df['Asset'], optimal_portfolio_df['Weight'], color='lightcoral')
    plt.xlabel('Asset Ticker')
    plt.ylabel('Portfolio Weight')
    plt.title('Optimal Portfolio Asset Allocation (QAOA Result)')
    plt.ylim(0, 1) # Weights are between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*70 + "\n")

    # --- Bitstring Distribution Plot (for the best performing HP set) ---
    # Re-sample with best_params to get the distribution for the best result
    # We use the same 'dev' as before which now has shots enabled.
    # We need to make sure best_params is correctly associated with the correct H_cost and p_layers for the best result.
    # To be precise, reconstruct H_cost for the best hyperparameters:
    best_q = best_tuned_qaoa_result['q_risk_aversion']
    best_l = best_tuned_qaoa_result['lambda_penalty']
    best_p_layers = int(best_tuned_qaoa_result['p_layers']) # Ensure integer type

    best_Q_matrix = pnp.zeros((num_assets, num_assets))
    for i in range(num_assets):
        for j in range(num_assets):
            if i == j:
                best_Q_matrix[i, i] = sigma_pnp[i, i] - best_q * mu_pnp[i]
            else:
                best_Q_matrix[i, j] = sigma_pnp[i, j]
    for i in range(num_assets):
        best_Q_matrix[i, i] += best_l * (1 - 2 * K_target_assets)
        for j in range(i + 1, num_assets):
            best_Q_matrix[i, j] += best_l * 2
            best_Q_matrix[j, i] += best_l * 2
    
    best_H_cost = build_qaoa_cost_hamiltonian(best_Q_matrix, num_assets)
    
    # Need to retrieve the actual best_params from the tuning_df.
    # The 'best_params' for the current loop are `best_params_for_hp_set`.
    # To get the one for the *overall best* result, we need to store it or re-run the optimization for that specific set.
    # For simplicity, let's assume `best_params_for_hp_set` from the *last* iteration where `best_tuned_qaoa_result` was found is available.
    # A more robust way would be to store `best_params_for_hp_set` directly in the `tuning_results` DataFrame.
    # For this demonstration, we'll re-run the optimization for the best HP set to get its `best_params`.

    # Re-run optimization for the best HP set to retrieve the optimal parameters
    temp_params = pnp.random.uniform(low=0, high=2*np.pi, size=(2, best_p_layers), requires_grad=True)
    temp_optimizer = qml.AdamOptimizer(stepsize=best_tuned_qaoa_result['stepsize'])
    for _ in range(optimization_steps_per_run):
        temp_params, _ = temp_optimizer.step_and_cost(qaoa_circuit, temp_params, h_cost=best_H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=best_p_layers)
    final_best_params = temp_params


    final_samples = qaoa_sampling_circuit(final_best_params, h_cost=best_H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=best_p_layers)
    final_sample_strings = ["".join(str(int(b)) for b in sample) for sample in final_samples]
    final_counts = Counter(final_sample_strings)
    total_shots_final = dev.shots # Use the shots from the device

    sorted_counts_for_plot = dict(sorted(final_counts.items(), key=lambda item: item[1], reverse=True))
    top_n_bitstrings_plot = min(10, len(sorted_counts_for_plot)) # Display top N bitstrings, or fewer if not enough unique

    bitstring_labels = list(sorted_counts_for_plot.keys())[:top_n_bitstrings_plot]
    bitstring_frequencies = [count / total_shots_final for count in list(sorted_counts_for_plot.values())[:top_n_bitstrings_plot]]

    plt.figure(figsize=(12, 6))
    plt.bar(bitstring_labels, bitstring_frequencies, color='lightgreen')
    plt.xlabel('Bitstring (Asset Selection)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Sampled Bitstrings (Top {top_n_bitstrings_plot}) for Best QAOA Result')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*70 + "\n")

    # --- Comparison Table: Classical vs. Quantum ---
    comparison_data = {
        'Metric': ['Annualized Return', 'Annualized Risk (Std Dev)', 'Sharpe Ratio'],
        'Classical Min Variance': [classical_min_var_return, classical_min_var_std_dev, classical_min_var_sharpe],
        'QAOA Portfolio': [best_qaoa_return, best_qaoa_std_dev, best_qaoa_sharpe] # Use best QAOA metrics
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Metric')

    print("\n--- Portfolio Performance Comparison (Classical vs. QAOA) ---")
    print(comparison_df.round(4)) # Round for cleaner display
    print("\n" + "="*70 + "\n")

else:
    print("\n--- No valid QAOA portfolios found meeting the K_target_assets constraint. ---")
    print("Consider adjusting hyperparameters (e.g., lambda_penalty, q_risk_aversion) or K_target_assets.")
    print("\n" + "="*70 + "\n")


print("\n--- Automated QAOA Hyperparameter Tuning and Portfolio Optimization Complete ---")
print("This model demonstrates a more robust approach to finding optimal QAOA parameters and portfolio selections.")
print("Further steps for real-world application and research would involve:")
print("- **Expanding Search Space:** Explore a wider range of hyperparameters and more granular steps.")
print("- **Advanced Optimizers:** Investigate Bayesian Optimization or other meta-heuristic algorithms for tuning.")
print("- **More Sophisticated QAOA Ansätze:** Explore different circuit architectures beyond the basic QAOA layers.")
print("- **Other Quantum Algorithms:** Investigate VQE with different ansätze or Quantum Annealing for QUBOs.")
print("- **Continuous Weights:** This current approach uses binary selection (0 or 1). For continuous weights, more complex encoding schemes (e.g., using multiple qubits per asset, or amplitude encoding) or different quantum algorithms are needed.")
print("- **Real Quantum Hardware:** Run the optimized circuit on actual quantum computers (QPUs) to observe hardware effects.")
print("- **Robustness Testing:** Evaluate portfolio performance on out-of-sample data.")
print("- **Transaction Costs & Constraints:** Incorporate more realistic financial constraints like transaction costs, minimum investment amounts, etc.")
