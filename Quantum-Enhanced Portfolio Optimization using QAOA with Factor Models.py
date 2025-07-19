# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from collections import Counter
import pennylane as qml
from pennylane import numpy as pnp
import time

# --- Data Fetching and Pre-processing ---
vanguard_tickers = ["VOO", "VTI", "BND", "VXUS", "VGT"]
start_date = "2018-01-01"
end_date = "2023-12-31"

print("--- Starting Data Preparation ---")
print(f"Fetching historical data for Vanguard ETFs from {start_date} to {end_date}...\n")

data_raw = yf.download(vanguard_tickers, start=start_date, end=end_date, auto_adjust=False)
data = data_raw['Adj Close']
returns = data.pct_change().dropna()

print("\nData Preparation Complete. First 5 rows of daily returns:")
print(returns.head())
print("\n" + "="*70 + "\n")

# --- Factor Analysis using PCA ---
num_assets = len(vanguard_tickers)
num_factors = 2 

print(f"--- Performing Factor Analysis with PCA ({num_factors} components) ---\n")
pca = PCA(n_components=num_factors)
pca.fit(returns)

factor_loadings = pd.DataFrame(pca.components_.T,
                               index=returns.columns,
                               columns=[f'Factor_{i+1}' for i in range(num_factors)])
print("Factor Loadings (B):")
print(factor_loadings)
print(f"\nTotal variance explained by {num_factors} factors: {pca.explained_variance_ratio_.sum():.4f}")
print("\n" + "="*70 + "\n")

factor_returns = pd.DataFrame(pca.transform(returns),
                              index=returns.index,
                              columns=[f'Factor_{i+1}' for i in range(num_factors)])
factor_covariance_matrix = factor_returns.cov()

original_covariance = returns.cov()
factor_explained_covariance_diag = np.diag(factor_loadings.values @ np.diag(pca.explained_variance_) @ factor_loadings.values.T)
specific_variance_diag = np.diag(original_covariance) - factor_explained_covariance_diag
specific_variance_diag[specific_variance_diag < 0] = 0
specific_risk_matrix_D = np.diag(specific_variance_diag)

print("Specific Risk Diagonal Matrix (D):")
print(pd.DataFrame(specific_risk_matrix_D, index=returns.columns, columns=returns.columns))
print("\n" + "="*70 + "\n")

B_matrix = factor_loadings.values
Sigma_F_matrix = factor_covariance_matrix.values

factor_model_covariance = B_matrix @ Sigma_F_matrix @ B_matrix.T + specific_risk_matrix_D
factor_model_covariance = pd.DataFrame(factor_model_covariance, index=returns.columns, columns=returns.columns)

print("Factor Model Reconstructed Covariance Matrix (Sigma):")
print(factor_model_covariance)
print("\n" + "="*70 + "\n")

expected_returns = returns.mean()
print("Expected Daily Returns (mu vector):")
print(expected_returns)
print("\n" + "="*70 + "\n")

# --- Classical Portfolio Optimization (Minimum Variance Example) ---
def calculate_portfolio_metrics(weights, expected_returns, cov_matrix, annualization_factor=252):
    portfolio_return = np.sum(weights * expected_returns) * annualization_factor
    portfolio_variance = weights.T @ cov_matrix @ weights * annualization_factor
    portfolio_std_dev = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std_dev, portfolio_variance

def classical_portfolio_variance_objective(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))

initial_weights = np.array(num_assets * [1. / num_assets])

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
num_qubits = num_assets

def build_qaoa_cost_hamiltonian(Q_matrix, num_qubits):
    coeffs = []
    ops = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                coeffs.append(Q_matrix[i, i] / 2)
                ops.append(qml.Identity(i))
                coeffs.append(-Q_matrix[i, i] / 2)
                ops.append(qml.PauliZ(i))
            elif i < j:
                coeffs.append(Q_matrix[i, j] / 4)
                ops.append(qml.Identity(i) @ qml.Identity(j))
                coeffs.append(-Q_matrix[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.Identity(j))
                coeffs.append(-Q_matrix[i, j] / 4)
                ops.append(qml.Identity(i) @ qml.PauliZ(j))
                coeffs.append(Q_matrix[i, j] / 4)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, ops).simplify()

# Device definition with reduced shots for faster sampling during tuning
dev = qml.device("lightning.qubit", wires=num_qubits, shots=1000)

@qml.qnode(dev)
def qaoa_circuit(params, h_cost, mixer_h, num_qubits, p_layers_circuit):
    gamma = params[0]
    beta = params[1]
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for layer in range(p_layers_circuit):
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
        for i in range(num_qubits):
            qml.RX(2 * beta[layer], wires=i)
    return qml.expval(h_cost)

mixer_h = qml.Hamiltonian([1.0] * num_qubits, [qml.PauliX(i) for i in range(num_qubits)])

# This qaoa_sampling_circuit_internal will be used within the tuning loop
@qml.qnode(dev)
def qaoa_sampling_circuit_internal(params, h_cost, mixer_h, num_qubits, p_layers_circuit):
    gamma = params[0]
    beta = params[1]
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    for layer in range(p_layers_circuit):
        for coeff, op in zip(h_cost.coeffs, h_cost.ops):
            if isinstance(op, qml.Identity): continue
            elif isinstance(op, qml.PauliZ) and len(op.wires) == 1: qml.RZ(2 * coeff * gamma[layer], wires=op.wires[0])
            elif isinstance(op, qml.ops.op_math.Prod) and len(op.wires) == 2:
                if all(isinstance(factor, qml.PauliZ) for factor in op.operands): qml.IsingZZ(2 * coeff * gamma[layer], wires=op.wires)
                else: qml.exp(op, coeff * gamma[layer])
            else: qml.exp(op, coeff * gamma[layer])
        for i in range(num_qubits):
            qml.RX(2 * beta[layer], wires=i)
    return qml.sample(wires=range(num_qubits))


# --- Automated Hyperparameter Tuning Function ---
def tune_qaoa_parameters(q_risk_aversion_values, lambda_penalty_values, p_layers_values, stepsize_values,
                         num_qaoa_runs_per_hp, optimization_steps_per_run,
                         expected_returns, factor_model_covariance, num_assets, K_target_assets):
    
    tuning_results = []
    sigma_pnp = pnp.array(factor_model_covariance.values, requires_grad=False)
    mu_pnp = pnp.array(expected_returns.values, requires_grad=False)

    for q_risk_aversion in q_risk_aversion_values:
        for lambda_penalty in lambda_penalty_values:
            for p_layers in p_layers_values:
                for stepsize in stepsize_values:
                    print(f"\n--- Tuning Run: q={q_risk_aversion}, lambda={lambda_penalty}, p={p_layers}, step={stepsize} ---")

                    Q_matrix = pnp.zeros((num_assets, num_assets))
                    for i in range(num_assets):
                        for j in range(num_assets):
                            if i == j:
                                Q_matrix[i, i] = sigma_pnp[i, i] - q_risk_aversion * mu_pnp[i]
                            else:
                                Q_matrix[i, j] = sigma_pnp[i, j]

                    for i in range(num_assets):
                        Q_matrix[i, i] += lambda_penalty * (1 - 2 * K_target_assets)
                        for j in range(i + 1, num_assets):
                            Q_matrix[i, j] += lambda_penalty * 2
                            Q_matrix[j, i] += lambda_penalty * 2

                    H_cost = build_qaoa_cost_hamiltonian(Q_matrix, num_assets)
                    
                    best_cost_for_hp_set = np.inf
                    best_params_for_hp_set = None
                    current_hp_cost_history = [] 

                    for run_idx in range(num_qaoa_runs_per_hp):
                        params = pnp.random.uniform(low=0, high=2*np.pi, size=(2, p_layers), requires_grad=True)
                        optimizer = qml.AdamOptimizer(stepsize=stepsize)
                        
                        run_cost_history = [] 
                        for i in range(optimization_steps_per_run):
                            params, cost = optimizer.step_and_cost(qaoa_circuit, params, h_cost=H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=p_layers)
                            run_cost_history.append(cost)
                        
                        current_run_final_cost = run_cost_history[-1]
                        if current_run_final_cost < best_cost_for_hp_set:
                            best_cost_for_hp_set = current_run_final_cost
                            best_params_for_hp_set = params
                            current_hp_cost_history = run_cost_history 
                    
                    samples = qaoa_sampling_circuit_internal(best_params_for_hp_set, h_cost=H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=p_layers)
                    sample_strings = ["".join(str(int(b)) for b in sample) for sample in samples]
                    counts = Counter(sample_strings)
                    
                    best_sharpe_for_hp_set = -np.inf
                    best_bitstring_for_hp_set = None
                    best_weights_for_hp_set = np.zeros(num_assets)
                    best_return_for_hp_set = 0
                    best_std_dev_for_hp_set = 0

                    for bitstring in counts.keys():
                        current_selected_assets_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
                        current_num_selected_assets = len(current_selected_assets_indices)

                        if current_num_selected_assets != K_target_assets or current_num_selected_assets == 0:
                            continue

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
                        'best_weights': best_weights_for_hp_set.tolist(),
                        'cost_history': current_hp_cost_history, # This was already there
                        'best_params': best_params_for_hp_set.tolist() # ADDED THIS LINE
                    })
    return pd.DataFrame(tuning_results)


# --- Define Hyperparameter Search Space (Optimized for Speed) ---
q_risk_aversion_values = [0.5]
lambda_penalty_values = [10.0]
p_layers_values = [1]
stepsize_values = [0.05]

num_qaoa_runs_per_hp_set = 1 
optimization_steps_per_run = 20 

K_target_assets = 2 

print("\n" + "="*70 + "\n")
print("--- Starting Automated QAOA Hyperparameter Tuning (Optimized for Speed) ---")
print(f"Evaluating {len(q_risk_aversion_values) * len(lambda_penalty_values) * len(p_layers_values) * len(stepsize_values)} hyperparameter combinations.")
print(f"Each combination will run {num_qaoa_runs_per_hp_set} QAOA optimizations for {optimization_steps_per_run} steps.")
print(f"Target number of selected assets (K): {K_target_assets}")
print("\n" + "="*70 + "\n")

start_tuning_time = time.time()
tuning_df = tune_qaoa_parameters(q_risk_aversion_values, lambda_penalty_values, p_layers_values, stepsize_values,
                                 num_qaoa_runs_per_hp_set, optimization_steps_per_run,
                                 expected_returns, factor_model_covariance, num_assets, K_target_assets)
end_tuning_time = time.time()
print(f"\nTotal QAOA Tuning Time: {end_tuning_time - start_tuning_time:.2f} seconds")


print("\n--- Hyperparameter Tuning Results ---")
print(tuning_df)
print("\n" + "="*70 + "\n")

# --- Select the Overall Best QAOA Portfolio from Tuning Results ---
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

    ### ADDED FROM CODE B FOR ENHANCED FINAL ANALYSIS ###
    # Recalculate and print the QUBO Matrix for the best hyperparameters
    final_best_q = best_tuned_qaoa_result['q_risk_aversion']
    final_best_l = best_tuned_qaoa_result['lambda_penalty']
    final_best_p_layers = int(best_tuned_qaoa_result['p_layers'])

    final_Q_matrix = pnp.zeros((num_assets, num_assets))
    sigma_pnp_final = pnp.array(factor_model_covariance.values, requires_grad=False)
    mu_pnp_final = pnp.array(expected_returns.values, requires_grad=False)

    for i in range(num_assets):
        for j in range(num_assets):
            if i == j:
                final_Q_matrix[i, i] = sigma_pnp_final[i, i] - final_best_q * mu_pnp_final[i]
            else:
                final_Q_matrix[i, j] = sigma_pnp_final[i, j]

    for i in range(num_assets):
        final_Q_matrix[i, i] += final_best_l * (1 - 2 * K_target_assets)
        for j in range(i + 1, num_assets):
            final_Q_matrix[i, j] += final_best_l * 2
            final_Q_matrix[j, i] += final_best_l * 2

    final_H_cost = build_qaoa_cost_hamiltonian(final_Q_matrix, num_assets)

    print("\n" + "="*70 + "\n")
    print("QUBO Matrix (Q) for Overall Best QAOA Portfolio:")
    print(pd.DataFrame(final_Q_matrix, index=vanguard_tickers, columns=vanguard_tickers))
    print("\nCost Hamiltonian (H_cost) for Overall Best QAOA Portfolio:")
    print(final_H_cost)
    print("\n" + "="*70 + "\n")

    # Retrieve the cost history stored during tuning
    final_cost_history_for_plot = best_tuned_qaoa_result['cost_history']


    print("\n" + "="*70 + "\n")
    print("--- QAOA Optimization Cost History for Overall Best QAOA Portfolio ---")
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(final_cost_history_for_plot)), final_cost_history_for_plot, marker='o', linestyle='-', markersize=4)
    plt.title(f'QAOA Optimization Cost History (Best HP Set: p={final_best_p_layers}, step={best_tuned_qaoa_result["stepsize"]})')
    plt.xlabel('Optimization Step')
    plt.ylabel('Cost Function Value')
    plt.grid(True)
    plt.show()
    print("\n" + "="*70 + "\n")
    ### END ADDED FROM CODE B FOR ENHANCED FINAL ANALYSIS ###


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
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*70 + "\n")

    ### ADDED FROM CODE B FOR ENHANCED FINAL ANALYSIS ###
    # Redefine sampling QNode and device for final plots with specific shots
    final_sampling_shots_value = 10000 # Use 10000 shots for high-quality final distribution
    dev_final_sampling = qml.device("lightning.qubit", wires=num_qubits, shots=final_sampling_shots_value)

    @qml.qnode(dev_final_sampling)
    def qaoa_sampling_circuit_final(params, h_cost, mixer_h, num_qubits, p_layers_circuit):
        gamma = params[0]
        beta = params[1]
        for i in range(num_qubits): qml.Hadamard(wires=i)
        for layer in range(p_layers_circuit):
            for coeff, op in zip(h_cost.coeffs, h_cost.ops):
                if isinstance(op, qml.Identity): continue
                elif isinstance(op, qml.PauliZ) and len(op.wires) == 1: qml.RZ(2 * coeff * gamma[layer], wires=op.wires[0])
                elif isinstance(op, qml.ops.op_math.Prod) and len(op.wires) == 2:
                    if all(isinstance(factor, qml.PauliZ) for factor in op.operands): qml.IsingZZ(2 * coeff * gamma[layer], wires=op.wires)
                    else: qml.exp(op, coeff * gamma[layer])
                else: qml.exp(op, coeff * gamma[layer])
            for i in range(num_qubits): qml.RX(2 * beta[layer], wires=i)
        return qml.sample(wires=range(num_qubits))

    # Retrieve the best_params from the tuning result
    final_best_params = best_tuned_qaoa_result['best_params'] # Now correctly retrieve best_params

    print("--- Sampling the QAOA circuit for Overall Best QAOA Portfolio with FULL shots ---")
    final_samples = qaoa_sampling_circuit_final(final_best_params, h_cost=final_H_cost, mixer_h=mixer_h, num_qubits=num_assets, p_layers_circuit=final_best_p_layers)
    final_sample_strings = ["".join(str(int(b)) for b in sample) for sample in final_samples]
    final_counts = Counter(final_sample_strings)
    
    total_shots_final_int = dev_final_sampling.shots.total_shots 

    print(f"\nTotal samples: {total_shots_final_int}")
    print("Bitstring Frequencies (Top 5):")
    for bitstring, count in final_counts.most_common(5):
        print(f"  {bitstring}: {count} times ({count/total_shots_final_int:.2%})")

    print(f"\n--- Evaluating Top Bitstrings from Final Sampling for Overall Best Portfolio ---")
    final_evaluated_bitstrings_data = []
    top_n_to_evaluate_final = 5 # Number of top bitstrings to display for final evaluation
    for bitstring, _ in final_counts.most_common(top_n_to_evaluate_final):
        current_selected_assets_indices = [i for i, bit in enumerate(bitstring) if bit == '1']
        current_num_selected_assets = len(current_selected_assets_indices)

        current_weights = np.zeros(num_assets)
        if current_num_selected_assets > 0:
            weight_per_selected_asset = 1.0 / current_num_selected_assets
            for idx in current_selected_assets_indices:
                current_weights[idx] = weight_per_selected_asset
        
        current_return, current_std_dev, _ = calculate_portfolio_metrics(
            current_weights, expected_returns.values, factor_model_covariance.values
        )
        current_sharpe = current_return / current_std_dev if current_std_dev != 0 else np.nan

        final_evaluated_bitstrings_data.append({
            'Bitstring': bitstring,
            'Assets': [vanguard_tickers[i] for i in current_selected_assets_indices],
            'Num_Assets': current_num_selected_assets,
            'Return': current_return,
            'Risk': current_std_dev,
            'Sharpe': current_sharpe
        })
    final_evaluated_df = pd.DataFrame(final_evaluated_bitstrings_data)
    print(final_evaluated_df.round(4))
    print("\n" + "="*70 + "\n")


    # --- Bitstring Distribution Plot ---
    sorted_counts_for_plot = dict(sorted(final_counts.items(), key=lambda item: item[1], reverse=True))
    top_n_bitstrings_plot = min(10, len(sorted_counts_for_plot))

    bitstring_labels = list(sorted_counts_for_plot.keys())[:top_n_bitstrings_plot]
    bitstring_frequencies = [count / total_shots_final_int for count in list(sorted_counts_for_plot.values())[:top_n_bitstrings_plot]]

    plt.figure(figsize=(12, 6))
    plt.bar(bitstring_labels, bitstring_frequencies, color='lightgreen')
    plt.xlabel('Bitstring (Asset Selection)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Sampled Bitstrings (Top {top_n_bitstrings_plot}) for Best QAOA Result (Full Shots)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("\n" + "="*70 + "\n")
    ### END ADDED FROM CODE B FOR ENHANCED FINAL ANALYSIS ###


    # --- Comparison Table: Classical vs. Quantum ---
    comparison_data = {
        'Metric': ['Annualized Return', 'Annualized Risk (Std Dev)', 'Sharpe Ratio'],
        'Classical Min Variance': [classical_min_var_return, classical_min_var_std_dev, classical_min_var_sharpe],
        'QAOA Portfolio': [best_qaoa_return, best_qaoa_std_dev, best_qaoa_sharpe]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.set_index('Metric')

    print("\n--- Portfolio Performance Comparison (Classical vs. QAOA) ---")
    print(comparison_df.round(4))
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
