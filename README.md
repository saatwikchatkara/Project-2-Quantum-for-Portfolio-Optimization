# Quantum-Enhanced Portfolio Optimization: A Hybrid Factor-QAOA Approach

## Team Name
QuantumQuantfinanceIIST

## Team Members
* **[Saatwik chatkara]** - WISER Enrollment ID: [gst-fnw9Qzp0LIIwBRy]
* **[Devang Narula]** - WISER Enrollment ID: [Your WISER ID 2]
* **[Mohamed Armoon Shaliq]** - WISER Enrollment ID: [Your WISER ID 3]

## Project Summary (Approx. 500 words)

The traditional paradigm of portfolio optimization, while foundational in finance, faces significant hurdles when applied to real-world market data. Estimating accurate expected returns and, more critically, the covariance matrix from historical data is notoriously challenging due to market noise, non-stationarity, and the "curse of dimensionality." These issues often lead to unstable portfolio weights and suboptimal investment performance. Our project tackles these limitations by introducing a robust, hybrid quantum-classical framework for portfolio selection, specifically addressing the common financial constraint of **cardinality** (selecting a fixed number of assets).

Our methodology begins with sophisticated classical data pre-processing. Instead of relying solely on the raw empirical covariance matrix, which is highly susceptible to estimation error, we employ a **Principal Component Analysis (PCA)-based factor model**. This model posits that asset returns are driven by a smaller set of underlying common factors, plus idiosyncratic (asset-specific) risk. By extracting these principal components and reconstructing the covariance matrix as $\Sigma = B \Sigma_F B^T + D$, where $B$ represents factor loadings, $\Sigma_F$ is the factor covariance, and $D$ captures specific risk, we achieve a more stable, less noisy, and financially interpretable risk model. This step is crucial for providing cleaner input to the subsequent optimization phase.

The core portfolio selection problem is then formulated as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem. Our objective function integrates three critical financial considerations: portfolio risk (variance), expected return, and a penalty for deviating from the target number of selected assets ($K$). The QUBO matrix, $Q$, encapsulates these terms, transforming the financial problem into a format directly compatible with quantum optimization algorithms. The binary nature of the QUBO, where each variable $x_i \in \{0, 1\}$ signifies asset inclusion or exclusion, inherently addresses the cardinality constraint effectively.

For solving this QUBO, we leverage the **Quantum Approximate Optimization Algorithm (QAOA)**, a leading candidate for near-term quantum computers (NISQ devices). Implemented using PennyLane, our QAOA circuit comprises alternating layers of a problem-specific Cost Hamiltonian (derived from the QUBO) and a universal Mixer Hamiltonian. The goal of QAOA is to variationally find optimal quantum gate parameters (angles $\gamma$ and $\beta$) that drive the quantum state towards a superposition where the amplitudes of optimal classical solutions are significantly amplified. We use the high-performance `lightning.qubit` simulator, configured for both exact expectation value calculations during optimization and shot-based sampling for final solution extraction.

A critical aspect of QAOA's practical application is its sensitivity to **hyperparameters**. We've implemented a systematic grid search to fine-tune key parameters: the risk aversion coefficient ($q$), the cardinality constraint penalty ($\lambda$), the number of QAOA layers ($p$), and the classical optimizer's learning rate (`stepsize`). For each hyperparameter combination, multiple QAOA optimization runs are performed to mitigate local minima issues, and the best-performing set of quantum parameters is used to sample asset combinations. The sampled bitstrings are then evaluated, and the portfolio yielding the highest Sharpe Ratio (annualized return per unit of risk) is identified as the optimal solution for that particular hyperparameter set.

Finally, our project provides comprehensive analytical tools. We compare the QAOA-derived optimal portfolio's performance (Sharpe Ratio, return, risk) against a classical minimum variance benchmark, offering insights into the quantum approach's potential advantages. Visualizations, including portfolio weight allocation charts and bitstring distribution plots, further aid in understanding the generated solutions. This hybrid framework offers a robust and adaptable pathway towards more sophisticated, quantum-accelerated financial decision-making, particularly in areas requiring combinatorial optimization.

## Project Presentation Deck

A detailed presentation covering the project's background, methodology, implementation, results, and future work is available here:
[Link to Project Presentation Deck (e.g., Google Slides, PDF, etc.)]
*(Please replace this placeholder with the actual link to your presentation file or online deck.)*

## Features

This project provides a comprehensive toolkit for quantum-enhanced portfolio optimization:

* **Automated Financial Data Ingestion:** Seamlessly downloads historical adjusted close prices for specified Exchange-Traded Funds (ETFs) or stocks using the `yfinance` library, providing a flexible data foundation.
* **Robust Factor Model Implementation:** Leverages Principal Component Analysis (PCA) to identify latent risk factors in asset returns. This allows for the construction of a more stable and less noisy covariance matrix, crucial for reliable risk assessment.
* **Classical Portfolio Optimization Benchmark:** Includes a classical minimum variance portfolio optimization using `scipy.optimize`, serving as a baseline to evaluate the performance of the quantum-derived solutions.
* **QUBO Problem Formulation:** Accurately translates the combined objectives of risk minimization, return maximization, and cardinality constraint enforcement into a QUBO matrix, making the problem amenable to quantum solvers.
* **PennyLane-Powered QAOA Core:** Implements the Quantum Approximate Optimization Algorithm using PennyLane, a leading framework for quantum machine learning. This includes the construction of the QAOA cost and mixer Hamiltonians, and the variational quantum circuit.
* **Systematic Hyperparameter Tuning Framework:** Employs a robust grid search to explore the critical hyperparameters of the QAOA and its associated objective function. This includes the risk aversion coefficient (`q`), the cardinality penalty (`lambda`), the number of QAOA layers (`p`), and the classical optimizer's step size.
* **Detailed Performance Analysis & Visualization:** Generates informative outputs including optimal portfolio weights, key financial performance metrics (annualized return, risk, Sharpe ratio), and insightful visualizations like portfolio allocation bar charts and quantum state bitstring distributions.
* **Hybrid Quantum-Classical Architecture:** Demonstrates a practical hybrid workflow where classical computation handles data preprocessing and hyperparameter search, while the quantum computer (simulated in this case) tackles the core combinatorial optimization problem.

## Methodology

### Factor Model for Covariance Estimation: Enhancing Robustness

The foundation of any robust portfolio optimization lies in an accurate and stable estimate of the asset covariance matrix ($\Sigma$). Traditional empirical covariance matrices, directly calculated from historical returns, suffer from significant noise, especially when dealing with a large number of assets or limited observations. This can lead to volatile optimization results and portfolios that are not truly diversified.

To overcome this, we adopt a **single-factor model** based on Principal Component Analysis (PCA). The core idea is that asset returns are driven by a small number of common (systematic) factors plus idiosyncratic (asset-specific) risk. The covariance matrix is then reconstructed using this factor structure:

$$
\Sigma = B \Sigma_F B^T + D
$$

Where:
* $\mathbf{\Sigma}$ is the $N \times N$ reconstructed covariance matrix for $N$ assets.
* $\mathbf{B}$ (Factor Loadings) is an $N \times K$ matrix where $K$ is the number of principal components (factors) extracted. Each element $B_{ij}$ represents the sensitivity of asset $i$ to factor $j$. This matrix is derived from the eigenvectors of the historical covariance matrix.
* $\mathbf{\Sigma_F}$ (Factor Covariance Matrix) is a $K \times K$ matrix representing the covariance between the extracted factors. Since PCA ensures orthogonal factors, $\Sigma_F$ will typically be a diagonal matrix with the eigenvalues representing the variance explained by each factor.
* $\mathbf{D}$ (Specific Risk) is an $N \times N$ diagonal matrix where $D_{ii}$ represents the idiosyncratic (asset-specific) variance of asset $i$ that is *not* explained by the common factors. This is calculated as the total variance of an asset minus the variance explained by the factors.

This factor model approach provides several benefits:
* **Reduced Noise:** By modeling only the systematic risk components and treating idiosyncratic risk as uncorrelated, the reconstructed covariance matrix is smoother and less prone to sampling errors.
* **Improved Stability:** The estimates become more stable over time, leading to more consistent portfolio decisions.
* **Dimensionality Reduction:** Instead of estimating $N(N+1)/2$ unique elements in a full covariance matrix, we estimate $N \times K$ factor loadings, $K$ factor variances, and $N$ specific variances, which is significantly fewer if $K \ll N$.

### Portfolio Optimization as a QUBO Problem: Bridging Classical and Quantum

To leverage quantum computing, the portfolio selection problem must be mapped into a form that quantum algorithms can process. The **Quadratic Unconstrained Binary Optimization (QUBO)** format is a natural fit for many combinatorial optimization problems and is directly compatible with QAOA.

Our objective function for portfolio selection, which we aim to minimize, balances risk, return, and a cardinality constraint:

$$
C(\mathbf{x}) = \mathbf{x}^T \Sigma \mathbf{x} - q \cdot \boldsymbol{\mu}^T \mathbf{x} + \lambda (\sum_{i=1}^N x_i - K)^2
$$

Here:
* $\mathbf{x} = [x_0, x_1, \ldots, x_{N-1}]^T$ is a binary vector where $x_i \in \{0, 1\}$. If $x_i = 1$, asset $i$ is selected for the portfolio; if $x_i = 0$, it is not.
* $\Sigma$ is the $N \times N$ (factor model) covariance matrix.
* $\boldsymbol{\mu}$ is the $N \times 1$ vector of expected returns.
* $q$ (`q_risk_aversion`) is a tunable hyperparameter representing the risk aversion coefficient. A higher $q$ places more emphasis on maximizing expected returns, potentially at the cost of higher risk.
* $\lambda$ (`lambda_penalty`) is a tunable hyperparameter representing the penalty strength for violating the cardinality constraint.
* $K$ (`K_target_assets`) is the target number of assets to be selected in the portfolio. The term $(\sum_{i=1}^N x_i - K)^2$ penalizes portfolios that do not have exactly $K$ assets.

To convert this expression into the standard QUBO form ($\mathbf{x}^T Q \mathbf{x}$), we expand the terms and derive the $Q$ matrix where diagonal elements $Q_{ii}$ correspond to coefficients of $x_i$ terms, and off-diagonal elements $Q_{ij}$ (for $i \ne j$) correspond to coefficients of $x_i x_j$ terms.

Once the QUBO matrix $Q$ is formed, it is transformed into a **Cost Hamiltonian ($H_C$)** for QAOA. This involves mapping each binary variable $x_i$ to a Pauli-Z operator on a corresponding qubit using the standard mapping: $x_i \rightarrow (I - Z_i)/2$. This conversion translates the classical optimization problem into an eigenvalue problem on a quantum system.

### QAOA Implementation with PennyLane: The Quantum Engine

The Quantum Approximate Optimization Algorithm (QAOA) is a variational quantum algorithm designed to find approximate solutions to combinatorial optimization problems. It's particularly well-suited for NISQ-era quantum computers due to its relatively shallow circuit depth and hybrid quantum-classical nature.

Our implementation uses **PennyLane**, a powerful open-source library that integrates quantum hardware and simulators with popular machine learning frameworks.

1.  **Quantum Device:** We employ `qml.device("lightning.qubit", wires=num_qubits, shots=10000)`.
    * `lightning.qubit` is a highly efficient state-vector simulator backend developed by PennyLane.
    * `wires=num_qubits` specifies the number of qubits, corresponding to the number of assets.
    * Crucially, `shots=10000` enables **shot-based simulation**. While `lightning.qubit` can perform analytical expectation value calculations for optimization (which are typically faster and exact), specifying shots allows it to perform actual measurement sampling, which is necessary for the `qml.sample` function used to obtain bitstring distributions from the final quantum state.

2.  **QAOA Circuit (`qaoa_circuit`):** This is the core quantum routine responsible for finding the optimal angles.
    * **Initialization:** All qubits are initialized into an equal superposition state using Hadamard gates (`qml.Hadamard`). This ensures that all possible bitstrings (portfolio selections) have an initial non-zero amplitude.
    * **Alternating Operators (Layers):** The circuit consists of `p_layers` (QAOA layers), each comprising:
        * **Cost Hamiltonian Evolution:** An operator $e^{-i \gamma H_C}$ is applied, where $H_C$ is the Cost Hamiltonian derived from the QUBO problem, and $\gamma$ is a variational angle. This operator encodes the problem's cost function into the quantum state, driving the amplitudes of low-cost solutions higher. Our implementation manually applies the exponential of each term in $H_C$ (Pauli-Z and IsingZZ terms) for precise control and efficiency.
        * **Mixer Hamiltonian Evolution:** An operator $e^{-i \beta H_M}$ is applied, where $H_M$ is the Mixer Hamiltonian, typically a sum of Pauli-X gates across all qubits ($\sum_i X_i$), and $\beta$ is another variational angle. This operator creates superpositions and allows the quantum state to explore the solution space, moving between different bitstring configurations.
    * **Output:** The `qaoa_circuit` returns the `qml.expval(h_cost)`, which is the expectation value of the Cost Hamiltonian. The objective of the classical optimizer is to minimize this value.

3.  **Classical Optimizer:** An `AdamOptimizer` is used to iteratively update the QAOA variational parameters (the $\gamma$ and $\beta$ angles for each layer). The optimizer receives the cost function (expectation value) and its gradients from PennyLane, adjusting the angles to reduce the cost.

4.  **QAOA Sampling Circuit (`qaoa_sampling_circuit`):** After the QAOA optimization converges for a given set of hyperparameters, the `best_params_for_hp_set` are fed into this circuit. Instead of returning an expectation value, this QNode returns `qml.sample(wires=range(num_qubits))`. This performs a shot-based measurement on the quantum state, yielding a distribution of bitstrings (portfolio selections) that are most likely to be optimal. The frequencies of these bitstrings provide insights into the algorithm's preference for certain asset combinations.

### Hyperparameter Tuning: Optimizing the Optimizer

The performance of QAOA is highly sensitive to its hyperparameters, which are not learned during the quantum-classical optimization loop. Therefore, a systematic **hyperparameter tuning** process is essential to find the configuration that yields the best portfolio performance.

We implement a **grid search** strategy, exploring predefined ranges for the following critical hyperparameters:

* **`q_risk_aversion_values`**: Controls the balance between risk minimization and return maximization in the financial objective.
* **`lambda_penalty_values`**: Determines the strength of the penalty for violating the cardinality constraint.
* **`p_layers_values`**: The number of QAOA layers, influencing the circuit's expressivity and optimization complexity.
* **`stepsize_values`**: The learning rate of the classical Adam optimizer.

For each unique combination of these hyperparameters:
1.  The corresponding `Q_matrix` and `H_cost` are constructed.
2.  Multiple independent QAOA optimization runs (`num_qaoa_runs_per_hp_set`) are performed to mitigate the chance of converging to local minima.
3.  The best-performing set of variational angles from these runs is identified based on the lowest final cost.
4.  The `qaoa_sampling_circuit` is then executed with these optimal angles to obtain a distribution of bitstrings.
5.  Each sampled bitstring that satisfies the `K_target_assets` cardinality constraint is evaluated for its financial performance (return, risk, Sharpe ratio).
6.  The bitstring yielding the highest Sharpe ratio for that hyperparameter set is recorded as the best result.

Finally, all tuning results are collected into a pandas DataFrame, allowing for easy comparison and selection of the overall best QAOA portfolio across the entire hyperparameter search space.

## Installation Guide

To set up and run this project, follow these steps:

1.  **Ensure Python is Installed:** This project requires Python 3.8 or higher. Download it from [python.org](https://www.python.org/downloads/).

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/QuantumQuants/quantum-portfolio-optimization.git](https://github.com/QuantumQuants/quantum-portfolio-optimization.git)
    cd quantum-portfolio-optimization
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

4.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    You'll see `(venv)` in your terminal prompt, indicating the virtual environment is active.

5.  **Install Project Dependencies:**
    ```bash
    pip install yfinance pandas numpy matplotlib scikit-learn pennylane pennylane-lightning
    ```

## Usage Instructions

Once the installation is complete and your virtual environment is activated, you can run the portfolio optimization pipeline.

1.  **Execute the Script:**
    Simply run the main Python script from your terminal:
    ```bash
    python main_portfolio_optimizer.py # Assuming your script is named main_portfolio_optimizer.py
    ```

2.  **Monitor Progress:**
    The script will print real-time updates to your console, detailing each step of data processing, classical optimization, and the start of each QAOA hyperparameter tuning run.

3.  **View Plots:**
    Two plots will be generated and displayed automatically:
    * A bar chart showing the optimal portfolio asset allocation (weights) as determined by QAOA.
    * A bar chart showing the frequency distribution of sampled bitstrings for the best QAOA result.
    Close the plots to allow the script to continue execution if it pauses.

## Configuration and Customization

You can modify key parameters directly within the Python script to adapt it to your specific research or analysis needs:

* **Asset Universe:** `vanguard_tickers` list.
* **Historical Data Range:** `start_date`, `end_date`.
* **Factor Model Parameters:** `num_factors`.
* **Portfolio Cardinality Constraint:** `K_target_assets`.
* **QAOA Hyperparameter Search Space:** `q_risk_aversion_values`, `lambda_penalty_values`, `p_layers_values`, `stepsize_values`.
* **QAOA Optimization Run Parameters:** `num_qaoa_runs_per_hp_set`, `optimization_steps_per_run`.
* **Quantum Device Shots:** `dev = qml.device("lightning.qubit", wires=num_qubits, shots=10000)`.

## Understanding the Results and Analysis

The project provides a multi-faceted analysis of the portfolio optimization:

1.  **Classical Minimum Variance Portfolio:** A traditional benchmark.
2.  **Hyperparameter Tuning Results (`tuning_df`):** A DataFrame summarizing the performance of each hyperparameter combination.
3.  **Overall Best QAOA Portfolio:** Details of the optimal hyperparameters found, the selected asset bitstring, and calculated portfolio metrics for the best-performing QAOA configuration.
4.  **Optimal Portfolio Weights (QAOA Result - Table & Chart):** Visual representation of capital allocation.
5.  **Bitstring Distribution Plot:** Illustrates the probabilities of various asset combinations being measured from the final quantum state.
6.  **Portfolio Performance Comparison (Classical vs. QAOA):** A concise table comparing annualized return, risk, and Sharpe ratio.

## Future Enhancements and Research Directions

This project serves as a foundational demonstration. Several avenues exist for significant enhancement and further research:

* **Advanced Hyperparameter Optimization:** Implement more efficient methods like Bayesian Optimization or Genetic Algorithms.
* **Dynamic QAOA Ansätze:** Explore adaptive QAOA strategies where circuit depth or structure changes dynamically.
* **Alternative Quantum Algorithms:** Investigate other quantum algorithms like VQE or specialized quantum annealing approaches.
* **Continuous Portfolio Weights:** Extend the model to handle continuous portfolio weights using advanced encoding schemes or hybrid algorithms.
* **Real Quantum Hardware Execution:** Transition from simulators to actual quantum processing units (QPUs), addressing real-world challenges like noise and hardware constraints.
* **Robustness and Out-of-Sample Testing:** Evaluate long-term performance on unseen market data.
* **Advanced Financial Constraints:** Incorporate more sophisticated constraints such as transaction costs, minimum investment amounts, or sector diversification.
* **Integration with Economic Models:** Explore more complex factor models or integrate with other economic theories.

## Contributing to the Project

We welcome contributions to enhance and expand this project! Please feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License Information

This project is open-source and made available under the **MIT License**.

The MIT License is a permissive free software license, meaning that you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to certain conditions. A copy of the license is typically included in a `LICENSE` file in the root of the repository.

## Acknowledgements

We extend our gratitude to the developers and communities of the following open-source projects, which are instrumental to this work:

* **PennyLane:** The quantum machine learning library that enables building and simulating quantum circuits.
* **Yfinance:** For providing convenient access to financial market data.
* **NumPy:** The fundamental package for numerical computing in Python.
* **Pandas:** For powerful data structures and data analysis tools.
* **Matplotlib:** For creating static, animated, and interactive visualizations.
* **Scikit-learn:** For machine learning utilities, specifically Principal Component Analysis.

This project is built upon the collective efforts of the open-source community.
