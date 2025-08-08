# Quantum-Enhanced Portfolio Optimization: A Hybrid Factor-QAOA Approach

## Team Name
collective qubits

## Team Members
* **Saatwik chatkara** - WISER Enrollment ID: [gst-fnw9Qzp0LIIwBRy]
* **CaleB** - WISER Enrollment ID: [gst-p1RL3gpqF9OQSR2]

## Project Summary

In the dynamic and often volatile realm of financial markets, the core challenge of **portfolio optimization** lies in constructing an investment portfolio that maximizes returns for a given level of risk, or minimizes risk for a target return. While seminal works like Markowitz's Modern Portfolio Theory laid the groundwork, practical application faces significant hurdles. A primary difficulty stems from the **estimation of the covariance matrix** of asset returns, which is notoriously noisy, unstable, and prone to large errors due to limited historical data, market non-stationarity, and the "curse of dimensionality" when dealing with a large asset universe. These estimation errors can lead to portfolios that are theoretically optimal but perform poorly in real-world scenarios, exhibiting unexpected volatility or failing to capture true diversification benefits.

Our project addresses these fundamental challenges by pioneering a **robust hybrid quantum-classical framework** for portfolio selection, specifically focused on satisfying practical financial constraints, such as **cardinality constraints**, which mandate selecting a precise number of assets ($K$) for the portfolio. The methodology is structured around three interconnected pillars:

1.  **Classical Factor Model for Robust Covariance Estimation:** To overcome the limitations of empirical covariance, we first employ a **Principal Component Analysis (PCA)-based factor model**. This model posits that asset returns are primarily driven by a smaller set of common (systematic) macroeconomic or statistical factors, in addition to asset-specific (idiosyncratic) risk. By decomposing and reconstructing the covariance matrix using PCA, we effectively denoise the input, leading to a more stable and accurate risk assessment. This factor-based covariance matrix provides a more reliable foundation for the subsequent optimization problem.

2.  **Portfolio Optimization as a Quadratic Unconstrained Binary Optimization (QUBO) Problem:** The multi-objective portfolio selection problem (balancing risk, return, and enforcing cardinality) is meticulously transformed into a QUBO problem. This transformation is critical as QUBOs represent a canonical form for many combinatorial optimization problems, making them directly amenable to quantum algorithms. Each asset's inclusion in the portfolio is represented by a binary variable, $x_i \in \{0, 1\}$. The objective function is constructed to penalize high risk, reward high expected returns (adjusted by a risk-aversion parameter), and strongly penalize deviations from the target asset count. This formulation allows the entire problem to be encoded into a single quadratic cost function.

3.  **Quantum Approximate Optimization Algorithm (QAOA) for QUBO Solution:** The heart of our quantum approach lies in the **Quantum Approximate Optimization Algorithm (QAOA)**. QAOA is a leading hybrid quantum-classical algorithm designed for solving combinatorial optimization problems on near-term quantum computers (NISQ devices). We utilize PennyLane, a quantum machine learning library, to construct and execute the QAOA circuit. This variational algorithm iteratively refines a set of quantum gate parameters (angles) by minimizing the expectation value of a problem-specific "Cost Hamiltonian" – a quantum mechanical operator representation of our financial QUBO. The quantum circuit prepares a state whose measurement probabilities are peaked around the optimal portfolio configurations. For practical simulation, we employ the high-performance `lightning.qubit` simulator.

Crucially, the performance of QAOA is highly sensitive to its **hyperparameters** (e.g., number of QAOA layers, optimization learning rates, and the financial parameters like risk aversion and cardinality penalty strength). To address this, our project implements a systematic **hyperparameter tuning framework**. This involves a grid-search approach that explores various combinations of these parameters. For each combination, multiple independent QAOA optimizations are executed to enhance robustness against local minima. The best-performing quantum parameters are then used to sample numerous portfolio configurations, which are subsequently evaluated using classical financial metrics (e.g., Sharpe Ratio). This rigorous tuning process ensures that the identified quantum-derived portfolio represents the best possible solution within the defined parameter space.

The project culminates in a comprehensive analysis where the QAOA-optimized portfolio's performance is rigorously compared against a classical minimum variance portfolio benchmark. Key financial metrics such as annualized return, risk (standard deviation), and Sharpe Ratio are presented. Visualizations, including asset allocation bar charts and bitstring distribution plots, provide intuitive insights into the quantum solution. This project not only demonstrates the feasibility of applying hybrid quantum algorithms to complex financial challenges but also highlights their potential to uncover more efficient and diversified portfolios beyond the reach of traditional methods for certain problem structures.

## Project Presentation Deck

A detailed presentation covering the project's background, theoretical underpinnings, methodological deep-dive, implementation details, experimental results, and future work is available here:
**[Quantum-Enhanced_Portfolio_Optimization.pdf]**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
  - [Factor Model for Covariance Estimation: Enhancing Robustness](#factor-model-for-covariance-estimation-enhancing-robustness)
  - [Portfolio Optimization as a QUBO Problem: Bridging Classical and Quantum](#portfolio-optimization-as-a-qubo-problem-bridging-classical-and-quantum)
  - [QAOA Implementation with PennyLane: The Quantum Engine](#qaoa-implementation-with-pennylane-the-quantum-engine)
  - [Hyperparameter Tuning: Optimizing the Optimizer](#hyperparameter-tuning-optimizing-the-optimizer)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Configuration and Customization](#configuration-and-customization)
- [Understanding the Results and Analysis](#understanding-the-results-and-analysis)
- [Future Enhancements and Research Directions](#future-enhancements-and-research-directions)
- [Contributing to the Project](#contributing-to-the-project)
- [License Information](#license-information)
- [Acknowledgements](#acknowledgements)

## Features

This project offers a robust set of features meticulously designed for quantum-enhanced portfolio optimization:

* **Automated Financial Data Ingestion with `yfinance`:**
    * **Detail:** Seamlessly downloads historical adjusted close prices for a user-defined list of Exchange-Traded Funds (ETFs) or individual stock tickers. This leverages `yfinance`'s efficient API to fetch reliable market data.
    * **Benefit:** Ensures that the optimization process is fed with up-to-date and consistent financial information, providing a flexible and easily adaptable data foundation for different asset universes or timeframes.

* **Robust Factor Model Implementation via PCA:**
    * **Detail:** Employs Principal Component Analysis (PCA) on historical daily returns to identify latent, uncorrelated risk factors. These factors capture the majority of systematic market movements. The covariance matrix is then reconstructed using these factors, significantly reducing the impact of noise and estimation errors inherent in raw empirical covariance matrices.
    * **Benefit:** Produces a more stable and statistically robust covariance matrix, which is paramount for generating well-diversified and resilient portfolios, mitigating "error maximization" common in mean-variance optimization.

* **Classical Portfolio Optimization Benchmark:**
    * **Detail:** Implements a classical minimum variance portfolio optimization using `scipy.optimize.minimize` with the `SLSQP` method. This process identifies portfolio weights that minimize portfolio risk subject to a budget constraint (weights sum to 1) and non-negativity.
    * **Benefit:** Provides a quantifiable baseline performance metric (annualized return, risk, Sharpe ratio) against which the quantum-derived portfolios can be directly compared, highlighting any potential quantum advantage.

* **Precise QUBO Problem Formulation:**
    * **Detail:** Meticulously translates the multi-objective financial problem (balancing risk minimization, return maximization, and strict adherence to a cardinality constraint) into a Quadratic Unconstrained Binary Optimization (QUBO) matrix. The objective function is $C(\mathbf{x}) = \mathbf{x}^T \Sigma \mathbf{x} - q \cdot \boldsymbol{\mu}^T \mathbf{x} + \lambda (\sum_{i=1}^N x_i - K)^2$, where each $x_i$ is a binary decision variable.
    * **Benefit:** This formal mapping is crucial as QUBO is the standard input format for many combinatorial optimizers, including QAOA and quantum annealers, enabling the problem to be solved on quantum hardware or simulators.

* **PennyLane-Powered QAOA Core:**
    * **Detail:** Utilizes PennyLane, a leading differentiable quantum computing framework, to construct, execute, and optimize the QAOA circuit. This involves defining the problem-specific Cost Hamiltonian ($H_C$) from the QUBO matrix (mapping $x_i$ to $(I - Z_i)/2$) and a generic Mixer Hamiltonian ($H_M = \sum_i X_i$). The QAOA circuit consists of alternating applications of $e^{-i\gamma H_C}$ and $e^{-i\beta H_M}$ layers.
    * **Benefit:** Provides a flexible and high-level interface for implementing QAOA, abstracting away low-level quantum gate operations and allowing for efficient classical optimization of quantum circuit parameters.

* **Systematic Hyperparameter Tuning Framework:**
    * **Detail:** Implements a robust grid search algorithm to explore the multi-dimensional hyperparameter space. Key tunable parameters include the risk aversion coefficient (`q_risk_aversion`), the cardinality constraint penalty strength (`lambda_penalty`), the number of QAOA layers (`p_layers`), and the classical optimizer's learning rate (`stepsize`). For each unique combination, multiple independent QAOA runs (`num_qaoa_runs_per_hp_set`) are performed, each involving a set number of `optimization_steps_per_run` for the classical Adam optimizer. The best-performing result (lowest cost) from these runs is then selected.
    * **Benefit:** Addresses the critical challenge of QAOA's sensitivity to hyperparameters. This systematic approach enhances the likelihood of discovering near-optimal QAOA configurations, leading to more robust and higher-quality portfolio solutions compared to arbitrary parameter choices.

* **Comprehensive Performance Analysis & Visualization:**
    * **Detail:** Generates detailed output including the derived QUBO matrix, the constructed Hamiltonians, the full hyperparameter tuning results table, and the optimal QAOA portfolio's specific asset selections and equal weights (given the binary nature). Financial performance metrics (annualized return, risk, Sharpe ratio) are calculated. Visualizations include a bar chart of optimal portfolio asset allocation and a bitstring frequency distribution plot from the quantum sampling, providing insights into the quantum state's preferences. **Furthermore, it now includes the explicit printout of the QUBO matrix and Cost Hamiltonian for the *overall best* QAOA portfolio, along with a plot of its optimization cost history and a detailed evaluation of its top sampled bitstrings.**
    * **Benefit:** Offers a clear, quantitative, and intuitive understanding of the quantum solution's characteristics and its comparative advantage over classical benchmarks, with enhanced diagnostic plots for the best-performing solution.

* **Hybrid Quantum-Classical Architecture:**
    * **Detail:** The project exemplifies a practical hybrid computing paradigm. Classical processors are utilized for data fetching, factor model estimation, QUBO formulation, and the iterative optimization loop of QAOA's variational parameters. The quantum simulator (or potentially hardware) is dedicated to executing the quantum circuit for cost evaluation and sampling.
    * **Benefit:** Demonstrates a viable pathway for current quantum computing capabilities, where classical computers handle computationally expensive but non-quantum-specific tasks, offloading the combinatorial optimization to quantum resources.

## Methodology

This section delves into the technical core of the project's methodology, outlining the sequential steps involved in moving from raw financial data to a quantum-derived optimal portfolio.

### Factor Model for Covariance Estimation: Enhancing Robustness

The bedrock of our portfolio optimization is a stable and reliable asset covariance matrix ($\Sigma$). Empirical covariance matrices, derived directly from historical data, often suffer from several shortcomings:
1.  **Noise Accumulation:** With $N$ assets, the empirical covariance matrix requires estimating $N(N+1)/2$ unique parameters. If the number of historical observations ($T$) is not significantly larger than $N$, these estimates can be highly noisy and unstable, leading to "error maximization" in portfolio optimization.
2.  **Lack of Economic Interpretation:** Empirical correlations might not always reflect fundamental economic relationships.
3.  **Non-Stationarity:** Market dynamics change, making purely historical correlations less predictive for future periods.

To mitigate these issues, we implement a **statistical factor model** using Principal Component Analysis (PCA). The underlying assumption is that a substantial portion of asset return variance can be explained by a few common factors, with the remaining variance attributed to idiosyncratic (asset-specific) risk.

The steps are as follows:
1.  **Data Preparation:** Daily percentage returns are calculated for the chosen Vanguard ETFs.
2.  **PCA Application:** PCA is applied to the time series of these daily returns. PCA identifies orthogonal linear combinations of the original asset returns that capture the maximum variance. These linear combinations are our "factors."
3.  **Factor Loadings ($B$):** The principal components (eigenvectors of the covariance matrix of returns) are used to derive the factor loadings matrix, $B$. This $N \times K$ matrix (where $N$ is the number of assets and $K$ is the chosen number of factors) quantifies each asset's sensitivity to each factor. $B_{ij}$ represents how much asset $i$'s return responds to a one-unit change in factor $j$.
4.  **Factor Returns ($F$):** The original returns are projected onto the principal components to get the time series of the factor returns.
5.  **Factor Covariance Matrix ($\Sigma_F$):** The covariance matrix of these factor returns is calculated. Since PCA produces orthogonal components, $\Sigma_F$ will be a diagonal matrix with eigenvalues on the diagonal, representing the variance explained by each factor.
6.  **Specific Risk ($D$):** The idiosyncratic (asset-specific) risk is estimated by taking the diagonal elements of the original empirical covariance matrix and subtracting the variance explained by the factors for each asset. This is represented as a diagonal matrix $D$, meaning idiosyncratic risks are assumed uncorrelated with each other and with the common factors.
7.  **Covariance Reconstruction:** The final robust covariance matrix ($\Sigma$) is reconstructed using the factor model equation:
    $$
    \Sigma = B \Sigma_F B^T + D
    $$
    This reconstructed matrix is smoother, more stable, and less susceptible to the noise of raw historical estimates, providing a more reliable input for the subsequent quantum optimization.

### Portfolio Optimization as a QUBO Problem: Bridging Classical and Quantum

To harness the power of quantum computing, the portfolio selection problem must be rigorously mapped into a Quadratic Unconstrained Binary Optimization (QUBO) format. A QUBO problem seeks to minimize a quadratic objective function of binary variables:
$$
\text{Minimize } C(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}
$$
where $\mathbf{x}$ is a vector of binary decision variables ($x_i \in \{0, 1\}$).

Our specific portfolio optimization problem aims to:
1.  **Minimize Risk:** Represented by the portfolio variance, $\mathbf{x}^T \Sigma \mathbf{x}$.
2.  **Maximize Return:** Represented by the portfolio expected return, $\boldsymbol{\mu}^T \mathbf{x}$.
3.  **Enforce Cardinality Constraint:** Select exactly $K$ assets, i.e., $\sum_{i=1}^N x_i = K$.

These objectives are combined into a single cost function, which we seek to minimize:
$$
C(\mathbf{x}) = \mathbf{x}^T \Sigma \mathbf{x} - q \cdot \boldsymbol{\mu}^T \mathbf{x} + \lambda (\sum_{i=1}^N x_i - K)^2
$$
Here:
* $N$: Total number of available assets.
* $x_i \in \{0, 1\}$: Binary variable; $1$ if asset $i$ is selected, $0$ otherwise.
* $\Sigma$: $N \times N$ reconstructed covariance matrix from the factor model.
* $\boldsymbol{\mu}$: $N \times 1$ vector of expected asset returns.
* $q$: **Risk Aversion Coefficient**. This hyperparameter scales the importance of expected returns. A higher $q$ pushes the solution towards higher returns, potentially accepting more risk. A lower $q$ emphasizes risk reduction.
* $K$: **Target Number of Assets**. This defines the exact number of assets required in the portfolio.
* $\lambda$: **Penalty Coefficient**. This hyperparameter determines the strength of the penalty for violating the cardinality constraint. A sufficiently large $\lambda$ is crucial to ensure the constraint is met.

To transform this into the $\mathbf{x}^T Q \mathbf{x}$ QUBO form, we expand the terms. Notably, for binary variables, $x_i^2 = x_i$. The squared sum in the penalty term expands to $\sum_i x_i^2 + \sum_{i \ne j} x_i x_j = \sum_i x_i + \sum_{i \ne j} x_i x_j$. This allows us to collect all linear terms ($x_i$) into the diagonal elements of $Q$ ($Q_{ii}$) and all quadratic terms ($x_i x_j$) into the off-diagonal elements ($Q_{ij}$). The constant term ($\lambda K^2$) does not influence the optimal $\mathbf{x}$ and can be disregarded for optimization purposes.

The resulting QUBO matrix $Q$ is then constructed as:
* $Q_{ii} = \Sigma_{ii} - q \cdot \mu_i + \lambda (1 - 2K)$
* $Q_{ij} = \Sigma_{ij} + 2\lambda$ (for $i \neq j$, contributing to $Q_{ij}x_i x_j + Q_{ji}x_j x_i$, where $Q$ is symmetric)

This $Q$ matrix serves as the direct input for building the QAOA Cost Hamiltonian.

### QAOA Implementation with PennyLane: The Quantum Engine

The Quantum Approximate Optimization Algorithm (QAOA) is a powerful variational quantum algorithm for finding approximate solutions to combinatorial optimization problems on noisy intermediate-scale quantum (NISQ) devices. It operates in a hybrid quantum-classical loop: a classical optimizer iteratively adjusts parameters of a quantum circuit, which then generates a quantum state from which solutions can be sampled.

Our implementation leverages **PennyLane**, a differentiable quantum computing framework that integrates quantum hardware and simulators with popular classical machine learning libraries like NumPy and PyTorch.

1.  **Quantum Device Initialization:**
    We use `qml.device("lightning.qubit", wires=num_qubits, shots=10000)`.
    * `lightning.qubit`: This is PennyLane's high-performance C++ backend simulator, known for its speed in state-vector simulations.
    * `wires=num_qubits`: Each qubit represents an asset ($x_i$), so `num_qubits` is equal to `num_assets`.
    * `shots=10000`: This parameter is crucial for `qml.sample` measurements. It instructs the simulator to perform 10,000 repetitions of the quantum circuit and return the bitstring outcomes. While `lightning.qubit` can compute expectation values (`qml.expval`) analytically even with shots defined (for certain observables like Pauli products), the `shots` value is essential for obtaining statistically meaningful distributions from `qml.sample`.

2.  **Cost Hamiltonian ($H_C$) Construction:**
    The `build_qaoa_cost_hamiltonian` function takes the classically derived `Q_matrix` and translates it into a sum of Pauli operators. The mapping $x_i \rightarrow (I - Z_i)/2$ is applied to each binary variable. This means:
    * Linear terms ($Q_{ii} x_i$) become $Q_{ii} (I - Z_i)/2$.
    * Quadratic terms ($Q_{ij} x_i x_j$) become $Q_{ij} (I - Z_i)/2 (I - Z_j)/2 = Q_{ij}/4 (I - Z_i - Z_j + Z_i Z_j)$.
    The `qml.Hamiltonian` object is then simplified to combine like terms. This `H_cost` encodes the optimization problem's cost function into the energy levels of the quantum system.

3.  **Mixer Hamiltonian ($H_M$):**
    The mixer Hamiltonian is typically a simple sum of Pauli-X operators across all qubits: $H_M = \sum_{i=0}^{N-1} X_i$. This choice ensures that the quantum state can explore the entire solution space by creating superpositions and allowing transitions between all possible bitstring states. It is critical for the QAOA's ability to "walk" the quantum state towards the minimum energy configuration.

4.  **QAOA Circuit (`qaoa_circuit`):**
    This is a PennyLane QNode (`@qml.qnode(dev)`) representing the quantum ansatz.
    * **Initial State Preparation:** All qubits are initialized to an equal superposition state using Hadamard gates (`qml.Hadamard`). This ensures that every possible bitstring (portfolio combination) has a non-zero amplitude initially.
    * **Alternating Operators:** The core of QAOA involves applying `p_layers` (QAOA layers), each consisting of two types of unitary operations parameterized by angles $\gamma$ and $\beta$:
        * **Problem Unitary ($U_P(\gamma)$):** This is $e^{-i\gamma H_C}$. Its application imparts phases to the quantum state, with phases proportional to the cost of the corresponding bitstring. This effectively "marks" good solutions with distinct phases. The implementation manually applies $e^{-i 2 \cdot \text{coeff} \cdot \gamma[layer] \cdot \text{Pauli}}$ for each term of $H_C$. For single $Z$ terms, `qml.RZ` is used. For $Z_i Z_j$ terms, `qml.IsingZZ` is used.
        * **Mixer Unitary ($U_M(\beta)$):** This is $e^{-i\beta H_M}$. Its application mixes the amplitudes, allowing transitions between different bitstring states. This helps the algorithm escape local minima and explore the solution space. For $H_M = \sum X_i$, $e^{-i\beta X_i}$ is equivalent to `qml.RX(2*beta, wires=i)`.
    * **Measurement:** The `qaoa_circuit` returns `qml.expval(h_cost)`. This is the expectation value of the Cost Hamiltonian, which acts as the objective function for the classical optimizer.

5.  **Classical Optimization Loop:**
    An `AdamOptimizer` is chosen for its robustness and adaptive learning rate capabilities. The classical optimizer iteratively updates the QAOA parameters (`gamma` and `beta` angles) to minimize the cost returned by the `qaoa_circuit`. This feedback loop drives the quantum state towards the ground state of the Cost Hamiltonian, which corresponds to the optimal classical solution.

6.  **QAOA Sampling Circuits (`qaoa_sampling_circuit_internal` and `qaoa_sampling_circuit_final`):**
    * `qaoa_sampling_circuit_internal`: Used *within* the hyperparameter tuning loop. It uses the `dev` initialized with `shots=10000` (consistent with the main tuning device) to ensure robust statistics during the evaluation of each hyperparameter set.
    * `qaoa_sampling_circuit_final`: A separate QNode defined *after* tuning, also using `dev_final_sampling` with `shots=10000`. This is used for the final, high-quality bitstring distribution plot and detailed evaluation of the overall best QAOA portfolio. It ensures that the final presented results are based on robust statistics.

### Hyperparameter Tuning: Optimizing the Optimizer

QAOA's performance is profoundly influenced by its hyperparameters, which are external to the quantum circuit's variational parameters ($\gamma, \beta$). These hyperparameters are typically tuned classically. An exhaustive **grid search** is employed to systematically explore combinations of these critical parameters:

* **`q_risk_aversion_values`**: Values for the risk aversion coefficient ($q$) that balances the risk-return trade-off in the QUBO. Exploring this range helps identify portfolios with different desired risk profiles.
* **`lambda_penalty_values`**: Values for the penalty coefficient ($\lambda$) that enforces the cardinality constraint. This parameter is critical; if too small, the constraint might be violated; if too large, it might create an optimization landscape that is too steep and hard to navigate for the QAOA.
* **`p_layers_values`**: The number of QAOA layers ($p$). A higher $p$ generally increases the expressivity of the QAOA ansatz, allowing it to potentially reach better solutions, but also increases the number of variational parameters ($2p$) and the circuit depth, making optimization more challenging and computationally expensive.
* **`stepsize_values`**: The learning rate for the classical `AdamOptimizer`. An optimal step size is crucial for efficient convergence; too large might cause overshooting, while too small can lead to prohibitively slow convergence.

The tuning process iterates through every combination of these hyperparameters. For each combination:
1.  **QUBO Construction:** The `Q_matrix` and `H_cost` are re-built based on the current `q_risk_aversion` and `lambda_penalty`.
2.  **Multiple QAOA Runs:** `num_qaoa_runs_per_hp_set` independent QAOA optimizations are performed. This is a common practice to mitigate the problem of local minima in variational algorithms. Each run starts with a new set of random initial parameters.
3.  **Best Parameter Selection:** Within each set of multiple runs for a given hyperparameter combination, the `best_params_for_hp_set` (gamma and beta angles) corresponding to the lowest achieved `final_cost` are selected. The `cost_history` of this best run is also stored.
4.  **Solution Sampling and Evaluation:** The `qaoa_sampling_circuit_internal` is invoked using these `best_params_for_hp_set`. The resulting bitstrings are collected, counted, and then individually evaluated for their financial metrics (return, risk, Sharpe Ratio). Crucially, only bitstrings that satisfy the `K_target_assets` cardinality constraint are considered valid for portfolio evaluation.
5.  **Best Portfolio for HP Set:** The bitstring that yields the highest Sharpe Ratio among the valid sampled portfolios for that specific hyperparameter combination is identified.
6.  **Result Aggregation:** All results for each hyperparameter combination (including the best-found Sharpe Ratio, bitstring, calculated portfolio metrics, **the `best_params` themselves**, and the `cost_history` of the best run) are stored in a pandas DataFrame (`tuning_df`).

Finally, the `tuning_df` is analyzed to identify the overall best QAOA portfolio across the entire hyperparameter search space, which is the entry with the highest Sharpe Ratio.

## Installation Guide

To set up and run this project, ensure you have a Python 3.8+ environment. A virtual environment is highly recommended to manage dependencies cleanly.

1.  **Clone the Repository:**
    Navigate to your desired directory in your terminal and execute:
    ```bash
    git clone [https://github.com/QuantumQuants/quantum-portfolio-optimization.git](https://github.com/QuantumQuants/quantum-portfolio-optimization.git)
    cd quantum-portfolio-optimization
    ```

2.  **Create a Virtual Environment:**
    Inside the project's root directory:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
    Your terminal prompt should now show `(venv)` indicating the active environment.

4.  **Install Project Dependencies:**
    With the virtual environment activated, install all required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Instructions

Once the project dependencies are installed and the virtual environment is active, you can execute the main script to run the quantum-enhanced portfolio optimization pipeline.

1.  **Execute the Script:**
    From the project's root directory in your activated terminal:
    ```bash
    python main_portfolio_optimizer.py # Assuming your main script is named 'main_portfolio_optimizer.py'
    ```

2.  **Monitor Execution Progress:**
    The script will print real-time updates to your console, guiding you through each major stage of the pipeline:
    * **Data Acquisition and Preparation:** Confirmation of data download and initial rows of calculated returns.
    * **Factor Analysis:** Details of factor loadings, explained variance, and the reconstructed covariance matrix.
    * **Classical Benchmark:** Weights and performance metrics of the classical minimum variance portfolio.
    * **QAOA Hyperparameter Tuning:** A log indicating the start of each hyperparameter combination being evaluated, including `q`, `lambda`, `p`, and `stepsize`. Due to the nature of optimization and sampling, these logs will update as each combination completes.
    * **Final Results:** Summary of the overall best QAOA portfolio, including optimal hyperparameters, selected assets, and comparative performance metrics.

3.  **Interact with Plots:**
    During execution, the script will generate and display multiple `matplotlib` plots:
    * **QAOA Optimization Cost History:** Shows the convergence of the cost function for the best-performing QAOA run.
    * **Optimal Portfolio Asset Allocation:** A bar chart visualizing the proportional weights of assets selected by the best QAOA portfolio.
    * **Distribution of Sampled Bitstrings:** A bar chart illustrating the frequency of the top (most probable) asset selection bitstrings obtained from the QAOA sampling circuit.
    You may need to close each plot window for the script to continue to the next stage or finalize its execution.

## Configuration and Customization

The project is designed with flexibility in mind, allowing users to easily modify core parameters directly within the `main_portfolio_optimizer.py` script to explore different scenarios or optimize for specific objectives.

* **Asset Universe:**
    ```python
    vanguard_tickers = ["VOO", "VTI", "BND", "VXUS", "VGT"]
    ```
    This list can be modified to include any other stock or ETF tickers supported by `yfinance`. Ensure that the chosen assets have sufficient historical data for the specified `start_date` and `end_date`.

* **Historical Data Range:**
    ```python
    start_date = "2018-01-01"
    end_date = "2023-12-31"
    ```
    Adjust these dates to analyze different market periods or to leverage a longer/shorter data history. Be mindful that very long periods might introduce issues of non-stationarity in asset returns, while very short periods might not provide enough data for robust statistical analysis.

* **Factor Model Parameters:**
    ```python
    num_factors = 2 # For 5 assets, 2-3 factors is a reasonable starting point.
    ```
    This integer determines the number of principal components to extract in the PCA step. The choice of `num_factors` is a trade-off: too few might not capture enough systematic risk, while too many can reintroduce noise or overfit to historical patterns. A good starting point is often to select enough factors to explain 80-90% of the total variance in returns, as indicated by `pca.explained_variance_ratio_.sum()`.

* **Portfolio Cardinality Constraint:**
    ```python
    K_target_assets = 2 # Example: select 2 out of 5 assets
    ```
    This integer defines the exact number of assets that the optimized portfolio should contain. The QAOA's objective function (via the `lambda_penalty`) is designed to strongly enforce this constraint.

* **QAOA Hyperparameter Search Space:**
    ```python
    q_risk_aversion_values = [0.1, 0.5, 1.0] # Original full range
    lambda_penalty_values = [5.0, 10.0] # Original full range
    p_layers_values = [1, 2] # Original full range
    stepsize_values = [0.01, 0.05] # Original full range
    ```
    These lists define the discrete values that the hyperparameter tuning process will iterate through.
    * **`q_risk_aversion_values`**: Controls the balance between maximizing return and minimizing risk in the QUBO objective. A higher `q` will emphasize returns more.
    * **`lambda_penalty_values`**: Determines how severely violations of the `K_target_assets` constraint are penalized. A value that is too low might result in portfolios not meeting the cardinality, while too high can make the optimization landscape difficult to traverse.
    * **`p_layers_values`**: The number of QAOA layers. More layers ($p$) generally enhance the circuit's ability to approximate the optimal solution but exponentially increase the number of variational parameters ($2p$) and computational cost.
    * **`stepsize_values`**: The learning rate for the classical `AdamOptimizer`. This parameter controls the magnitude of updates to the QAOA angles during gradient descent.

* **QAOA Optimization Run Parameters:**
    ```python
    num_qaoa_runs_per_hp_set = 3 # Original value
    optimization_steps_per_run = 50 # Original value
    ```
    These parameters control the duration and robustness of the QAOA optimization for *each* hyperparameter combination during the tuning process.
    * **`num_qaoa_runs_per_hp_set`**: The number of independent QAOA optimizations performed for each unique set of hyperparameters. Increasing this (e.g., to 3 or 5) improves the chance of escaping local minima and finding a better optimal `(gamma, beta)` parameter set for a given `(q, lambda, p, stepsize)` combination.
    * **`optimization_steps_per_run`**: The number of gradient descent iterations performed by the `AdamOptimizer` for each QAOA run. Increasing this (e.g., to 50 or 100) allows the optimizer more time to converge to a lower cost, potentially yielding better quantum states.

* **Quantum Device Shots:**
    ```python
    dev = qml.device("lightning.qubit", wires=num_qubits, shots=10000) # Original value
    # ...
    final_sampling_shots_value = 10000
    dev_final_sampling = qml.device("lightning.qubit", wires=num_qubits, shots=final_sampling_shots_value)
    ```
    The `shots` parameter dictates how many times the quantum circuit is measured when `qml.sample` is called. Both the device used during tuning and for final analysis are set to `10000` shots for robust statistical results.

## Understanding the Results and Analysis

The project provides a comprehensive output designed to facilitate a deep understanding of the quantum-enhanced portfolio optimization process and its efficacy.

1.  **Classical Minimum Variance Portfolio:**
    This section presents the computed asset weights and key performance metrics (annualized return, annualized risk/standard deviation, and Sharpe Ratio) for a portfolio optimized using traditional, purely classical methods to minimize variance. This serves as a vital **benchmark** against which the performance of the quantum-derived portfolios can be directly evaluated. It highlights the baseline achievable without quantum computational advantages.

2.  **Hyperparameter Tuning Results (`tuning_df` DataFrame):**
    This detailed pandas DataFrame is the core output of the automated tuning process. Each row in this table represents a unique combination of `q_risk_aversion`, `lambda_penalty`, `p_layers`, and `stepsize` that was evaluated. Key columns provide crucial insights:
    * `q_risk_aversion`, `lambda_penalty`, `p_layers`, `stepsize`: The specific hyperparameter set for that row.
    * `final_cost`: The minimum cost function value achieved by the QAOA optimization for this specific hyperparameter set. A lower cost indicates better optimization performance in terms of the QUBO objective.
    * `best_bitstring`: The binary string (e.g., '01101') that represents the asset selection found among the sampled solutions for this hyperparameter set, which yielded the highest Sharpe Ratio *and* satisfied the `K_target_assets` cardinality constraint.
    * `best_sharpe`, `best_return`, `best_risk`: The annualized Sharpe Ratio, annualized return, and annualized standard deviation (risk) calculated for the portfolio corresponding to the `best_bitstring`. These are the ultimate financial performance indicators.
    * `best_weights`: The allocated weights for each asset in this portfolio.
    * `cost_history`: The list of cost values over optimization steps for the best QAOA run within this hyperparameter set.
    * `best_params`: The optimized gamma and beta angles for the best QAOA run within this hyperparameter set.
    This table allows for granular comparison across different tuning configurations.

3.  **Overall Best QAOA Portfolio:**
    Following the completion of the full hyperparameter tuning, the `tuning_df` is filtered to exclude invalid portfolios (those not meeting constraints) and then sorted by Sharpe Ratio. The entry with the **highest `best_sharpe`** is identified as the "Overall Best QAOA Portfolio." This represents the most financially attractive portfolio discovered within the entire hyperparameter search space. Its specific hyperparameters, selected assets, and detailed performance metrics are then printed.

4.  **QUBO Matrix (Q) and Cost Hamiltonian (H_cost) for Overall Best QAOA Portfolio:**
    For the overall best QAOA portfolio, the corresponding QUBO matrix and its mapped Cost Hamiltonian are explicitly recalculated and printed. This provides a direct view of the specific optimization problem instance that QAOA successfully solved.

5.  **QAOA Optimization Cost History for Overall Best QAOA Portfolio:**
    A dedicated plot shows the convergence of the cost function over optimization steps for the QAOA run that yielded the overall best portfolio. This visualizes the optimization process and how the cost was minimized.

6.  **Optimal Portfolio Weights (QAOA Result - Table & Chart):**
    * **Table:** A clear tabular display lists each asset and its corresponding optimal weight as determined by the best QAOA portfolio. Due to the binary selection nature and equal weighting among selected assets in this model, weights will typically be $1/K$ for selected assets and $0$ for unselected ones.
    * **Bar Chart:** A visual bar chart graphically represents these optimal weights. This provides an intuitive understanding of the asset allocation strategy recommended by the quantum approach, making it easy to see which assets are included and their proportional contribution.

7.  **Detailed Evaluation of Top Bitstrings from Final Sampling:**
    A table is presented showing the top most frequently sampled bitstrings from the `qaoa_sampling_circuit_final` (which uses `10000` shots). For each of these bitstrings, its corresponding assets, number of assets, return, risk, and Sharpe Ratio are calculated and displayed. This offers a deeper dive into the solutions that the quantum state favors, beyond just the single best one.

8.  **Bitstring Distribution Plot (Full Shots):**
    This bar chart visualizes the empirical probability distribution of bitstrings sampled from the final quantum state of the QAOA circuit, specifically using the optimal parameters found for the "Overall Best QAOA Portfolio" and a high number of shots (`10000`). This plot provides a statistically robust insight into the "quantum solution space" and the algorithm's convergence characteristics, showing which asset combinations are most probable.

9.  **Portfolio Performance Comparison (Classical vs. QAOA):**
    A concise comparison table provides a side-by-side view of the key financial performance metrics (Annualized Return, Annualized Risk, Sharpe Ratio) for both the `Classical Min Variance` portfolio and the `QAOA Portfolio`. This table is the culminating point of the analysis, offering a direct, quantitative comparison that highlights:
    * Whether the quantum approach achieved superior risk-adjusted returns (a higher Sharpe Ratio).
    * How it balanced return generation against risk minimization compared to a purely risk-focused classical strategy.
    This comparison is crucial for assessing the practical value and potential advantage of employing quantum optimization for this financial problem.

## Future Enhancements and Research Directions

This project lays a robust foundation for quantum-enhanced portfolio optimization. Several exciting and impactful avenues exist for future work, pushing the boundaries of this application:

* **Advanced Hyperparameter Optimization Strategies:**
    * **Beyond Grid Search:** While grid search is systematic, it can be computationally expensive for high-dimensional hyperparameter spaces. Investigate more efficient classical optimization algorithms for QAOA parameters, such as **Bayesian Optimization**, **Genetic Algorithms**, **Random Search**, or **Reinforcement Learning-based approaches**. These methods can explore the parameter space more intelligently, potentially finding better optima with fewer evaluations.
    * **Adaptive Tuning:** Implement dynamic or adaptive methods where QAOA parameters are adjusted on-the-fly based on the real-time performance of the quantum circuit.

* **Exploring More Sophisticated QAOA Ansätze:**
    * **Custom Mixer Hamiltonians:** Experiment with alternative mixer Hamiltonians beyond the standard sum of Pauli-X gates (e.g., `qml.mixing.XYMixer()`, or problem-specific mixers). Different mixers might lead to faster convergence or better exploration of the solution space for specific QUBO structures.
    * **Higher Layer Depths (`p`):** Investigate the impact of increasing the number of QAOA layers (`p`). While this increases computational cost, a deeper circuit might be able to find better approximations to the true optimal solution. Techniques for mitigating barren plateaus (vanishing gradients) would be relevant here.
    * **Parameter Initialization:** Explore more sophisticated strategies for initializing QAOA parameters (e.g., warm-starting from classical solutions, or using angles found from previous similar problems or datasets).

* **Integration with Other Quantum Algorithms:**
    * **Variational Quantum Eigensolver (VQE):** As a more general variational algorithm for finding ground states of Hamiltonians, VQE with different ansätze (e.g., Hardware-Efficient Ansätze, Unitary Coupled Cluster (UCC), Unitary Coupled Cluster Singles and Doubles (UCCSD)) could be explored as an alternative to QAOA for solving the QUBO problem.
    * **Quantum Annealing:** For problems directly mappable to QUBOs, dedicated quantum annealers (e.g., D-Wave systems) offer a different computational paradigm. Integrating with D-Wave's Ocean SDK could provide an alternative, potentially faster, hardware-native approach for the QUBO solution.
    * **Grover's Algorithm:** While not directly solving QUBOs, aspects of amplitude amplification could be integrated into search routines for optimal bitstrings.

* **Transition to Continuous Portfolio Weights:**
    * The current model provides a binary selection (asset included or not). For real-world applications, continuous portfolio weights are often required. Future work could explore methods to handle continuous weights:
        * **Amplitude Encoding:** Encoding weight values into the amplitudes of quantum states, though this can be challenging for arbitrary precision and non-normalized weights.
        * **Basis Encoding with Multiple Qubits:** Representing each asset's weight using multiple qubits, where a higher number of qubits per asset allows for finer granularity in weight allocation. This rapidly increases the total qubit count.
        * **Hybrid Integer/Continuous Optimization:** Developing more sophisticated hybrid algorithms where quantum optimization handles the combinatorial asset selection, and classical optimization then determines the optimal continuous weights for the selected assets.

* **Execution on Real Quantum Hardware (QPUs):**
    * Transitioning from `lightning.qubit` simulator to actual quantum processing units (QPUs) available through cloud platforms (e.g., IBM Quantum, Amazon Braket, Microsoft Azure Quantum via PennyLane's various device plugins).
    * This would introduce real-world challenges such as **quantum noise, limited qubit connectivity, finite coherence times, and gate fidelities**. Research into **error mitigation techniques** (e.g., zero-noise extrapolation, measurement error mitigation) would be crucial to achieve reliable results on current hardware.

* **Robustness Testing and Out-of-Sample Evaluation:**
    * Crucially, validate the long-term performance and stability of the quantum-optimized portfolios on **out-of-sample (unseen) market data**. This simulates real trading conditions and helps assess the generalizability and practical utility of the optimized portfolios beyond the training period.
    * Perform sensitivity analysis of the optimal portfolios to small changes in input parameters ($\mu$, $\Sigma$).

* **Incorporation of Advanced Financial Constraints and Features:**
    * **Transaction Costs:** Model the cost associated with buying and selling assets, influencing portfolio rebalancing decisions.
    * **Minimum Investment Amounts/Lot Sizes:** Account for practical constraints where assets can only be traded in specific quantities.
    * **Liquidity Constraints:** Factor in the ease or difficulty of executing large trades without significantly impacting market prices.
    * **Sector/Industry Diversification:** Add constraints to ensure the portfolio is diversified across specific economic sectors or industries, preventing over-concentration.
    * **Tax Implications:** Consider capital gains/losses and their impact on portfolio performance.

* **Integration with Economic Models:**
    * Explore more complex and predictive factor models beyond PCA (e.g., macroeconomic factor models, industry factor models) to enhance the quality of expected returns and covariance matrix inputs.
    * Combine with Bayesian methods for parameter estimation to incorporate prior beliefs and handle uncertainty more rigorously.

## Contributing to the Project

We enthusiastically welcome contributions from the community to improve, expand, and refine this project! If you have ideas, bug fixes, or new features, please don't hesitate to get involved.

* **Bug Reports:** If you encounter any issues, errors, or unexpected behavior, please open a detailed issue on the GitHub repository. Provide clear steps to reproduce the bug, error messages, and your environment details.
* **Feature Requests:** Have a new idea for a feature, a different optimization algorithm, or an additional financial constraint you'd like to see implemented? Open an issue to discuss your proposal.
* **Code Contributions:**
    1.  **Fork the repository:** Create your own copy of the project on GitHub.
    2.  **Create a new branch:** For each new feature or bug fix, create a dedicated branch (e.g., `git checkout -b feature/your-awesome-feature` or `git checkout -b bugfix/fix-issue-123`).
    3.  **Implement your changes:** Write clean, well-commented code. Follow existing code style.
    4.  **Write Tests:** If applicable, add or update unit tests to cover your new functionality or bug fix.
    5.  **Update Documentation:** If your changes affect how the project is used, update the relevant sections of this README or create new documentation files.
    6.  **Commit your changes:** Use clear and concise commit messages.
    7.  **Push your branch:** Push your local branch to your forked repository.
    8.  **Open a Pull Request (PR):** Submit a pull request from your forked repository to the `main` branch of this project. Clearly describe the purpose of your PR and the changes you've made.

## License Information

This project is open-source and made available under the **MIT License**.

The MIT License is a highly permissive free software license. It grants users the freedom to:
* Use, copy, modify, and merge the software.
* Publish, distribute, sublicense, and sell copies of the software.
These freedoms are granted with minimal restrictions, primarily requiring the inclusion of the original copyright and license notice. A copy of the license text is typically found in the `LICENSE` file in the root of the repository. This choice of license encourages broad adoption and collaboration.

## Acknowledgements

We extend our sincere gratitude to the developers and vibrant communities behind the following open-source projects. This work would not have been possible without their foundational contributions:

* **PennyLane:** The cornerstone quantum machine learning library that empowers the development and simulation of quantum circuits for variational algorithms.
* **Yfinance:** For its invaluable service in providing accessible and reliable historical financial market data, enabling robust analysis.
* **NumPy:** The indispensable library for numerical computing in Python, offering powerful array objects and mathematical functions.
* **Pandas:** For its unparalleled data structures (`DataFrame`) and high-performance data analysis tools, crucial for handling and manipulating financial time series.
* **Matplotlib:** The versatile plotting library that facilitates the creation of high-quality static, animated, and interactive visualizations to interpret complex results.
* **Scikit-learn:** A comprehensive and widely used machine learning library, specifically for its Principal Component Analysis (PCA) implementation, vital for our factor model.

This project stands as a testament to the power of open collaboration and shared knowledge within the scientific and software development communities.

