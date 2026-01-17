# Binance ML Training Platform – System Overview (Phase 5)

**Creation date:** 2025-11-26 06:52 (UTC+09:00)

This document summarizes the current state of the Binance ML Training Platform and identifies which design aspects are implemented versus still pending. The focus is on why each component exists, what it does conceptually, and how the main ideas are realized. Operational details such as setup or commands are intentionally omitted.

---

## 1. Configuration, Validation, and Security

### 1.1 Configuration as the Single Source of Truth

**Why**  
The configuration file is intended to be the single source of truth for data ranges, model structure, optimization, logging, and diagnostics. This supports reproducible experiments and controlled comparisons across environments.

**What**  
The configuration encodes, in a structured way:
- Data sources and time ranges.
- The asset universe and target asset.
- Temporal feature definitions.
- Model architecture and core training hyperparameters.
- Hyperparameter optimization settings.
- MLflow logging behavior.
- Evaluation options.
- Data diagnostics options.
- The run mode, distinguishing trial from production runs.

**How**  
At startup, the system:
- Loads configuration from YAML.
- Resolves environment variable placeholders for sensitive fields such as URIs and credentials.
- Validates the resulting structure against a schema that checks for required sections, required keys, and types.
- Enforces additional security validation by checking a list of required environment variables and failing fast if any are missing.

**Status:** Implemented.

### 1.2 Logging and Observability

**Why**  
The platform should make internal behavior transparent: which configuration paths are chosen, how data is shaped, and how models behave during training and evaluation.

**What**  
Logging captures, in human-readable form:
- Data loading progress and the number of samples obtained.
- Data diagnostics metrics and detected anomalies.
- Training configuration details and effective sample sizes.
- Hyperparameter optimization progress and best-trial information.
- Evaluation metrics and calibration diagnostics.
- MLflow integration outcomes, including failures to log.

**How**  
The system constructs a colored logging configuration from the YAML file, including:
- Log level selection.
- Color codes for different severities.
- Inclusion of function names to localize messages in the code base.

**Status:** Implemented.

---

## 2. Data Ingestion and Representation

### 2.1 DataObject Abstraction

**Why**  
A single in-memory abstraction allows all later stages (temporal features, diagnostics, training, evaluation) to be independent of the physical storage layer.

**What**  
The `DataObject` includes:
- **Metadata**: asset list, time range, number of samples, order book depth, and related indices.
- **Order books**: per-asset containers for raw rows and snapshot-level features.
- **Temporal features**: local and global feature matrices aligned with samples.
- **Targets**: labels and supporting quantities such as price changes and their prediction horizons (how far into the future the change is measured).

**How**  
The data loading stage:
- Builds a structurally valid `DataObject` from configuration metadata.
- Populates order-book rows from the time-series database.
- Defines the effective number of samples using the row count of the configured target asset.

**Status:** Implemented for baseline representation. Here "baseline representation" means the minimal but fully consistent structure described in this section, sufficient for the current training, evaluation, and diagnostics pipelines without additional external data sources or more complex target constructs.

### 2.2 GreptimeDB Ingestion and Multi-Source Time Split

**Why**  
Historical data may reside in different physical databases. The platform aims to present a logically continuous time series while maintaining clear boundaries between physical sources.

**What**  
The ingestion layer:
- Performs connectivity checks to configured GreptimeDB instances.
- Fetches raw rows for each asset over a global logical time range.
- When multi-database ingestion is enabled, divides the global range into non-overlapping time intervals, each mapped to one database connection.
- Aggregates rows per asset across these intervals into a unified, chronologically ordered series.

**How**  
Conceptually, the ingestion:
- Reads connection definitions and time ranges from configuration.
- Validates that connection time ranges do not overlap.
- Issues per-connection queries for each asset restricted to that connection’s interval.
- Concatenates and sorts per-asset rows, preserving chronological order.
- Propagates errors if time ranges overlap or assumptions about ordering are violated.

**Status:** Implemented (single and multi-database time-split modes).

**Optional, far-future ideas (not required for the current use case):**
- Alternative source types such as on-disk files, although the configuration anticipates a file-based mode.
- Integration with formal data versioning tooling, which would only become meaningful if the platform later evolves to persist derived statistical products (for example, long-term aggregates of means, variances, or percentiles) via a separate statistics-sinking component.

---

## 3. Temporal Features and Labeling

### 3.1 Temporal Features

**Why**  
Market behavior exhibits strong temporal regularities (for example, intraday patterns and session-dependent activity). Making these regularities explicit as features can stabilize learning and improve interpretability.

**What**  
The system constructs:
- **Local temporal features** per sample, such as hour-of-day, day-of-week, and minute-of-hour, typically encoded in a cyclic form.
- **Global temporal features**, including:
  - A "days since start" index to capture long-term trends.
  - A "market session" indicator that assigns each timestamp to a named session (e.g., Asian, European, American) based on configurable time windows.

**How**  
The temporal feature module:
- Normalizes timestamps into a consistent internal representation.
- Computes local and global features aligned with the sample index space.
- Attaches the resulting matrices to the `DataObject` as separate local and global arrays, with exactly one row per sample.
- Carefully maintains consistency between sample indices and underlying snapshot indices via anchor indices.

**Status:** Implemented.

### 3.2 Target Construction and Labeling Scheme

**Why**  
The objective is to classify future price movements into discrete intensity levels rather than predict a single scalar change. This allows separate treatment of upward and downward movements and supports more nuanced trading rules.

**What**  
The current labeling scheme uses:
- A prediction horizon defined in seconds.
- Percentage-based boundaries that partition price changes into several intensity bins.
- A two-headed classification output: one head for upward intensity and one for downward intensity.

**How**  
Conceptually:
- Price changes over the prediction horizon are mapped into categorical labels according to the configured percentage boundaries.
- For training, labels are converted into one-hot vectors for each head.
- The model’s output layer has two softmax heads, with the number of classes determined by the number of configured boundaries.

**Status:** Implemented.

---

## 4. Training Pipeline

### 4.1 Input Representation and Synthetic Fallback

**Why**  
The model expects a spatio-temporal tensor summarizing the top of the order book over a fixed visible window. However, not all datasets will expose snapshot-level features in the desired form.

**What**  
Training uses:
- A tensor of shape (samples, time, height, width, channels).
- Each sample represents a visible window discretized by a configured cadence.
- When snapshot features are absent, configurable fallback strategies are applied.

**How**  
The training pipeline:
- Constructs input tensors from top-of-book snapshot features for the target asset, using anchor indices to align snapshots with sample indices.
- When snapshot features are missing, chooses between failure, skipping training, or building synthetic inputs, depending on configuration.

**Status:** Implemented.

### 4.2 Temporal Feature Integration into Inputs

**Why**  
Combining temporal covariates directly with spatial order-book channels allows the model to exploit both microstructure and calendar-driven structure in a unified representation.

**What**  
When enabled, temporal features:
- Are concatenated along the channel dimension after broadcasting across time and spatial axes.
- Preserve the original temporal and spatial geometry of the input.

**How**  
Both training and evaluation pipelines:
- Concatenate selected local and global temporal feature matrices into a combined feature matrix.
- Validate that there is exactly one feature row per sample.
- Broadcast these features across the entire time–height–width grid.
- Append them as additional channels to the input tensor.

**Status:** Implemented.

### 4.3 Sample Weighting via Exponential Decay

**Why**  
Recent samples are often more informative than distant historical data. Exponential decay in sample weights provides a principled way to favour recent data while retaining older observations.

**What**  
The current design:
- Computes sample age in days from anchor timestamps.
- Applies an exponential decay with a configurable half-life.
- Uses these weights in the loss function for both classification heads.

**How**  
The training pipeline:
- Derives per-sample timestamps from snapshot timestamps and anchor indices.
- Computes ages relative to the most recent sample.
- Applies an exponential function with the configured half-life to obtain positive weights.
- Passes the resulting vector as sample weights to the training procedure for both output heads.
- Logs basic statistics (minimum, maximum, mean, standard deviation) of the weights to MLflow.

**Status:** Implemented.

### 4.4 Dataset Caching and Hashing

**Why**  
Recomputing identical datasets is costly and can introduce subtle inconsistencies between experiments. A dataset hash and cache support efficient reuse and traceability.

**What**  
The system:
- Computes a hash summarizing the effective dataset (training and validation inputs and labels).
- Optionally stores these arrays as a compressed file with a versioned name.
- Logs the dataset version and hash to MLflow.

**How**  
Conceptually:
- The effective training and validation arrays are combined into a single hash value.
- A caching utility writes these arrays into a compressed file in a configurable directory, using a filename pattern that encodes asset, model, dataset version, and dataset hash.
- MLflow receives the dataset hash, version, and cache path as parameters for later inspection.

**Status:** Implemented.

### 4.5 Model Architecture and Training Dynamics

**Why**  
A hybrid convolutional–recurrent architecture is appropriate for order-book sequences: convolutional layers capture local spatial structure, while recurrent layers capture temporal dependencies.

**What**  
The model:
- Applies convolutional layers over the spatial axes of the visible window.
- Summarizes temporal dynamics with a recurrent layer.
- Uses dense layers feeding two softmax heads for intensity classification.

**How**  
The training pipeline:
- Infers input shape from the constructed input tensor.
- Builds and compiles a CNN+LSTM model according to configuration parameters.
- Uses standard classification losses and metrics.
- Configures callbacks such as early stopping and learning-rate reduction from the YAML file.
- Executes training and records the training history.

**Status:** Implemented.

**Not yet implemented:**
- Fine-tuning from a previously trained model, despite configuration fields describing a base model.
- Class rebalancing strategies beyond time-based sample weighting, even though configuration anticipates class balancing.

---

## 5. Hyperparameter Optimization and Run Modes

### 5.1 Optuna-Based Hyperparameter Search

**Why**  
Manual tuning of high-dimensional hyperparameter spaces is inefficient. Automated search aims to maximize a validation metric under a fixed training procedure.

**What**  
The hyperparameter optimization component:
- Samples architectural and optimization hyperparameters from a bounded search space.
- Runs the full training pipeline for each trial.
- Uses a configurable validation metric to score trials.

**How**  
The system:
- Defines an objective function that clones the base configuration, applies trial-specific hyperparameters, runs the training pipeline, and reads the chosen metric from the training history via shared metadata.
- Invokes Optuna to perform multiple trials and select the best configuration.
- Logs hyperparameter optimization metadata (framework, number of trials, chosen metric, best values) to MLflow.

**Status:** Implemented.

### 5.2 Separation of Trial vs Production Behavior

**Why**  
Trial runs can generate many short-lived models. These should not pollute the model registry or clutter experiment artifacts. Only the final production training should produce long-lived artifacts.

**What**  
Two complementary mechanisms separate trial and production behavior:
- A per-trial model logging control within the hyperparameter optimization configuration.
- A global run mode distinguishing trial from production runs.

**How**  
Conceptually:
- During hyperparameter optimization, the objective function adjusts the MLflow configuration for each trial. When trial model logging is disabled, both trained-model logging and model registration are turned off for that trial.
- After optimization:
  - In trial mode, the system stops after completing the search, without performing a final training and evaluation run.
  - In production mode, the system takes the best configuration (if available) and runs a full training and evaluation pipeline, with normal model logging and optional registration.

**Status:** Implemented.

---

## 6. Data Diagnostics and Evaluation

### 6.1 Data Diagnostics

**Why**  
High-quality data is essential for meaningful modeling. Diagnostics are intended to uncover structural issues such as implausible spreads, negative prices, missing timestamps, and prominent outliers before training.

**What**  
The diagnostics stage:
- Samples a representative subset of training snapshots.
- Computes summary statistics for prices, quantities, and spreads.
- Quantifies anomalies (negative prices, inverted spreads, extreme spreads).
- Evaluates continuity of timestamps and the size of temporal gaps.
- Optionally exports sampled records, anomalies, and visual summaries.

**How**  
The diagnostics process:
- Draws samples from the training region using uniform or random strategies.
- Constructs per-snapshot records, including prices, quantities, optional labels, and timestamps.
- Computes metrics for spreads, anomalies, and gaps.
- Logs these metrics to MLflow.
- Writes CSV artifacts summarizing sampled data and detected anomalies.
- Optionally generates and logs time-series plots, histograms, and heatmaps according to configuration.

**Status:** Implemented.

### 6.2 Evaluation and Calibration

**Why**  
A structured evaluation phase is required to quantify both classification performance and the calibration of probabilistic predictions on held-out data.

**What**  
The evaluation component:
- Recomputes test indices using the same splitting logic as training.
- Builds evaluation inputs consistent with the training representation, including temporal features.
- Computes accuracy and macro-averaged precision, recall, and F1 scores for both heads.
- Optionally quantifies calibration through bin-based metrics.
- Logs confusion matrices and calibration curves as artifacts.

**How**  
The evaluator:
- Constructs evaluation tensors from snapshot features or synthetic inputs, mirroring the choices made in training.
- Applies the trained model to obtain class probabilities for both heads.
- Derives predicted classes and confusion matrices.
- Computes per-class and macro metrics for each head.
- Optionally computes calibration curves and derived metrics such as Brier score and expected calibration error.
- Logs scalar metrics and calibration artifacts to MLflow.

**Status:** Implemented.

**Not yet implemented:**
- Backtesting of trading strategies using model predictions, despite the presence of a configuration section anticipating this.
- Detailed portfolio and transaction cost modeling beyond scalar evaluation metrics.

---

## 7. Testing and Offline Validation

### 7.1 Unit and Integration Tests

**Why**  
The platform targets offline experimentation. Tests are required to ensure correctness of transformations and control logic without relying on live services.

**What**  
The test suite currently covers:
- Configuration loading and environment-variable resolution.
- Temporal feature construction and attachment.
- Chronological train/validation/test splitting.
- Multi-database ingestion logic and enforcement of non-overlapping time ranges.
- Sample-weighting behavior within the training pipeline.
- Core evaluation logic, including tensor construction and metric computation.

**How**  
Tests:
- Use mocks and stubs to simulate external services such as the database HTTP API.
- Construct small synthetic `DataObject` instances with controlled shapes and edge cases.
- Assert on shapes, ranges, and failure modes under invalid configurations.

**Status:** Implemented for the current feature set.

---

## 8. Summary of Pending Features

This section lists aspects that are either planned or anticipated by configuration but are not yet fully implemented in the code base. Items marked as far-future are not required for the current intended use:

- [Far-future, optional] Alternative data sources besides the current time-series database, such as file-based ingestion paths.
- [Far-future, optional] Integration with formal data versioning tools, primarily relevant if a dedicated statistics-sinking component is introduced.
- Class rebalancing strategies beyond time-based sample weighting.
- Fine-tuning or warm-start training from previously registered models.
- Backtesting of trading strategies and richer portfolio-level evaluation.
- Feature importance and interpretability analyses.
- Richer hyperparameter search spaces, including more model-specific parameters.

These items define the main frontier for subsequent phases of development.


---

## 9. Conceptual Architecture Layers

This section summarizes the intended separation of concerns between the current system and later trading-oriented components. The goal is to keep the present model focused on generating signals, while leaving capital allocation and execution effects to dedicated future layers.

### 9.1 Layer 1 – Signal Model

**Role**  
Map the current order-book and temporal context to probabilities over future price-move "intensity" at a fixed prediction horizon.

**Question answered**  
"Given what the market looks like now, how likely are different price-move scenarios at horizon H?"

**Relation to the code**  
This layer corresponds to the CNN+LSTM model and the associated training, hyperparameter optimization, diagnostics, and evaluation logic described in earlier sections. It defines the baseline representation in which:
- Inputs are order-book tensors enriched with temporal features.
- Outputs are probabilities over discrete up/down intensity classes at a specified horizon.

**Status**  
Implemented in Phase 5.

### 9.2 Layer 2 – Trading Policy (Signal → Trade Decisions)

**Role**  
Take the signal from Layer 1 and convert it into concrete trading decisions under explicit risk preferences.

**Question answered**  
"Given the model’s view of future price-move intensities and my risk constraints, should I trade now, in which direction, and with what position size?"

**Intended behaviour**  
This layer would:
- Consume per-sample probabilities from the signal model.
- Incorporate risk/reward metrics, base capital, and constraints such as maximum drawdown or maximum leverage.
- Decide when to open, close, or hold positions, and how much capital to allocate to each trade.

**Relation to the code**  
No dedicated trading policy module is implemented in Phase 5. The current system stops at signal generation and statistical evaluation. Policy logic is deliberately left as a separate, future component that will consume model outputs rather than modify the model itself.

### 9.3 Layer 3 – Execution and Market Dynamics Simulation

**Role**  
Model how trades produced by the policy are actually executed in a real market, taking into account latency, order-book depth, slippage, transaction costs, and stress conditions.

**Question answered**  
"If I follow this trading policy in a realistic market environment, what happens to my capital over time?"

**Intended behaviour**  
This layer would:
- Simulate the path of account equity under a sequence of trades suggested by Layer 2.
- Incorporate estimates of exchange latency, order placement and confirmation delays, and partial fills.
- Model explicit and implicit costs (fees, spreads, slippage) and possibly stress scenarios such as periods of heavy API saturation.
- Produce distributions of equity curves and risk metrics (for example, drawdowns and volatility) rather than only per-sample classification metrics.

**Relation to the code**  
Phase 5 does not implement this layer. The current evaluation focuses on per-sample prediction quality and basic calibration. A future simulation or backtesting subsystem is expected to build on top of the existing signal outputs and evaluation inputs, without requiring changes to the learned models as long as input/output semantics and horizons remain stable.
