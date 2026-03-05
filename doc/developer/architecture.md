# mlpack Architecture Reference

> **Purpose**: Reverse-engineered architectural knowledge extracted bottom-up
> from the mlpack source tree.  Every entry answers: what is it, what does it
> depend on, what depends on it, which execution layer it belongs to, which
> invariants it assumes, when you would build it, and what breaks without it.
>
> **Classification buckets** used throughout this document:
> - **Core Primitive** – must exist for anything to work
> - **Orchestration / Control** – wires primitives together into a pipeline
> - **Performance Optimization** – speeds things up but not semantically required
> - **Developer Ergonomics** – makes the library easier to use correctly
> - **Debug / Visualization** – aids diagnosis; never in a hot path
> - **Experimental / Optional** – useful but not load-bearing

---

## 1. Foundation Layer (memory / tensor / configuration)

---

FILE: `src/mlpack/base.hpp`
ROLE: Absolute bootstrap header; pulls in standard C++ headers, Armadillo, and
      compiler-specific inline/OpenMP guards.  Every other mlpack file is
      downstream.
DEPENDS ON: `<armadillo>`, `config.hpp` (or `MLPACK_CUSTOM_CONFIG_FILE`),
            `core/util/arma_traits.hpp`, `core/util/omp_reductions.hpp`,
            `core/arma_extend/find_nan.hpp`
USED BY: `prereqs.hpp` → everything in mlpack
CORE OR AUX: **Core Primitive**
INVARIANTS: C++17 is available; Armadillo ≥ 9.x; M_PI is defined after
            inclusion; `mlpack_force_inline` is resolved to a valid attribute.
REBUILD PHASE: **v0** – the very first file you write; nothing compiles without it.
NOTES FOR FLUXIONS: If you ever swap Armadillo for another linear-algebra
                    backend, the swap lives here.  Keep the `mlpack_force_inline`
                    macro; it materially affects throughput on GCC/Clang hot paths.

---

FILE: `src/mlpack/config.hpp`
ROLE: CMake-generated (or shipped) compile-time feature flags
      (`MLPACK_HAS_STB`, `MLPACK_HAS_BFD_DL`, version strings, etc.).
DEPENDS ON: CMake configure step
USED BY: `base.hpp`, scattered `#ifdef MLPACK_HAS_*` guards
CORE OR AUX: **Core Primitive**
INVARIANTS: Always present after installation; never edited by hand.
REBUILD PHASE: **v0** – generate with CMake's `configure_file`.
NOTES FOR FLUXIONS: Do not hard-code feature flags; let CMake detect them.

---

FILE: `src/mlpack/prereqs.hpp`
ROLE: Second-level bootstrap; adds cereal serialisation includes and the
      `size_checks.hpp` utility on top of `base.hpp`.
DEPENDS ON: `base.hpp`, cereal headers, `core/cereal/*`, `core/arma_extend/*`,
            `core/data/has_serialize.hpp`, `core/util/size_checks.hpp`
USED BY: `core.hpp` (and transitively every mlpack header)
CORE OR AUX: **Core Primitive**
INVARIANTS: Cereal is available; Armadillo matrices are serialisable after
            inclusion.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The split between `base.hpp` (no cereal) and `prereqs.hpp`
                    (cereal added) is intentional: `base.hpp` can be used in
                    environments without cereal.

---

FILE: `src/mlpack/core.hpp`
ROLE: Single umbrella include for the entire `core/` subsystem; the conventional
      entry point for every mlpack method.
DEPENDS ON: `prereqs.hpp`, `core/stb/*`, `core/httplib/*`, `core/util/*`,
            `core/data/*`, `core/math/*`, `core/distances/*`,
            `core/distributions/*`, `core/kernels/*`, `core/metrics/*`,
            `core/tree/*`, `core/cv/*`, `core/hpt/*`
USED BY: Every `methods/` header, every binding, every test
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Including this header is sufficient to write any mlpack algorithm.
REBUILD PHASE: **v0** – but trivial; just forward-includes.
NOTES FOR FLUXIONS: Keep this as a convenience umbrella only.  Individual
                    subsystems already manage their own include chains.

---

## 2. Armadillo Extension / Serialisation Glue

---

FILE: `src/mlpack/core/arma_extend/find_nan.hpp`
ROLE: Adds `find_nan()` / `find_nonfinite()` utilities to Armadillo's namespace
      for safe numeric handling.
DEPENDS ON: `<armadillo>`
USED BY: `base.hpp`, any code that needs NaN detection in matrices
CORE OR AUX: **Core Primitive**
INVARIANTS: Only used on Armadillo matrix types; result type mirrors
            `arma::uvec`.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: Armadillo itself lacks these; they are critical for safe
                    gradient computations.

---

FILE: `src/mlpack/core/arma_extend/serialize_armadillo.hpp`
ROLE: Teaches cereal how to archive Armadillo matrices, cubes, and sparse
      matrices; without this, models cannot be saved/loaded.
DEPENDS ON: cereal, `<armadillo>`
USED BY: `prereqs.hpp`, all `serialize()` methods in layers and models
CORE OR AUX: **Core Primitive**
INVARIANTS: Cereal archive types (JSON, binary, XML, portable-binary) are all
            supported; sparse matrices require separate handling.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: Any custom matrix type (e.g. GPU tensor) must add its own
                    cereal specialisation here or in an analogous file.

---

FILE: `src/mlpack/core/cereal/pointer_wrapper.hpp`
ROLE: Wraps raw pointers so cereal can serialise polymorphic layer hierarchies
      without requiring `shared_ptr`.
DEPENDS ON: cereal
USED BY: ANN layer serialisation (`methods/ann/layer/serialization.hpp`)
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Pointer is non-null at serialise time; ownership is managed by the
            caller (not cereal).
REBUILD PHASE: **v1**
NOTES FOR FLUXIONS: The design choice to keep layers as raw pointers (not
                    `unique_ptr`) is intentional for performance.  The wrapper
                    preserves that choice while still supporting serialisation.

---

FILE: `src/mlpack/core/cereal/low_precision.hpp`
ROLE: Allows models to be serialised with reduced floating-point precision
      (e.g., `float` instead of `double`) to shrink checkpoint files.
DEPENDS ON: cereal, `<armadillo>`
USED BY: Optional; included by users who want compact checkpoints
CORE OR AUX: **Performance Optimization**
INVARIANTS: Precision loss is one-way; loading a low-precision archive into a
            `double` model is safe but lossy.
REBUILD PHASE: **v2**

---

## 3. Utilities

---

FILE: `src/mlpack/core/util/log.hpp` / `log_impl.hpp`
ROLE: Provides `Log::Debug`, `Log::Info`, `Log::Warn`, `Log::Fatal`
      output streams with per-level compile-time suppression.
DEPENDS ON: `prefixedoutstream.hpp`, `nulloutstream.hpp`
USED BY: Every algorithm in `methods/`, all bindings
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: `Log::Fatal` calls `std::exit()` when a newline is flushed; all
            other levels are no-ops unless the corresponding `MLPACK_PRINT_*`
            macro is defined (library usage) or the binding framework enables
            them.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: Replace `NullOutStream` with a thread-local buffer if you
                    need async logging.

---

FILE: `src/mlpack/core/util/timers.hpp` / `timers_impl.hpp`
ROLE: High-resolution wall-clock timer keyed on string labels; used to report
      per-phase timing from CLI bindings.
DEPENDS ON: `<chrono>`, `log.hpp`
USED BY: CLI bindings, optionally methods that want timing output
CORE OR AUX: **Debug / Visualization**
INVARIANTS: Timers are not thread-safe by default; each binding runs in a single
            thread for timing purposes.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/util/params.hpp` / `params_impl.hpp`
ROLE: The `Params` class stores all typed parameter values for a single binding
      invocation; it is the runtime parameter bag passed between the binding
      framework and the algorithm.
DEPENDS ON: `param_data.hpp`, `binding_details.hpp`
USED BY: All bindings (CLI, Python, Julia, Go, R), `IO::Parameters()`
CORE OR AUX: **Orchestration / Control**
INVARIANTS: Parameter names are unique within a binding; all `Get<T>()` calls
            are type-safe; the `FunctionMapType` dispatch table is populated at
            static-init time.
REBUILD PHASE: **v1**
NOTES FOR FLUXIONS: This is the central handshake point between the user-facing
                    binding API and the C++ algorithm.  Keep it decoupled from
                    any particular binding format.

---

FILE: `src/mlpack/core/util/mlpack_main.hpp`
ROLE: Dispatcher that `#include`s the correct binding-type-specific main header
      based on `BINDING_TYPE` preprocessor constant.
DEPENDS ON: `bindings/{cli,python,julia,go,R,markdown,tests}/mlpack_main.hpp`
USED BY: Every `*_main.cpp` file for each algorithm
CORE OR AUX: **Orchestration / Control**
INVARIANTS: `BINDING_NAME` must be defined before including this file;
            `BINDING_TYPE` defaults to `BINDING_TYPE_UNKNOWN` (compile error).
REBUILD PHASE: **v1**
NOTES FOR FLUXIONS: This single-file dispatch is the linchpin of the multi-
                    language binding system.  Adding a new target language
                    requires only a new `BINDING_TYPE_*` constant and a new
                    branch here.

---

FILE: `src/mlpack/core/util/io.hpp` / `io_impl.hpp`
ROLE: Global `IO` singleton that owns the registry of all binding `Params`
      objects; acts as the DI container for the parameter system.
DEPENDS ON: `params.hpp`, `program_doc.hpp`
USED BY: Bindings at startup/teardown; `Params` retrieval
CORE OR AUX: **Orchestration / Control**
INVARIANTS: Only one `IO` instance per process; parameter registration happens
            at static-init time.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/util/size_checks.hpp`
ROLE: Validates that dataset dimensions are consistent with algorithm
      assumptions (e.g., same number of points in features and labels).
DEPENDS ON: `log.hpp`
USED BY: `prereqs.hpp`, essentially all supervised methods
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Emits a fatal error (via `Log::Fatal`) on mismatch; never silently
            truncates data.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/util/arma_traits.hpp`
ROLE: Type-trait helpers (`IsArma<T>`, `GetColType<T>`, `GetURowType<T>`, etc.)
      that let template code branch on whether a type is an Armadillo matrix,
      sparse matrix, cube, or row vector.
DEPENDS ON: `<armadillo>`
USED BY: `base.hpp`, ANN framework, nearly all template algorithms
CORE OR AUX: **Core Primitive**
INVARIANTS: Traits are purely compile-time; zero runtime cost.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/util/ens_traits.hpp`
ROLE: Type-trait helpers for the ensmallen optimizer library, enabling
      algorithms to detect whether an optimizer supports gradients,
      constraints, etc.
DEPENDS ON: `<ensmallen.hpp>`, `arma_traits.hpp`
USED BY: ANN training (FFN, RNN), optimizable methods
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Traits are compile-time only.
REBUILD PHASE: **v1**

---

## 4. Mathematics Primitives

---

FILE: `src/mlpack/core/math/math.hpp` (and constituent files)
ROLE: Portable mathematical utilities not in `<cmath>` or Armadillo: digamma,
      trigamma, log-sum-exp, random number generation with seeds, random bases,
      range objects, data shuffling, and covariance utilities.
DEPENDS ON: `base.hpp`
USED BY: Distributions, GMM, HMM, decision trees, ANN loss functions
CORE OR AUX: **Core Primitive**
INVARIANTS: All functions are stateless (or take explicit RNG state); no global
            mutable state outside the RNG seed.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: `math::Random()` / `math::RandInt()` wrap Armadillo's RNG;
                    call `math::RandomSeed(seed)` for reproducibility.

---

FILE: `src/mlpack/core/math/make_alias.hpp` / `unwrap_alias.hpp`
ROLE: Create zero-copy Armadillo aliases (`arma::mat` backed by external
      memory) and safely unwrap them; critical for the ANN weight-sharing
      pattern where all layer parameters live in a single flat weight vector.
DEPENDS ON: `<armadillo>`
USED BY: ANN `FFN`/`RNN` weight management, layer `SetWeights()`
CORE OR AUX: **Core Primitive**
INVARIANTS: The backing memory must outlive the alias; `Unwrap()` performs a
            copy only when the input is an alias that would otherwise alias
            temporary storage.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The "single flat weight vector, aliases into layers" pattern
                    is the core memory layout of the ANN framework.  Do not
                    break this invariant.

---

## 5. Data Loading / Preprocessing

---

FILE: `src/mlpack/core/data/data.hpp` (and constituent files)
ROLE: Dataset I/O (CSV, ARFF, binary, image via STB, text via string encoding),
      preprocessing (normalisation, one-hot encoding, imputation, binarisation,
      train/test splitting), and label handling.
DEPENDS ON: `core/math/*`, `core/stb/*`, cereal, `<armadillo>`
USED BY: Every method that reads data from disk; all CLI bindings
CORE OR AUX: **Core Primitive**
INVARIANTS: `data::Load()` and `data::Save()` auto-detect file format by
            extension; column-major storage (one data point per column) is the
            universal convention.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The column-major convention (`arma::mat` is column-major by
                    default) is a hard invariant throughout the library.  Every
                    algorithm assumes points are columns.

---

FILE: `src/mlpack/core/data/dataset_mapper.hpp`
ROLE: Bidirectional mapping between categorical string labels and integer
      indices; enables transparent handling of non-numeric features.
DEPENDS ON: `<armadillo>`, `<map>`, `<string>`
USED BY: `data::Load()`, `DatasetInfo`, decision trees, Naive Bayes
CORE OR AUX: **Core Primitive**
INVARIANTS: Mapping is deterministic given the same input order; the integer
            representation is consecutive starting at 0.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/data/scaler_methods/` (min_max_scaler, standard_scaler, …)
ROLE: Stateful data normalisers that fit on training data and transform both
      train and test sets consistently.
DEPENDS ON: `<armadillo>`
USED BY: Preprocessing pipelines, `mlpack_preprocess_scale` binding
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: `Fit()` must be called before `Transform()`; inverse transform
            undoes the normalisation exactly.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/data/split_data.hpp`
ROLE: Randomly partitions a dataset into train and test subsets, with optional
      stratification.
DEPENDS ON: `core/math/random.hpp`, `<armadillo>`
USED BY: Cross-validation framework, user code
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Shuffle is reproducible given the same `math::RandomSeed()`.
REBUILD PHASE: **v1**

---

## 6. Spatial Index Structures (Trees)

---

FILE: `src/mlpack/core/tree/binary_space_tree/` (kd-tree, ball-tree, RP-tree variants)
ROLE: Generic binary space-partitioning tree templated on bound type and split
      strategy; the default spatial index for nearest-neighbour search.
DEPENDS ON: `core/math/*`, `core/distances/*`, `hrectbound.hpp`,
            `ballbound.hpp`, `statistic.hpp`
USED BY: `methods/neighbor_search`, `methods/kde`, `methods/range_search`,
         `methods/emst`, `methods/lsh`
CORE OR AUX: **Core Primitive**
INVARIANTS: Points are stored in a contiguous column-major reference set;
            tree construction reorders the dataset; child bounds are contained
            within parent bounds (spatial nesting invariant).
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The tree is not a data structure in the OOP sense; it is a
                    *policy class* – you swap bounds and split rules as template
                    parameters to get different tree flavours.

---

FILE: `src/mlpack/core/tree/cover_tree/`
ROLE: Cover tree for metric spaces where explicit coordinates may not exist;
      supports exact and approximate nearest-neighbour in general metric spaces.
DEPENDS ON: `core/distances/`, `statistic.hpp`
USED BY: `methods/neighbor_search` (cover-tree variant), `methods/fastmks`
CORE OR AUX: **Core Primitive**
INVARIANTS: Metric must satisfy the triangle inequality; cover constant
            `base > 1`.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/tree/rectangle_tree/` (R-tree, R*-tree, X-tree, …)
ROLE: Disk-friendly rectangle trees for spatial database-style queries.
DEPENDS ON: `hrectbound.hpp`, `statistic.hpp`
USED BY: Spatial join queries; less commonly used than binary space trees
CORE OR AUX: **Experimental / Optional**
INVARIANTS: Bounding rectangles at each node must contain all child rectangles;
            minimum bounding rectangle property.
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/core/tree/tree_traits.hpp`
ROLE: Compile-time trait class describing tree capabilities (dual-tree support,
      self-child queries, etc.); drives algorithm policy selection.
DEPENDS ON: nothing (empty base)
USED BY: `methods/neighbor_search`, `methods/kde`, traversal code
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: Default trait values are conservative (false); specialise to opt in.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/tree/binary_space_tree/dual_tree_traverser.hpp`
ROLE: Dual-tree recursion engine; prunes branch-and-bound exploration using
      bound estimates; the innermost loop of kNN/range-search algorithms.
DEPENDS ON: binary space tree, `traversal_info.hpp`
USED BY: `methods/neighbor_search`, `methods/range_search`, `methods/emst`
CORE OR AUX: **Performance Optimization**
INVARIANTS: The score function must return `DBL_MAX` to skip a node pair;
            recursion is depth-first; no global state.
REBUILD PHASE: **v1**
NOTES FOR FLUXIONS: This traverser, together with the tree bounds, is
                    responsible for the sub-linear scaling of kNN.  Do not
                    replace with brute-force until v2.

---

## 7. Distance Metrics

---

FILE: `src/mlpack/core/distances/lmetric.hpp`
ROLE: L^p distance (L1, L2, L∞) as a stateless policy class; the default
      distance for most spatial algorithms.
DEPENDS ON: `base.hpp`
USED BY: `methods/neighbor_search`, `methods/kmeans`, `methods/kde`, trees
CORE OR AUX: **Core Primitive**
INVARIANTS: Template parameter `Power` is the p-value; `TakeRoot = true` for
            proper distances, `false` for squared distances (faster, breaks
            triangle inequality).
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/distances/mahalanobis_distance.hpp`
ROLE: Mahalanobis distance parameterised by a covariance matrix; used in GMM
      and metric learning methods.
DEPENDS ON: `<armadillo>`
USED BY: `methods/gmm`, `methods/lmnn`, `methods/nca`
CORE OR AUX: **Core Primitive**
INVARIANTS: Covariance matrix must be positive semi-definite; distance is
            symmetric.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/distances/ip_metric.hpp`
ROLE: Inner-product metric (angle / cosine distance) for use with cover trees
      and FastMKS.
DEPENDS ON: `<armadillo>`
USED BY: `methods/fastmks`
CORE OR AUX: **Core Primitive**
INVARIANTS: Only valid for unit-norm vectors (cosine distance); raw
            inner-product variant is not a proper metric.
REBUILD PHASE: **v1**

---

## 8. Probability Distributions

---

FILE: `src/mlpack/core/distributions/gaussian_distribution.hpp`
ROLE: Full-covariance multivariate Gaussian with `Train()`, `Probability()`,
      `LogProbability()`, and `Random()`.
DEPENDS ON: `core/math/*`, `<armadillo>`
USED BY: `methods/gmm`, `methods/hmm`, `methods/ann/dists`
CORE OR AUX: **Core Primitive**
INVARIANTS: Covariance must be positive definite; Cholesky factorisation is
            cached and recomputed on `Train()`.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/distributions/discrete_distribution.hpp`
ROLE: Multinomial (categorical) distribution over a finite alphabet; used for
      HMM emission models and Naive Bayes.
DEPENDS ON: `<armadillo>`
USED BY: `methods/hmm`, `methods/naive_bayes`
CORE OR AUX: **Core Primitive**
INVARIANTS: Probabilities sum to 1; internally stored as log-probabilities to
            avoid underflow.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/core/distributions/regression_distribution.hpp`
ROLE: Gaussian distribution conditioned on a linear predictor; used as the
      emission model for regression HMMs.
DEPENDS ON: `gaussian_distribution.hpp`, `methods/linear_regression`
USED BY: `methods/hmm`
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

## 9. Kernels

---

FILE: `src/mlpack/core/kernels/kernels.hpp` (and constituent files)
ROLE: Stateless kernel functions (Gaussian, Laplacian, Epanechnikov, linear,
      polynomial, hyperbolic-tangent, cosine similarity, etc.) conforming to
      the `kernel concept` (an `Evaluate(a, b)` method).
DEPENDS ON: `<armadillo>`
USED BY: `methods/kernel_pca`, `methods/fastmks`, `methods/nystroem_method`,
         KDE, LSH
CORE OR AUX: **Core Primitive**
INVARIANTS: All kernels are symmetric; `Evaluate(a, a) > 0`.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The kernel concept is duck-typed; any class with a matching
                    `Evaluate()` works.  `kernel_traits.hpp` provides optional
                    static assertions.

---

## 10. Classical ML Algorithms

The following entries share a common pattern:
- Header-only or header + `_impl.hpp` design (full template instantiation at
  user site).
- `Train(data, labels, ...)` method for fitting.
- `Classify()` / `Predict()` method for inference.
- `Serialize()` method for model persistence via cereal.
- Corresponding `*_main.cpp` file that wires the algorithm to the binding system.

---

FILE: `src/mlpack/methods/linear_regression/`
ROLE: Ordinary least-squares regression via direct solve or gradient descent.
DEPENDS ON: `core.hpp`, optionally ensmallen
USED BY: `regression_distribution.hpp`, user code, CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/logistic_regression/`
ROLE: L2-regularised logistic regression trained by L-BFGS (via ensmallen).
DEPENDS ON: `core.hpp`, ensmallen
USED BY: CLI binding, user code
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/naive_bayes/`
ROLE: Gaussian Naive Bayes classifier; incrementally trainable.
DEPENDS ON: `core.hpp`
USED BY: CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/decision_tree/`
ROLE: C4.5-style decision tree (and random forest base) with pluggable
      split-quality metrics (Gini, information gain, MSE, MAD).
DEPENDS ON: `core.hpp`
USED BY: `methods/random_forest`, `methods/hoeffding_trees`, CLI binding
CORE OR AUX: **Core Primitive**
INVARIANTS: Categorical features must be pre-encoded as integers via
            `DatasetMapper`.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/random_forest/`
ROLE: Ensemble of decision trees trained on bootstrapped subsamples with random
      feature subsets.
DEPENDS ON: `methods/decision_tree/`, `core/math/random.hpp`
USED BY: CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/kmeans/`
ROLE: Lloyd's k-means with pluggable initialisation (random, K-Means++,
      Hartigan) and acceleration strategies (dual-tree, Hamerly, etc.).
DEPENDS ON: `core/tree/*`, `core/distances/*`, `core.hpp`
USED BY: `methods/gmm`, CLI binding
CORE OR AUX: **Core Primitive**
INVARIANTS: All points must be finite; k ≤ n_points.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/gmm/`
ROLE: Gaussian Mixture Model trained via EM; supports diagonal and full
      covariances.
DEPENDS ON: `core/distributions/*`, `methods/kmeans/`
USED BY: `methods/hmm`, CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/hmm/`
ROLE: Hidden Markov Model (Viterbi, Baum-Welch, forward/backward algorithms)
      with pluggable emission distributions.
DEPENDS ON: `core/distributions/*`, `core/math/*`
USED BY: CLI binding
CORE OR AUX: **Core Primitive**
INVARIANTS: Emission distribution type is a template parameter; all
            probabilities are in log-space internally.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/neighbor_search/`
ROLE: Exact and approximate kNN / kFN search via single-tree and dual-tree
      traversal; the library's canonical spatial query algorithm.
DEPENDS ON: `core/tree/*`, `core/distances/*`
USED BY: `methods/lmnn`, `methods/nca`, `methods/emst`, CLI binding
CORE OR AUX: **Core Primitive**
INVARIANTS: Query set and reference set share the same dimensionality; k ≤
            n_reference_points.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The `NSModel` wrapper serialises the tree + algorithm
                    choice together, enabling language-binding model round-trips.

---

FILE: `src/mlpack/methods/pca/`
ROLE: PCA via Armadillo's SVD; optional randomised SVD back-end for large
      datasets.
DEPENDS ON: `core.hpp`, optionally `methods/randomized_svd/`
USED BY: CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/dbscan/`
ROLE: Density-based clustering (DBSCAN) using range-search for neighbour
      queries.
DEPENDS ON: `methods/range_search/`
USED BY: CLI binding
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/adaboost/`
ROLE: AdaBoost.MH with pluggable weak learners (Perceptron, decision stump).
DEPENDS ON: `methods/perceptron/`, `methods/decision_tree/`
USED BY: CLI binding
CORE OR AUX: **Orchestration / Control**
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/lars/`
ROLE: LASSO / Elastic-Net regression via the LARS algorithm.
DEPENDS ON: `core.hpp`
USED BY: `methods/sparse_coding/`
CORE OR AUX: **Core Primitive**
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/sparse_coding/`
ROLE: Dictionary learning via sparse coding (LASSO sub-problem + dictionary
      update).
DEPENDS ON: `methods/lars/`
USED BY: CLI binding
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/methods/cf/`
ROLE: Collaborative filtering (matrix factorisation + neighbourhood methods)
      for recommender systems.
DEPENDS ON: `methods/amf/`, `core.hpp`
USED BY: CLI binding
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/methods/xgboost/`
ROLE: Gradient boosted trees (XGBoost-style); most recent addition to the
      methods layer.
DEPENDS ON: `methods/decision_tree/`, `core.hpp`
USED BY: CLI binding
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

## 11. Neural Network Framework

---

FILE: `src/mlpack/methods/ann/layer/layer.hpp`
ROLE: Abstract base class `Layer<MatType>` defining the contract
      (`Forward`, `Backward`, `Gradient`, `Parameters`, `SetWeights`,
      `OutputDimensions`, `Clone`); all neural network layers inherit from this.
DEPENDS ON: `core.hpp`
USED BY: Every layer implementation, `FFN`, `RNN`, `MultiLayer`
CORE OR AUX: **Core Primitive**
INVARIANTS: Layers do NOT allocate their own weight memory; weights are
            allocated by the containing network as a single flat vector and
            aliased into each layer.  Layers hold only shape metadata until
            `SetWeights()` is called.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: The "weights live outside the layer" design is the key
                    architectural invariant of the ANN framework.  It enables
                    single-allocation gradient storage and in-place ensmallen
                    optimiser updates.

---

FILE: `src/mlpack/methods/ann/layer/multi_layer.hpp`
ROLE: Composite container layer; chains sub-layers sequentially, propagating
      forward/backward passes and accumulating gradients.
DEPENDS ON: `layer.hpp`, every concrete layer type
USED BY: `FFN`, `RNN`, user-defined sub-networks
CORE OR AUX: **Core Primitive**
INVARIANTS: Sub-layers are stored as heap-allocated raw pointers; the
            `MultiLayer` owns them.
REBUILD PHASE: **v0**

---

FILE: `src/mlpack/methods/ann/ffn.hpp` / `ffn_impl.hpp`
ROLE: Feed-forward network container; orchestrates layer graph, weight
      allocation, training loop (via ensmallen), and prediction.
DEPENDS ON: `layer/multi_layer.hpp`, `init_rules/`, `loss_functions/`,
            `<ensmallen.hpp>`
USED BY: User code, `methods/reinforcement_learning` (Q-network)
CORE OR AUX: **Orchestration / Control**
INVARIANTS: Input data is column-major (one point per column); `InputDimensions`
            must be set before training if input is not a flat vector; the flat
            weight vector is managed by `FFN` and aliased into layers.
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: `FFN::Train()` passes `*this` as an `EnsembleFunction` to
                    ensmallen.  Any ensmallen-compatible optimiser (SGD, Adam,
                    L-BFGS, …) works without modification.

---

FILE: `src/mlpack/methods/ann/rnn.hpp` / `rnn_impl.hpp`
ROLE: Recurrent network container; wraps `FFN` and extends it with BPTT
      (backpropagation through time) over sequence cubes.
DEPENDS ON: `ffn.hpp`
USED BY: Sequence-to-sequence tasks; `methods/reinforcement_learning` with
         recurrent policies
CORE OR AUX: **Core Primitive**
INVARIANTS: Input is `arma::cube` (features × batch × time-steps); response
            cube has the same slice count unless `single = true`.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/ann/dag_network.hpp`
ROLE: DAG-structured network where layers can have multiple inputs and outputs
      (as opposed to the sequential `MultiLayer`).
DEPENDS ON: `layer/layer.hpp`
USED BY: Advanced architectures (ResNet-style skip connections)
CORE OR AUX: **Experimental / Optional**
INVARIANTS: The layer graph must be a DAG; topological sort is computed at
            first `Forward()` call.
REBUILD PHASE: **v2**
NOTES FOR FLUXIONS: Until DAG support is needed, use `MultiLayer` (simpler
                    invariants, lower overhead).

---

FILE: `src/mlpack/methods/ann/layer/` (convolution, LSTM, GRU, BatchNorm, etc.)
ROLE: Concrete layer implementations; each file is an isolated computation unit
      satisfying the `Layer` contract.
DEPENDS ON: `layer.hpp`, `core/math/*`, `<armadillo>`
USED BY: `FFN`, `RNN`, `MultiLayer`
CORE OR AUX: **Core Primitive**
INVARIANTS: Per-layer invariants documented in each header; general rule:
            input dimensions are validated in `OutputDimensions()`, not in
            `Forward()`.
REBUILD PHASE: **v1** (basic: Linear, ReLU, Softmax, BatchNorm) /
               **v2** (specialised: grouped convolution, attention, NoisyLinear)

---

FILE: `src/mlpack/methods/ann/loss_functions/`
ROLE: Loss function objects (MSE, NLL, cross-entropy, hinge, huber, etc.)
      serving as the `OutputLayerType` template argument to `FFN`/`RNN`.
DEPENDS ON: `layer.hpp`
USED BY: `FFN`, `RNN` as template parameters
CORE OR AUX: **Core Primitive**
INVARIANTS: Every loss implements `Forward()` (scalar loss), `Backward()`
            (gradient wrt output), and optionally a denominator for normalisation.
REBUILD PHASE: **v0** (NLL, MSE) / **v1** (rest)

---

FILE: `src/mlpack/methods/ann/init_rules/`
ROLE: Weight-initialisation strategies (random uniform, Glorot, He, orthogonal,
      etc.) called once per network at construction.
DEPENDS ON: `core/math/random.hpp`, `<armadillo>`
USED BY: `FFN`, `RNN` as template parameters
CORE OR AUX: **Developer Ergonomics**
INVARIANTS: `Initialize(weights, rows, cols)` takes the full flat weight vector
            and must fill it completely.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/methods/ann/models/`
ROLE: Pre-built model architectures (YOLOv3, etc.) composed from primitive
      layers; essentially reference implementations.
DEPENDS ON: All layer types, `FFN`
USED BY: User code, CLI bindings for those models
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

## 12. Reinforcement Learning

---

FILE: `src/mlpack/methods/reinforcement_learning/q_learning.hpp`
ROLE: DQN / double-DQN agent; orchestrates environment interaction, experience
      replay, and neural-network Q-function updates.
DEPENDS ON: `methods/ann/ffn.hpp`, `replay/`, `training_config.hpp`, ensmallen
USED BY: CLI binding, user experiments
CORE OR AUX: **Orchestration / Control**
INVARIANTS: Environment type must satisfy the environment concept (provides
            `State`, `Action`, `Step()`, `IsTerminal()`); network output
            dimension equals action-space size.
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/methods/reinforcement_learning/ddpg.hpp`
ROLE: Deep Deterministic Policy Gradient for continuous action spaces; uses
      actor + critic networks.
DEPENDS ON: `ffn.hpp`, `replay/`, `training_config.hpp`
USED BY: CLI binding
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/methods/reinforcement_learning/sac.hpp`
ROLE: Soft Actor-Critic agent; entropy-regularised continuous-control RL.
DEPENDS ON: `ffn.hpp`, `replay/`, `training_config.hpp`
USED BY: CLI binding
CORE OR AUX: **Experimental / Optional**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/methods/reinforcement_learning/replay/`
ROLE: Experience replay buffers (random, prioritised, sum-tree prioritised);
      decouples data collection from training.
DEPENDS ON: `<armadillo>`, environment concept
USED BY: `q_learning.hpp`, `ddpg.hpp`, `sac.hpp`, `td3.hpp`
CORE OR AUX: **Core Primitive** (for RL subsystem)
INVARIANTS: Buffer never grows beyond `maxSize`; uniform random sampling
            without replacement.
REBUILD PHASE: **v1** (random replay) / **v2** (prioritised)

---

FILE: `src/mlpack/methods/reinforcement_learning/environment/`
ROLE: Reference environment implementations (CartPole, Mountain Car, Pendulum,
      Acrobot, Continuous Mountain Car, DoublePole) for testing and examples.
DEPENDS ON: `core/math/*`
USED BY: Tests, CLI binding examples
CORE OR AUX: **Debug / Visualization**
INVARIANTS: Each environment is a self-contained state machine; `Step()` is
            deterministic given the same state/action.
REBUILD PHASE: **v1**

---

## 13. Cross-Validation and Hyperparameter Tuning

---

FILE: `src/mlpack/core/cv/`
ROLE: K-fold and holdout (simple) cross-validation framework; computes
      generalisation metrics for any mlpack classifier or regressor.
DEPENDS ON: `core/math/*`, `core/data/split_data.hpp`, metric concepts
USED BY: `core/hpt/`, CLI bindings that accept `--cv` arguments
CORE OR AUX: **Orchestration / Control**
INVARIANTS: The model type must implement `Train()` and `Classify()`/
            `Predict()` with standard signatures.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/core/hpt/`
ROLE: Hyperparameter optimisation framework built on top of the CV framework;
      searches a product-space of hyperparameter values using a pluggable
      strategy (currently grid search).
DEPENDS ON: `core/cv/`, `fixed.hpp`, `deduce_hp_types.hpp`
USED BY: CLI bindings, user code
CORE OR AUX: **Orchestration / Control**
INVARIANTS: Each dimension of the hyperparameter space must be enumerable.
REBUILD PHASE: **v2**

---

## 14. Multi-language Binding System

---

FILE: `src/mlpack/bindings/cli/mlpack_main.hpp`
ROLE: Defines `PARAM_*` macros and `mlpackMain()` entry point for command-line
      executables; uses CLI11 under the hood.
DEPENDS ON: `core/util/params.hpp`, CLI11 (vendored)
USED BY: Every `*_main.cpp` when `BINDING_TYPE == BINDING_TYPE_CLI`
CORE OR AUX: **Orchestration / Control**
INVARIANTS: `mlpackMain()` is the user-defined entry point; it is called after
            argument parsing has populated `IO::Parameters()`.
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/bindings/python/mlpack_main.hpp`
ROLE: Defines the Python binding entry point; marshals Armadillo ↔ NumPy array
      conversions and generates Cython/pybind11 wrappers.
DEPENDS ON: `core/util/params.hpp`, NumPy C API
USED BY: Python package build system
CORE OR AUX: **Orchestration / Control**
INVARIANTS: NumPy arrays are assumed to be C-contiguous (row-major) and are
            transposed on entry to match mlpack's column-major convention.
REBUILD PHASE: **v1**
NOTES FOR FLUXIONS: The row-major ↔ column-major transpose on the Python
                    boundary is automatic but has a copy cost for large datasets
                    unless the user pre-transposes.

---

FILE: `src/mlpack/bindings/julia/mlpack_main.hpp`
ROLE: Julia binding entry point; provides Julia ↔ Armadillo matrix passing via
      the Julia C API.
DEPENDS ON: `core/util/params.hpp`, Julia C API
USED BY: mlpack.jl package
CORE OR AUX: **Orchestration / Control**
REBUILD PHASE: **v1**

---

FILE: `src/mlpack/bindings/go/mlpack_main.hpp`
ROLE: Go binding entry point; bridges CGo ↔ Armadillo.
DEPENDS ON: `core/util/params.hpp`, CGo
USED BY: mlpack Go package
CORE OR AUX: **Orchestration / Control**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/bindings/R/mlpack_main.hpp`
ROLE: R binding entry point; uses Rcpp to bridge R matrices ↔ Armadillo.
DEPENDS ON: `core/util/params.hpp`, Rcpp
USED BY: mlpackR package
CORE OR AUX: **Orchestration / Control**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/bindings/markdown/mlpack_main.hpp`
ROLE: Documentation-only binding; generates Markdown API documentation from
      `PARAM_*` declarations without executing any algorithm.
DEPENDS ON: `core/util/params.hpp`
USED BY: Documentation build pipeline
CORE OR AUX: **Debug / Visualization**
REBUILD PHASE: **v2**

---

FILE: `src/mlpack/bindings/util/strip_type.hpp` / `camel_case.hpp`
ROLE: String utilities used by code generators (Python, Julia, Go, R) to
      convert C++ type names into idiomatic target-language names.
DEPENDS ON: `<string>`
USED BY: Binding code generators
CORE OR AUX: **Developer Ergonomics**
REBUILD PHASE: **v1**

---

## 15. Tests

---

FILE: `src/mlpack/tests/`
ROLE: Catch2-based unit and integration tests covering every algorithm,
      utility, tree structure, binding, and serialisation path.
DEPENDS ON: All of `src/mlpack/`, Catch2
USED BY: CI pipeline; not part of the installed library
CORE OR AUX: **Debug / Visualization**
INVARIANTS: Each test file is compiled with `BINDING_TYPE == BINDING_TYPE_TEST`
            so `mlpackMain()` is called directly without CLI parsing.
REBUILD PHASE: **v0** (skeleton with a few smoke tests) → **v1** / **v2** (full
               coverage)
NOTES FOR FLUXIONS: The test-binding infrastructure (`bindings/tests/`) is the
                    canonical way to test an algorithm end-to-end including
                    serialisation and the parameter system.

---

## 16. Build System

---

FILE: `CMakeLists.txt` (root)
ROLE: Top-level CMake entry point; detects dependencies (Armadillo, cereal,
      ensmallen, STB, CLI11, pybind11 / Cython, Julia, Go, Rcpp), sets
      compile flags, and delegates to subdirectory `CMakeLists.txt` files.
DEPENDS ON: CMake ≥ 3.13, all external dependencies
USED BY: Developers, CI pipeline, package managers
CORE OR AUX: **Orchestration / Control**
REBUILD PHASE: **v0**
NOTES FOR FLUXIONS: Every new method `*_main.cpp` must be registered in the
                    relevant `CMakeLists.txt` with `add_mlpack_executable()` (or
                    equivalent) to get CLI, Python, Julia, Go, and R wrappers
                    automatically generated.

---

## Rebuild Roadmap Summary

| Phase | What to Build | Why |
|-------|---------------|-----|
| **v0** | `base.hpp`, `config.hpp`, `prereqs.hpp`, `core.hpp`, arma_extend, cereal glue, `math/`, `data/` (load/save), `util/log`, `util/size_checks`, `arma_traits`, `distances/lmetric`, `distributions/gaussian+discrete`, `tree/binary_space_tree`, `neighbor_search`, `linear_regression`, `logistic_regression`, `naive_bayes`, `decision_tree`, `kmeans`, `pca`, `ann/layer/layer.hpp + Linear + ReLU + Softmax + NLL`, `ffn.hpp` | Minimal working ML library |
| **v1** | `rnn.hpp`, cover tree, rectangle tree basics, `gmm`, `hmm`, `random_forest`, `dbscan`, `adaboost`, `lars`, `kernel_pca`, remaining layers (Conv, LSTM, GRU, BatchNorm), `cv/`, CLI binding, Python binding, replay buffers, RL environments | Full classical + deep learning surface |
| **v2** | `dag_network`, HPT, `sparse_coding`, `cf`, `xgboost`, Go/R bindings, markdown binding, advanced RL (SAC, TD3, DDPG), rectangle tree variants, `low_precision` cereal, model zoo (YOLO) | Extended capabilities and language coverage |

---

## Global Architectural Invariants

1. **Column-major data**: all matrices store one data point per column
   (`n_features × n_points`).  Transposing on language-binding entry/exit is
   mandatory.

2. **Weights live outside layers**: the ANN `FFN`/`RNN` container allocates one
   flat weight vector; each `Layer` receives an alias.  Never allocate weights
   inside a layer constructor.

3. **Policy / strategy via templates, not virtual dispatch**: trees, kernels,
   distance metrics, split rules, optimisers, loss functions, and init rules
   are all template parameters.  Virtual dispatch is used only for the
   `Layer` polymorphism inside `MultiLayer`.

4. **Stateless algorithms, stateful models**: algorithms (`Train()`) do not
   store data; trained models store only parameters.

5. **Serialisation is mandatory**: every public model class must implement
   `serialize()` via cereal to support model persistence and the test binding
   round-trip.

6. **Log::Fatal is the error-reporting mechanism**: recoverable errors use
   `Log::Warn`; unrecoverable conditions use `Log::Fatal` (which terminates
   the process).  Do not throw exceptions across algorithm boundaries.

7. **ensmallen is the optimiser substrate**: any optimisable objective
   implements `Evaluate()` + `Gradient()` (or `EvaluateWithGradient()`) and is
   passed to an ensmallen optimiser.  mlpack does not implement optimisers
   itself.
