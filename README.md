# ActiveInference_Agent_Test_on_TradingBot (BUT ACTUALLY LLM AS GENERATIVE MODEL IS WAY TOO UNSTRUCTURED. ACTIVE INFERENCE ENGINE ON AGENT IS JUST A USELESS DECORATION.)

Automatic Quantitative Trading AI Agent empowered by local active inference engine 
A hierarchical Bayesian trading system grounded in Karl Friston's Active Inference framework (Free Energy Principle), combining a Python port of SPM's spm_MDP_VB_X solver with LLM-driven generative model construction and live Alpaca order execution.


Table of Contents

Overview
Theoretical Background
Architecture
Repository Structure
Key Components
Three-Layer Memory System
Installation
Configuration
Usage
Backtesting
Testing
Design Decisions
References
License


Overview
This project implements a fully autonomous algorithmic trading agent using Active Inference — a first-principles neuroscientific theory of perception, learning, and decision-making. Rather than optimising a reward function (as in reinforcement learning), the agent minimises variational free energy: the gap between its generative model of the market and what it actually observes.
Key properties:

Hierarchical POMDP — two levels of partially observable Markov decision processes (POMDPs), solved jointly via variational Bayes (VB) message passing.
LLM-guided model construction — an LLM (Claude Sonnet) reads live news and macro text each cycle, then proposes the generative model parameters (likelihood matrices, prior distributions, hidden-state semantics) dynamically. The number of hidden states is variable per cycle.
Dirichlet parameter learning — concentration parameters a, b, d, e are updated online from trading outcomes, enabling the generative model to adapt to changing market regimes.
Three-layer memory — raw trade logs, persistent Dirichlet parameters on disk, and LLM-driven reflection that enriches future model construction.
Live execution — integrates with the Alpaca brokerage API for market order placement.
Telegram notifications — every trading cycle produces a structured Telegram report.


Theoretical Background
Active Inference formulates an agent as a system that maintains a generative model of its environment — a joint probability distribution over hidden states s, observations o, and policies π:
P(o, s, π) = P(o | s) · P(s | π) · P(π)
The agent never directly optimises reward. Instead it minimises the variational free energy F, which upper-bounds surprise:
F = E_Q[ln Q(s,π) − ln P(o,s,π)]
  = KL[Q(s,π) || P(s,π)] − ln P(o)
Policy selection is driven by expected free energy G, which decomposes into:

Pragmatic value (extrinsic) — expected reward / preference satisfaction
Epistemic value (intrinsic) — expected information gain about hidden states (curiosity / ambiguity resolution)

G(π) = E_Q[ln P(o | C) − ln Q(o | π)]
     = Pragmatic value + Epistemic value
This gives the agent a natural explore-exploit trade-off without heuristics.
Learning updates the Dirichlet concentration parameters {a, b, d} after each observation using free energy gradients, subject to a forgetting rule:
θ ← (θ − θ₀)(1 − ω) + θ₀ + η · δθ
where ω is the forgetting rate and η is the learning rate.
The VB solver (spm_MDP_VB_X) is a Python port of Friston et al.'s SPM MATLAB implementation, performing gradient descent on free energy over ni = 32 iterations per timestep.
For the full mathematical derivation see the bundled tutorial: Smith, Friston & Whyte (2022), Journal of Mathematical Psychology.

Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        heartbeat.py                             │
│                  (autonomous trading loop)                      │
└──────────────┬──────────────────────────────────────────────────┘
               │ each cycle
   ┌───────────▼────────────────────────────────────────────────┐
   │  Step 1: Fetch news + macro corpus   (web_search_tool)     │
   │  Step 2: LLM reflection → variable-N llm_params            │
   │  Step 3: Fetch Alpaca price bars                           │
   │  Step 4: Build L1 MDPs  (model_builder.py)                 │
   │  Step 5: Build L2 MDP   (model_builder.py)                 │
   │  Step 6: Inject price observations                         │
   │  Step 7: Restore / resize Dirichlet params (memory.py)     │
   │  Step 8: spm_MDP_VB_X → beliefs + policy  (solver.py)     │
   │  Step 9: Save updated Dirichlet params                     │
   │  Step 10: Confidence gate → Alpaca trade execution         │
   │  Step 11: Log trade + update P&L          (memory.py)      │
   │  Step 12: Telegram report                                  │
   └────────────────────────────────────────────────────────────┘

Hierarchical MDP structure:

Layer 2 — Market Regime MDP
  Hidden states : risk-on bull | risk-off bear | uncertain | trending  (4)
  Observations  : price bar (bull/bear/doji) + L1a posterior + L1b posterior
  Policies      : buy | hold | sell  (3)
        ▲                ▲
        │   hierarchical │ link
        │                │
Layer 1a — Sentiment MDP        Layer 1b — Volatility MDP
  States : bullish/bearish/…      States : high/normal/low vol/…
  Obs    : news + macro text       Obs    : same text corpus
  A, D   : ← LLM each cycle       A, D   : ← LLM each cycle
  N      : variable (2–8)         N      : variable (2–8)

Repository Structure
.
├── active_inference_engine/
│   ├── mdp_model.py          # MDPModel dataclass (A, B, C, D, a, b, d, e, …)
│   ├── solver.py             # spm_MDP_VB_X — variational Bayes VB solver
│   ├── inference.py          # Belief update equations (forward/backward pass)
│   ├── learning.py           # Dirichlet parameter updates (a, b, d, e)
│   ├── utils.py              # spm_norm, spm_softmax, spm_log, spm_kl, …
│   ├── run_trials.py         # Multi-trial runner (simulation / testing)
│   └── trading_model.py      # Trading-specific MDP builder + Alpaca + LLM calls
│
├── memory/
│   ├── memory.py             # Three-layer memory (trade log, params, LLM reflection)
│   └── model_builder.py      # Variable-N MDP construction + Dirichlet resize
│
├── heartbeat.py              # Autonomous trading loop
├── main.py                   # Upsonic agent + Telegram interface
├── backtest_layer_b.py       # Historical backtest (frozen LLM params)
├── test_active_inference.py  # Unit + integration tests for the VB engine
│
└── astepbysteptutorialonactiveinferenceandits37kzhtsx56.pdf
                              # Smith, Friston & Whyte (2022) — bundled reference

Key Components
mdp_model.py — MDPModel Dataclass
Central data structure holding all matrices for one POMDP level.
SymbolDescriptionALikelihood tensor — P(o | s), shape (No, Ns₀, …, NsNf-1)BTransition tensor — P(s' | s, u), shape (Ns, Ns, Nu)CLog-preference vector — fixed, not learnedDPrior over initial states — P(s₀)a, b, d, eLearnable Dirichlet concentration parametersa_0, b_0, d_0, e_0Frozen baselines for the forgetting ruleVPolicy matrix — shape (T-1, n_policies, n_factors)XBayesian model-average posterior beliefs (output)RPolicy posteriors (output)GExpected free energy per policy (output)FVariational free energy per policy (output)
solver.py — spm_MDP_VB_X
Python port of Friston et al.'s variational Bayes message-passing algorithm. Performs:

Forward pass — belief propagation through time under each policy
Backward pass — expected free energy computation
Policy selection — softmax over −G with precision parameter β
Action sampling — MAP action from posterior policy
Hierarchical update — passes L1 posteriors up as L2 observations

inference.py — Belief Updates
Implements the variational message-passing equations:

spm_forwards / spm_backwards — forward/backward belief propagation
spm_MDP_G — expected free energy calculation (pragmatic + epistemic)
Gradient-descent inner loop with ni iterations and time constant tau

learning.py — Dirichlet Updates
Online Bayesian learning after each observation:

Updates a (likelihood params) from δF/δa
Updates b (transition params) from state transitions
Updates d (prior params) from initial state beliefs
Updates e (policy params) from policy posteriors
Applies forgetting rule: θ ← (θ − θ₀)(1 − ω) + θ₀ + η·δθ

trading_model.py — Trading Integration

fetch_l1_observations — web search for news headlines + macro text
call_llm_for_l1 — calls the LLM API with the news corpus; receives JSON with A_sentiment, A_volatility, D_sentiment, D_volatility, n_sent, n_vol, and reasoning
bars_to_regime — discretises OHLCV bars into {bull, bear, doji} observations
execute_signal — places market orders via Alpaca if confidence ≥ MIN_CONFIDENCE
run_trading_step — complete one-step pipeline from data fetch to signal

model_builder.py — Variable-N MDP Construction
Builds L1 and L2 MDP objects from LLM-proposed parameters each cycle, with dynamic hidden-state counts N ∈ [2, 8]. Handles:

Normalising LLM-provided A matrices (columns must sum to 1)
Resizing Dirichlet params when N changes between cycles (new states get weak uniform priors)
Assembling the hierarchical link between L1 posteriors and L2 observation modalities


Three-Layer Memory System
Layer 1 — Trade Log          (trade_log.jsonl)
  Raw append-only record of every signal, execution, and P&L update.
  Fields: symbol, action, confidence, regime_belief, sent_belief,
          vol_belief, llm_reasoning, executed, order_id, entry_price.

Layer 2 — Parameter Store    (dirichlet_params.pkl)
  Persistent Dirichlet concentration parameters {a, b, d, e} saved
  to disk after every cycle. Loaded and resized (if N changed) at
  the start of the next cycle. This is the agent's long-term memory
  of market dynamics — it does NOT reset between sessions.

Layer 3 — LLM Reflection     (in-memory, called each cycle)
  The last REFLECTION_WINDOW trades are fed back to the LLM as
  context. The LLM reasons backwards: "given these outcomes, what
  hidden conditions must have been present?" and adjusts A matrix
  and D prior proposals accordingly. This enriches the generative
  model beyond what hand-defined discrete states can capture.

Installation
Requirements: Python 3.11+
bash# Clone
git clone https://github.com/your-username/active-inference-trading.git
cd active-inference-trading

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Core dependencies:
PackagePurposenumpyAll tensor / matrix operationsanthropicLLM API (Claude Sonnet)alpaca-trade-apiBrokerage — market data + order executionupsonicAutonomous agent framework + Telegram interfacepython-dotenvEnvironment variable managementpandas, matplotlibBacktesting outputrequestsTelegram bot API

Configuration
Copy .env.example to .env and fill in your credentials:
env# Anthropic (LLM)
ANTHROPIC_API_KEY=sk-ant-...

# Alpaca (broker)
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # paper trading

# Telegram (optional notifications)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_WEBHOOK_URL=...

# Trading parameters (all optional — defaults in config.py)
SYMBOL=SPY
QTY=1
MIN_CONFIDENCE=0.45
HEARTBEAT_INTERVAL=1800        # seconds between cycles (default 30 min)
MARKET_HOURS_ONLY=true
REFLECTION_WINDOW=10           # trades fed back to LLM for reflection
Key config.py parameters:
ParameterDefaultDescriptionT6Timesteps per MDP episodeN_REGIME4L2 hidden states (regime factor)NI32VB gradient-descent iterationsTAU4.0VB time constantALPHA512.0Action precisionBETA1.0Policy prior precisionETA1.0Learning rateOMEGA1.0Forgetting rate (1.0 = no forgetting)MIN_STATES2Minimum hidden states per L1 factorMAX_STATES8Maximum hidden states per L1 factorCONCENTRATION2.0Initial Dirichlet concentrationD_FREEZE_SCALE16.0Scale for frozen time/position priors

Usage
1. Dry Run (no orders placed)
bashpython -m active_inference_engine.trading_model
Runs one complete inference cycle on live data and prints the trading signal without placing any orders.
2. Autonomous Heartbeat
bash# Terminal 1 — autonomous trading loop
python heartbeat.py

# Terminal 2 — Telegram chat interface (optional)
python main.py
The heartbeat loop runs every HEARTBEAT_INTERVAL seconds during market hours. It logs all trades to memory/trading/trade_log.jsonl and saves updated Dirichlet params to memory/trading/dirichlet_params.pkl.
3. Signal Only (no execution)
pythonfrom active_inference_engine.trading_model import run_trading_step

step = run_trading_step("AAPL")
print(step["action"])        # 'buy' | 'hold' | 'sell'
print(step["confidence"])    # float in [0, 1]
print(step["regime_belief"]) # np.ndarray, shape (4,)

Backtesting
The Layer B backtester replays historical Alpaca bars with a single frozen LLM parameter set (one API call), enabling fast evaluation of the VB engine on real price data.
bash# Default: SPY, 2024-06-01 to 2024-12-31
python backtest_layer_b.py

# Custom symbol and date range
python backtest_layer_b.py --symbol AAPL --start 2024-01-01 --end 2024-12-31

# Fast mode — skip LLM call, use uniform priors
python backtest_layer_b.py --fast

# Set confidence threshold for high-confidence trade analysis
python backtest_layer_b.py --conf 0.55
Output:

Console summary — P&L, win rate, Sharpe ratio, max drawdown
backtest_results.csv — one row per signal window
backtest_report.png — equity curve + signal breakdown charts


Testing
bash# Run full test suite
python -m pytest test_active_inference.py -v

# Run a specific test class
python -m pytest test_active_inference.py::TestSolver -v
The test suite covers:

mdp_model.py — MDPModel construction and baseline freezing
solver.py — spm_MDP_VB_X correctness on toy POMDPs
inference.py — belief update equations
learning.py — Dirichlet update rules and forgetting
utils.py — spm_norm, spm_softmax, spm_log, spm_kl
run_trials.py — multi-trial simulation
Integration tests — full hierarchical solve with known-good fixtures


Design Decisions
Why Active Inference instead of RL?
Active inference provides a unified treatment of perception, learning, and action under one objective (free energy minimisation). The epistemic value term naturally handles uncertainty — the agent seeks information when uncertain, rather than requiring an explicit exploration bonus.
Why variable-N hidden states?
Fixed discrete states (e.g., bull/bear/neutral) are insufficient to represent the true complexity of market regimes. By delegating state-space design to the LLM each cycle, the model can expand its representational capacity when it detects systematic prediction errors — analogous to adding new hidden units in a neural network.
Why LLM for generative model parameters rather than end-to-end?
The VB solver operates on the structured probabilistic model (A, B, C, D matrices). Using the LLM to fill in these matrices from natural language observations (news, macro commentary) decouples interpretable probabilistic inference from high-dimensional language understanding. The result is a traceable, mathematically grounded decision pipeline.
Why Dirichlet parameters on disk?
Dirichlet parameters {a, b, d} are the agent's accumulated beliefs about market dynamics — analogous to long-term memory. Persisting them between sessions means the agent continues learning across days and sessions without restarting from uninformative priors.
Paper-trading default
ALPACA_BASE_URL defaults to the paper-trading endpoint. Switch to https://api.alpaca.markets only after validating the strategy thoroughly in paper mode.

References

Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632. https://doi.org/10.1016/j.jmp.2021.102632 (bundled as PDF)

Friston, K. J., et al. (2017). Active inference and learning. Neuroscience & Biobehavioral Reviews.

Friston, K. J., et al. (2017). Active inference: a process theory. Neural Computation.

Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. Biological Cybernetics.

SPM (Statistical Parametric Mapping) MATLAB toolbox — spm_MDP_VB_X.m by Karl Friston, UCL. https://www.fil.ion.ucl.ac.uk/spm/
