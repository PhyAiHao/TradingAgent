"""
config.py  —  All adjustable parameters for the active inference model
======================================================================
Edit values here to change model behaviour without touching any other file.

All values match the MATLAB Step_by_Step_Hierarchical_Model.m defaults
unless noted otherwise.
"""

# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

# Number of variational Bayes gradient-descent steps per timestep.
# MATLAB spm_MDP_VB_X_tutorial default: Ni = 32
NI = 32

# Gradient-descent time constant.
# MATLAB default (spm_MDP_VB_X_tutorial line 227): tau = 4
TAU = 4.0

# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

# Action precision — higher = more deterministic action selection.
# MATLAB default: alpha = 512
ALPHA = 512.0

# Policy precision (inverse temperature over policies).
# MATLAB default: beta = 1
BETA = 1.0

# ---------------------------------------------------------------------------
# Belief reset (ERP simulation)
# ---------------------------------------------------------------------------

# ERP reset parameter.
# MATLAB Step_by_Step line 67 and 218: mdp.erp = 1 (explicitly set)
# erp=1 means no reset; beliefs carry over fully between timesteps.
# This is appropriate for rapid tone succession.
ERP = 1.0

# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------

# Learning rate.
# MATLAB spm_MDP_VB_X_tutorial default (line 225): eta = 1
ETA = 1.0

# Forgetting rate.
# MATLAB spm_MDP_VB_X_tutorial default (line 226): omega = 1 (no forgetting)
OMEGA = 1.0

# ---------------------------------------------------------------------------
# Generative model — likelihood precision
# ---------------------------------------------------------------------------

# Precision for softmax-smoothing of likelihood matrices A.
# MATLAB Step_by_Step line 43: pr1 = 2
PR1 = 2.0   # Level-1 (tone perception)

# MATLAB Step_by_Step line 147: pr2 = 2
PR2 = 2.0   # Level-2 (sequence perception)

# Concentration scale: multiply softmax output to set initial magnitude.
# MATLAB Step_by_Step lines 46, 150-151: a = a*100
CONCENTRATION_SCALE = 100.0

# ---------------------------------------------------------------------------
# Task structure
# ---------------------------------------------------------------------------

# Number of timesteps per trial.
# MATLAB Step_by_Step line 181: T = 6
T = 6

# Number of policies.
# MATLAB Step_by_Step line 183: Pi = 2
N_POLICIES = 2

# Number of trials per experimental condition.
# MATLAB Step_by_Step line 252: N = 10
N_TRIALS = 10

# ---------------------------------------------------------------------------
# Preferences (C matrix)
# ---------------------------------------------------------------------------

# MATLAB Step_by_Step lines 198-200:
#   C2{2}(2,6) = -1  (prefer not to be incorrect at last timestep)
#   C2{2}(3,6) =  1  (prefer to be correct at last timestep)
PREF_CORRECT   =  1.0
PREF_INCORRECT = -1.0

# ---------------------------------------------------------------------------
# Learning freeze
# ---------------------------------------------------------------------------

# Multiply d priors by this factor to prevent learning in non-informative
# factors (time-in-trial, report state).
# MATLAB Step_by_Step lines 89-90: d2{2}*100, d2{3}*100
D_FREEZE_SCALE = 100.0

# ---------------------------------------------------------------------------
# ERP plotting windows
# ---------------------------------------------------------------------------

# Smoothing window (uniform filter). Set to 1 for no smoothing.
# MATLAB uses no explicit smoothing; set small for cleaner plots.
ERP_SMOOTH_SIZE = 5

# L2 window: MATLAB Step_by_Step line 312: index = (96:140)
# In Python 0-based: 95:140
L2_WIN_LO = 95    # Python slice start
L2_WIN_HI = 140   # Python slice end

# L1 window: MATLAB Step_by_Step line 320: index = (70:120)
# In Python 0-based: 69:120
L1_WIN_LO = 69    # Python slice start
L1_WIN_HI = 120   # Python slice end
