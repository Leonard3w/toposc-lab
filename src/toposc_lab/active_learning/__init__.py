"""Exact-label active learning. Predictions are proposals, never physics records."""

from toposc_lab.active_learning.acquisition import Acquisition, AcquisitionConfig, acquire
from toposc_lab.active_learning.campaign import CampaignConfig, CampaignResult, run_campaign
from toposc_lab.active_learning.cycle import CycleConfig, CycleResult, run_cycle
from toposc_lab.active_learning.exact import Verification, verify_exact
from toposc_lab.active_learning.pool import Candidate, CandidateSpace, generate_candidate_pool
from toposc_lab.active_learning.prediction import Prediction, predict_pool
from toposc_lab.active_learning.storage import append_verified
from toposc_lab.active_learning.training import FittedSurrogate, TrainingConfig, retrain

__all__ = [
    "Acquisition",
    "AcquisitionConfig",
    "CampaignConfig",
    "CampaignResult",
    "Candidate",
    "CandidateSpace",
    "CycleConfig",
    "CycleResult",
    "FittedSurrogate",
    "Prediction",
    "TrainingConfig",
    "Verification",
    "acquire",
    "append_verified",
    "generate_candidate_pool",
    "predict_pool",
    "retrain",
    "run_campaign",
    "run_cycle",
    "verify_exact",
]
