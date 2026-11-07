"""
HMM TTS Backend Wrapper (SPEC Section 4.5).
Cho phép gọi trực tiếp:
    from hmm_tts_backend import synthesize_hmm
    y_hmm, sr, tracks = synthesize_hmm("Mời khách hàng số 23 đến quầy 5")
"""

from src.hmm_tts import synthesize_hmm, plot_hmm_trajectories, generate_hmm_trajectories

__all__ = ["synthesize_hmm", "plot_hmm_trajectories", "generate_hmm_trajectories"]
