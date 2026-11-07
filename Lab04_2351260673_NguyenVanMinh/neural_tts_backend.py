"""
Neural TTS Backend Wrapper (SPEC Section 4.6).
Cho phép gọi trực tiếp:
    from neural_tts_backend import synthesize
    y_nn, sr, extra = synthesize("Mời khách hàng số 23 đến quầy 5")
    sf.write("audio_output/neural_tts.wav", y_nn, sr)
"""

from src.neural_tts import synthesize, plot_neural_melspec

__all__ = ["synthesize", "plot_neural_melspec"]
