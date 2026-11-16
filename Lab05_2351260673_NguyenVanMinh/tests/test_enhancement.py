import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.enhancement import add_noise_at_snr, snr_db, save_pcm16, load_audio, power


class NoiseTests(unittest.TestCase):
    def test_targets_with_short_noise(self):
        rng = np.random.default_rng(42)
        speech = rng.normal(0, .15, 16000)
        noise = rng.normal(size=3000)
        for target in [0, 5, 10]:
            noisy, scaled_noise, alpha = add_noise_at_snr(speech, noise, target)
            self.assertAlmostEqual(snr_db(speech, noisy), target, places=8)
            np.testing.assert_allclose(noisy - speech, scaled_noise, atol=1e-15)
            self.assertGreater(alpha, 0)

    def test_shared_scaling_survives_wav_export(self):
        rng = np.random.default_rng(7)
        speech = rng.normal(size=16000)
        noisy, _, _ = add_noise_at_snr(speech, rng.normal(size=16000), 5)
        gain = .95 / max(np.max(np.abs(speech)), np.max(np.abs(noisy)))
        with tempfile.TemporaryDirectory() as folder:
            clean_path, noisy_path = Path(folder)/'clean.wav', Path(folder)/'noisy.wav'
            save_pcm16(clean_path, speech * gain, 16000)
            save_pcm16(noisy_path, noisy * gain, 16000)
            clean, sr = load_audio(clean_path)
            observed, _ = load_audio(noisy_path)
            self.assertEqual(sr, 16000)
            self.assertLess(abs(snr_db(clean, observed) - 5), .2)

    def test_invalid_input_is_rejected(self):
        for speech, noise in [(np.zeros(10), np.ones(10)), (np.ones(10), np.zeros(10)),
                              (np.ones(10), np.array([])), (np.array([np.nan]), np.ones(10))]:
            with self.assertRaises(ValueError):
                add_noise_at_snr(speech, noise, 5)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                save_pcm16(Path(folder)/'clipped.wav', np.array([1.1]), 16000)



class EnhancementTests(unittest.TestCase):
    def test_zero_and_short_signals_keep_length(self):
        from src.enhancement import spectral_subtraction, wiener_filter
        x = np.zeros(16317)
        for method in [spectral_subtraction, wiener_filter]:
            for length in [123, len(x)]:
                output = method(x[:length])
                self.assertEqual(output.shape, x[:length].shape)
                np.testing.assert_allclose(output, x[:length], atol=1e-9)

    def test_stationary_noise_reduction(self):
        from src.enhancement import spectral_subtraction, wiener_filter
        rng = np.random.default_rng(17)
        t = np.arange(32000) / 16000
        # Blind noise estimation needs speech variation; a constant tone plus
        # noise cannot be identified as speech without an external prior.
        envelope = ((t > .35) & (t < .9)) | ((t > 1.2) & (t < 1.7))
        clean = .2 * np.sin(2 * np.pi * 440 * t) * envelope
        noise = rng.normal(size=len(t))
        noisy, added, _ = add_noise_at_snr(clean, noise, 0)
        for method in [spectral_subtraction, wiener_filter]:
            output = method(noisy)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(snr_db(clean, output), snr_db(clean, noisy) + 3)

    def test_independent_api_and_gain_attribution(self):
        import inspect
        from src.enhancement import spectral_subtraction, wiener_filter
        x = np.random.default_rng(8).normal(0, .1, 18000)
        for method in [spectral_subtraction, wiener_filter]:
            self.assertNotIn('noise', inspect.signature(method).parameters)
            output, info = method(x, return_diagnostics=True)
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(np.isfinite(info['noise_psd']).all())
            self.assertTrue(((info['gain'] >= 0) & (info['gain'] <= 1)).all())
            self.assertLess(power(output), power(x)*.5)
            np.testing.assert_allclose(method(x*.3), output*.3, atol=1e-10)

    def test_invalid_spectral_parameters(self):
        from src.enhancement import spectral_subtraction, wiener_filter
        for method in [spectral_subtraction, wiener_filter]:
            with self.assertRaises(ValueError):
                method(np.ones(1000), hop_length=401)
        for params in [dict(beta=-1), dict(floor=-.1), dict(beta=np.nan)]:
            with self.assertRaises(ValueError):
                spectral_subtraction(np.ones(1000), **params)


if __name__ == '__main__':
    unittest.main()
