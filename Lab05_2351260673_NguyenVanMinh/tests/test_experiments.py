import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.enhancement import save_pcm16
from src.experiments import select_ss_config, run_enhancement_cases


class ExperimentTests(unittest.TestCase):
    def test_test_files_cannot_choose_configuration(self):
        with self.assertRaises(ValueError):
            select_ss_config([Path('spk01_test_01.wav')], [], [], [], [], {})

    def test_dev_selection_and_exported_benchmark(self):
        rng = np.random.default_rng(9)
        sr = 16000
        t = np.arange(2*sr)/sr
        clean = .2*np.sin(2*np.pi*440*t)*((t>.4)&(t<1.4))
        noise = rng.normal(0,.05,len(t))
        stft = dict(n_fft=512,win_length=400,hop_length=160)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            speech_path, noise_path = folder/'spk_dev_01.wav', folder/'noise.wav'
            save_pcm16(speech_path,clean,sr)
            save_pcm16(noise_path,noise,sr)
            config, summary, trials = select_ss_config(
                [speech_path],[noise_path],[0],[1.5,2.],[.01],stft)
            self.assertEqual(len(summary),2)
            self.assertEqual(len(trials),2)
            self.assertEqual(config['beta'],summary.iloc[0].beta)
            cases, metrics, rate = run_enhancement_cases(
                speech_path,[noise_path],[0,5],config,stft,folder/'output')
            self.assertEqual(rate,sr)
            self.assertEqual(len(metrics),6)
            self.assertEqual(len(list((folder/'output').glob('*.wav'))),7)
            for signals in cases.values():
                self.assertEqual(set(signals),{'Clean','Noisy','SS','Wiener'})
                for x in signals.values():
                    self.assertEqual(x.shape,clean.shape)
                    self.assertLess(np.max(abs(x)),1)
            self.assertTrue((abs(metrics.snr_in-metrics.target_db)<.2).all())


if __name__ == '__main__':
    unittest.main()
