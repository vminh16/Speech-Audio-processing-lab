import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.enhancement import save_pcm16
from src.speaker import speaker_embedding
from src.integration import run_integrated_enhancement


class IntegrationTests(unittest.TestCase):
    def test_shared_inputs_frozen_models_and_complete_claim_trials(self):
        sr=16000
        rng=np.random.default_rng(55)
        t=np.arange(sr)/sr
        a=.1*np.sin(2*np.pi*230*t)+rng.normal(0,.002,sr)
        b=.1*np.sin(2*np.pi*560*t)+rng.normal(0,.002,sr)
        options=dict(n_mfcc=20,use_delta=True,cmn=False)
        models={'a':speaker_embedding(a,**options),'b':speaker_embedding(b,**options)}
        before={k:v.copy() for k,v in models.items()}
        speaker=dict(models=models,summary=dict(features=options,threshold=.99))
        dataset=[dict(file='a_test.wav',split='test',speaker='a',x=a,sr=sr),
                 dict(file='b_test.wav',split='test',speaker='b',x=b,sr=sr)]
        conversation=dict(audio=np.r_[a,b],sr=sr,predictions=pd.DataFrame([
            dict(turn=1,start=0.,end=1.,true='a'),dict(turn=2,start=1.,end=2.,true='b')]),
            annotations=[],intervals=[])
        with tempfile.TemporaryDirectory() as folder:
            noise=Path(folder)/'noise.wav'
            save_pcm16(noise,rng.normal(0,.03,sr),sr)
            result=run_integrated_enhancement(dataset,conversation,speaker,
                dict(beta=2.,floor=.01),dict(n_fft=512,win_length=400,hop_length=160),
                [noise],[5],Path(folder)/'results')
            table=result['predictions']
            self.assertEqual(set(table.method),{'Raw','SS','Wiener'})
            self.assertEqual(set(table.scope),{'test','conversation'})
            self.assertEqual(len(result['trials']),len(table)*2)
            self.assertTrue((abs(table[table.noise=='noise'].snr_in-5)<.2).all())
            for k,v in before.items():np.testing.assert_array_equal(v,models[k])
            for key,signals in result['conversation_cases'].items():
                self.assertEqual(signals['Raw'].shape,(2*sr,))
                self.assertEqual(signals['SS'].shape,signals['Raw'].shape)

if __name__=='__main__':unittest.main()
