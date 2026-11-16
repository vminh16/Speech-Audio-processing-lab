import inspect
import unittest
import numpy as np
from src.separation import mix_sources, separate_ica, match_sources


class SeparationTests(unittest.TestCase):
    def test_blind_ica_recovers_non_gaussian_sources(self):
        rng=np.random.default_rng(9)
        sources=np.column_stack([rng.laplace(size=12000),rng.uniform(-1,1,12000)])
        mixtures=mix_sources(sources,[[1,.6],[.4,1]])
        estimated,info=separate_ica(mixtures,return_diagnostics=True)
        aligned,before,after=match_sources(sources,estimated)
        self.assertTrue(info['converged'])
        self.assertEqual(aligned.shape,sources.shape)
        self.assertGreater(np.diag(after).min(),.99)
        self.assertNotIn('matrix',inspect.signature(separate_ica).parameters)
        self.assertNotIn('references',inspect.signature(separate_ica).parameters)

    def test_permutation_and_sign_are_evaluation_only(self):
        rng=np.random.default_rng(3)
        sources=rng.normal(size=(1000,2))
        estimates=np.column_stack([-sources[:,1]*3,sources[:,0]*2])
        aligned,before,after=match_sources(sources,estimates)
        np.testing.assert_allclose(np.diag(after),1,atol=1e-12)
        self.assertGreater(np.corrcoef(sources[:,0],aligned[:,0])[0,1],.999)
        self.assertGreater(before[0,1],.999)

    def test_copied_mono_is_not_two_microphones(self):
        mono=np.random.default_rng(1).normal(size=1000)
        with self.assertRaises(ValueError):
            separate_ica(np.column_stack([mono,mono]))
        with self.assertRaises(ValueError):
            mix_sources(np.column_stack([mono,mono]),[[1,1],[1,1]])

if __name__=='__main__':unittest.main()
