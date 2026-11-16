import unittest
import numpy as np
from src.speaker import (speaker_embedding, enroll_speakers, cosine_scores,
                         choose_threshold, verification_metrics)


class SpeakerTests(unittest.TestCase):
    def test_embedding_dimensions_and_cmn(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, .1, 16000)
        for delta, size in [(False,40),(True,80)]:
            e = speaker_embedding(x,use_delta=delta)
            self.assertEqual(e.shape,(size,))
            self.assertAlmostEqual(np.linalg.norm(e),1)
        normalized = speaker_embedding(x,cmn=True)
        np.testing.assert_allclose(normalized[:20],0,atol=1e-12)
        with self.assertRaises(ValueError):
            speaker_embedding(np.zeros(16000))

    def test_cosine_models_use_enrollment_vectors(self):
        vectors = np.array([[1.,0.],[1.,.1],[0.,1.],[.1,1.]])
        models = enroll_speakers(vectors,['a','a','b','b'])
        scores, speakers = cosine_scores(np.eye(2),models)
        self.assertEqual([speakers[i] for i in scores.argmax(axis=1)],['a','b'])
        for model in models.values():
            self.assertAlmostEqual(np.linalg.norm(model),1)

    def test_threshold_frozen_then_evaluated_on_new_trials(self):
        threshold, curve = choose_threshold(np.array([.8,.9]),np.array([.1,.2]))
        self.assertGreater(threshold,.2)
        self.assertLessEqual(threshold,.8)
        result = verification_metrics(np.array([.3,.9]),np.array([.1,.95]),threshold)
        self.assertEqual(result['far'],.5)
        self.assertEqual(result['frr'],.5)
        self.assertEqual(result['false_accepts'],1)
        self.assertEqual(result['false_rejects'],1)
        with self.assertRaises(ValueError):
            choose_threshold(np.array([]),np.array([.1]))



class ConversationTests(unittest.TestCase):
    def test_turns_exclude_overlap_and_merge_adjacent_same_speaker(self):
        from src.speaker_experiments import conversation_intervals
        annotations = [dict(start=0.,end=2.,speaker='a'),dict(start=2.,end=4.,speaker='a'),
                       dict(start=1.,end=3.,speaker='b'),dict(start=4.,end=5.,speaker='c')]
        intervals = conversation_intervals(annotations,6.)
        self.assertEqual([(r['start'],r['end'],r['speaker']) for r in intervals],
                         [(0.,1.,'a'),(1.,3.,'overlap'),(3.,4.,'a'),(4.,5.,'c'),(5.,6.,'silence')])


if __name__=='__main__':
    unittest.main()
