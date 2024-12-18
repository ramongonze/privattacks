from privattacks.data import Data
from privattacks.attacks import Attack
import pandas as pd
import unittest
import os

class TestAttack(unittest.TestCase):
    def setUp(self):
        self.dataset1 = pd.DataFrame({
            "age":[20,30,30,30,30,55,55,55],
            "education":["Master", "High School", "High School", "PhD", "PhD", "Bachelor", "Bachelor", "Bachelor"],
            "income":["<10K", "[10K,50K]", "<10K", "[10K,50K]", "[10K,50K]", ">50K", ">50K", "[10K,50K]"]
        })
        self.data1 = Data(dataframe=self.dataset1)
        self.attack1 = Attack(self.data1)
        self.qids1 = ["age", "education"]
        self.sensitive1 = ["income"] 


    def test_prior_reid(self):
        self.assertAlmostEqual(self.attack1.prior_reid(), 1/8)

    def test_posterior_reid(self):
        self.assertAlmostEqual(self.attack1.posterior_reid(self.qids1), 1/2)

    def test_prior_ai(self):
        self.assertAlmostEqual(self.attack1.prior_ai(self.sensitive1)[self.sensitive1[0]], 1/2)

    def test_posterior_ai(self):
        self.assertAlmostEqual(self.attack1.posterior_ai(self.qids1, self.sensitive1)[self.sensitive1[0]], 3/4)

    def test_posterior_reid_ai(self):
        posterior_reid, posteriors_ai = self.attack1.posterior_reid_ai(self.qids1, self.sensitive1)
        self.assertAlmostEqual(posterior_reid, 1/2)
        self.assertAlmostEqual(posteriors_ai[self.sensitive1[0]], 3/4)

if __name__ == '__main__':
    unittest.main()