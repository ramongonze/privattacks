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

        self.dataset2 = pd.DataFrame({
            "age":[25,25,25,25,25,49,49,49,49,60],
            "disease":["heart","asthma","asthma","asthma","heart","asthma","asthma","heart","heart","heart"]
        })
        self.data2 = Data(dataframe=self.dataset2)
        self.attack2 = Attack(self.data2)
        self.qids2 = ["age"]
        self.sensitive2 = ["disease"] 


    def test_prior_reid(self):
        self.assertAlmostEqual(self.attack1.prior_reid(), 1/8)
        self.assertAlmostEqual(self.attack2.prior_reid(), 1/10)

    def test_posterior_reid(self):
        self.assertAlmostEqual(self.attack1.posterior_reid(self.qids1), 1/2)
        self.assertAlmostEqual(self.attack2.posterior_reid(self.qids2), 3/10)

    def test_prior_ai(self):
        self.assertAlmostEqual(self.attack1.prior_ai(self.sensitive1)[self.sensitive1[0]], 1/2)
        self.assertAlmostEqual(self.attack2.prior_ai(self.sensitive2)[self.sensitive2[0]], 1/2)

    def test_posterior_ai(self):
        self.assertAlmostEqual(self.attack1.posterior_ai(self.qids1, self.sensitive1)[self.sensitive1[0]], 3/4)
        self.assertAlmostEqual(self.attack2.posterior_ai(self.qids2, self.sensitive2)[self.sensitive2[0]], 6/10)

    def test_posterior_reid_ai(self):
        posterior_reid, posteriors_ai = self.attack1.posterior_reid_ai(self.qids1, self.sensitive1)
        self.assertAlmostEqual(posterior_reid, 1/2)
        self.assertAlmostEqual(posteriors_ai[self.sensitive1[0]], 3/4)
        
        posterior_reid, posteriors_ai = self.attack2.posterior_reid_ai(self.qids2, self.sensitive2)
        self.assertAlmostEqual(posterior_reid, 3/10)
        self.assertAlmostEqual(posteriors_ai[self.sensitive2[0]], 6/10)

if __name__ == '__main__':
    unittest.main()