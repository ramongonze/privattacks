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
            "income":["low", "medium", "low", "medium", "medium", "high", "high", "medium"]
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
        self.assertAlmostEqual(self.attack1.prior_ai(self.sensitive1)[self.sensitive1[0]], 1/3)
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

    def test_posterior_reid_subset(self):
        results = self.attack1.posterior_reid_subset(self.qids1, 1, len(self.qids1))

        # Age
        true_posterior = 3/8
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Age and disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

    def test_posterior_ai_subset(self):
        results = self.attack1.posterior_ai_subset(self.qids1, self.sensitive1, 1, len(self.qids1))
        true_posterior = 3/4
        
        # Age
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Disease
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Age and disease
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

    def test_posterior_reid_ai_subset(self):
        results = self.attack1.posterior_reid_ai_subset(self.qids1, self.sensitive1, 1, len(self.qids1))
        
        # Re-identificadtion
        # Age
        true_posterior = 3/8
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Age and disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Attribute inference
        true_posterior = 3/4
        # Age
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Disease
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Age and disease
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

    def test_posterior_reid_ai_subset_parallel(self):
        results = self.attack1.posterior_reid_ai_subset(self.qids1, self.sensitive1, 1, len(self.qids1), n_processes=4)
        
        # Re-identificadtion
        # Age
        true_posterior = 3/8
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # # Age and disease
        true_posterior = 1/2
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_reid"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)
        
        # Attribute inference
        true_posterior = 3/4
        # Age
        calculated_posterior = float(results[results["qids"] == "age"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Disease
        calculated_posterior = float(results[results["qids"] == "education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

        # Age and disease
        calculated_posterior = float(results[results["qids"] == "age,education"]["posterior_income"].to_list()[0])
        self.assertAlmostEqual(true_posterior, calculated_posterior)

if __name__ == '__main__':
    unittest.main()