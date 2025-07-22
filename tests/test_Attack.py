from privattacks.data import Data
from privattacks.attacks import Attack
import pandas as pd
import numpy as np
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
        self.assertAlmostEqual(self.attack1.prior_vulnerability("reid"), 1/8)
        self.assertAlmostEqual(self.attack2.prior_vulnerability("reid"), 1/10)

    def test_posterior_reid(self):
        self.assertAlmostEqual(self.attack1.posterior_vulnerability("reid", self.qids1), 1/2)
        self.assertAlmostEqual(self.attack2.posterior_vulnerability("reid", self.qids2), 3/10)

    def test_prior_ai(self):
        self.assertAlmostEqual(self.attack1.prior_vulnerability("ai", self.sensitive1)[self.sensitive1[0]], 1/3)
        self.assertAlmostEqual(self.attack2.prior_vulnerability("ai", self.sensitive2)[self.sensitive2[0]], 1/2)

    def test_posterior_ai(self):
        self.assertAlmostEqual(self.attack1.posterior_vulnerability(
            "ai", self.qids1, self.sensitive1)[self.sensitive1[0]], 3/4
        )
        self.assertAlmostEqual(self.attack2.posterior_vulnerability(
            "ai", self.qids2, self.sensitive2)[self.sensitive2[0]], 6/10
        )

    def test_posterior_reid_ai(self):
        posteriors = self.attack1.posterior_vulnerability("all", self.qids1, self.sensitive1)
        posterior_reid, posteriors_ai = posteriors["reid"], posteriors["ai"]
        self.assertAlmostEqual(posterior_reid, 1/2)
        self.assertAlmostEqual(posteriors_ai[self.sensitive1[0]], 3/4)
        
        posteriors = self.attack2.posterior_vulnerability("all", self.qids2, self.sensitive2)
        posterior_reid, posteriors_ai = posteriors["reid"], posteriors["ai"]
        self.assertAlmostEqual(posterior_reid, 3/10)
        self.assertAlmostEqual(posteriors_ai[self.sensitive2[0]], 6/10)

    def test_posterior_reid_record(self):
        _, distribution = self.attack1.posterior_vulnerability("reid", self.qids1, distribution=True)
        distribution = sorted(distribution)
        distribution_gt = sorted([1, 1/2, 1/2, 1/2, 1/2, 1/3, 1/3, 1/3])
        for i in np.arange(len(distribution)):
            self.assertAlmostEqual(distribution[i], distribution_gt[i])
        
        # Test average
        self.assertAlmostEqual(np.array(distribution).mean(), 1/2)

        _, distribution = self.attack2.posterior_vulnerability("reid", self.qids2, distribution=True)
        distribution = sorted(distribution)
        distribution_gt = sorted([1/5]*5 + [1/4]*4 + [1])
        for i in np.arange(len(distribution)):
            self.assertAlmostEqual(distribution[i], distribution_gt[i])

        # Test average
        self.assertAlmostEqual(np.array(distribution).mean(), 3/10)

    def test_posterior_ai_record(self):
        _, distribution = self.attack1.posterior_vulnerability("ai", self.qids1, self.sensitive1, distribution=True)
        distribution = sorted(distribution[self.sensitive1[0]])
        distribution_gt = sorted([1, 1/2, 1/2, 1, 1, 2/3, 2/3, 2/3])
        for i in np.arange(len(distribution)):
            self.assertAlmostEqual(distribution[i], distribution_gt[i])
        
        # Test average
        self.assertAlmostEqual(np.array(distribution).mean(), 3/4)

        _, distribution = self.attack2.posterior_vulnerability("ai", self.qids2, self.sensitive2, distribution=True)
        distribution = sorted(distribution[self.sensitive2[0]])
        distribution_gt = sorted([3/5]*5 + [2/4]*4 + [1])
        for i in np.arange(len(distribution)):
            self.assertAlmostEqual(distribution[i], distribution_gt[i])

        # Test average
        self.assertAlmostEqual(np.array(distribution).mean(), 6/10)

    def test_posterior_reid_ai_record(self):
        posteriors = self.attack1.posterior_vulnerability("all", self.qids1, self.sensitive1, distribution=True)
        (_, distribution_reid), (_,distribution_ai) = posteriors["reid"], posteriors["ai"]

        distribution_reid = sorted(distribution_reid)
        distribution_reid_gt = sorted([1, 1/2, 1/2, 1/2, 1/2, 1/3, 1/3, 1/3])
        distribution_ai = sorted(distribution_ai[self.sensitive1[0]])
        distribution_ai_gt = sorted([1, 1/2, 1/2, 1, 1, 2/3, 2/3, 2/3])
        for i in np.arange(len(distribution_reid)):
            self.assertAlmostEqual(distribution_reid[i], distribution_reid_gt[i])

        for i in np.arange(len(distribution_ai)):
            self.assertAlmostEqual(distribution_ai[i], distribution_ai_gt[i])
        
        # Test average
        self.assertAlmostEqual(np.array(distribution_reid).mean(), 1/2)
        self.assertAlmostEqual(np.array(distribution_ai).mean(), 3/4)

        posteriors = self.attack2.posterior_vulnerability("all", self.qids2, self.sensitive2, distribution=True)
        (_, distribution_reid), (_,distribution_ai) = posteriors["reid"], posteriors["ai"]
        distribution_reid = sorted(distribution_reid)
        distribution_reid_gt = sorted([1/5]*5 + [1/4]*4 + [1])
        distribution_ai = sorted(distribution_ai[self.sensitive2[0]])
        distribution_ai_gt = sorted([3/5]*5 + [2/4]*4 + [1])
        for i in np.arange(len(distribution_reid)):
            self.assertAlmostEqual(distribution_reid[i], distribution_reid_gt[i])

        for i in np.arange(len(distribution_ai)):
            self.assertAlmostEqual(distribution_ai[i], distribution_ai_gt[i])

        # Test average
        self.assertAlmostEqual(np.array(distribution_reid).mean(), 3/10)
        self.assertAlmostEqual(np.array(distribution_ai).mean(), 6/10)

    def test_posterior_reid_subset(self):
        combinations = list(range(1, len(self.qids1)+1))
        results = self.attack1.posterior_vulnerability("reid", self.qids1, combinations=combinations)

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
        combinations = list(range(1, len(self.qids1)+1))
        results = self.attack1.posterior_vulnerability("ai", self.qids1, self.sensitive1, combinations=combinations)
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
        combinations = list(range(1, len(self.qids1)+1))
        results = self.attack1.posterior_vulnerability("all", self.qids1, self.sensitive1, combinations=combinations)
        
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
        combinations = list(range(1, len(self.qids1)+1))
        results = self.attack1.posterior_vulnerability(
            "all", self.qids1, self.sensitive1, combinations=combinations, n_processes=4
        )
        
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