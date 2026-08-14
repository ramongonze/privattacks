from privattacks.data import Data
from privattacks.attacks import NoisyAttack
import pandas as pd
import numpy as np
import unittest

class TestNoisyAttack(unittest.TestCase):
    def setUp(self):
        self.true_df = pd.DataFrame({
            "age":[20,30,30,30,30,55,55,55],
            "education":["Master", "High School", "High School", "PhD", "PhD", "Bachelor", "Bachelor", "Bachelor"],
            "income":["low", "medium", "low", "medium", "medium", "high", "high", "medium"]
        })
        self.data_true = Data(dataframe=self.true_df)

        # Noised report shares the same domain (and codes) as "income" so that raw-code
        # comparisons in tests/fake strategies are meaningful.
        self.noised_df = pd.DataFrame({
            "income_report": ["low", "medium", "medium", "medium", "medium", "medium", "medium", "high"]
        })
        self.data_noised = Data(
            dataframe=self.noised_df,
            domains={"income_report": self.data_true.domains["income"]}
        )

        self.qids = ["age", "education"]
        self.sensitive = {"income": ["income_report"]}

    # --- Constructor validation ---

    def test_invalid_mechanism_raises(self):
        with self.assertRaises(ValueError):
            NoisyAttack(self.data_true, self.data_noised, mechanism="not_a_mechanism")

    def test_mechanism_case_insensitive(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="GRR")
        self.assertEqual(attack.mechanism, "grr")

    def test_mismatched_row_counts_raises(self):
        data_noised_wrong = Data(dataframe=pd.DataFrame({"income_report": ["low"]*5}))
        with self.assertRaises(ValueError):
            NoisyAttack(self.data_true, data_noised_wrong, mechanism="grr")

    # --- _check_cols validation ---

    def test_check_cols_invalid_qid_raises(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(ValueError):
            attack.posterior_vulnerability(["not_a_qid"], self.sensitive)

    def test_check_cols_invalid_sensitive_key_raises(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(ValueError):
            attack.posterior_vulnerability(self.qids, {"not_a_col": ["income_report"]})

    def test_check_cols_invalid_noised_col_raises(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(ValueError):
            attack.posterior_vulnerability(self.qids, {"income": ["no_such_col"]})

    def test_check_cols_string_instead_of_list_raises(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(TypeError):
            attack.posterior_vulnerability(self.qids, {"income": "income_report"})

    def test_check_cols_non_dict_sensitive_raises(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(TypeError):
            attack.posterior_vulnerability(self.qids, ["income"])

    # --- Dispatch smoke test ---

    def test_stub_raises_not_implemented(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        with self.assertRaises(NotImplementedError):
            attack.posterior_vulnerability(self.qids, self.sensitive)

    # --- Partitioning/aggregation correctness (via a fake injected strategy) ---

    def _fake_majority_strategy(self, noised_partition, true_partition):
        # Adversary guesses the majority noised report value for everyone in the partition;
        # scores it against each individual's true value.
        reports = noised_partition[:, 0]
        values, counts = np.unique(reports, return_counts=True)
        majority_value = values[np.argmax(counts)]
        return float(np.sum(true_partition == majority_value))

    def test_posterior_vulnerability_partition_and_aggregation(self):
        attack = NoisyAttack(self.data_true, self.data_noised, mechanism="grr")
        attack._strategies["grr"] = self._fake_majority_strategy

        posteriors = attack.posterior_vulnerability(self.qids, self.sensitive)
        self.assertAlmostEqual(posteriors["income"], 5/8)

        posteriors, distribution = attack.posterior_vulnerability(self.qids, self.sensitive, distribution=True)
        expected = [1.0, 0.5, 0.5, 1.0, 1.0, 1/3, 1/3, 1/3]
        for got, exp in zip(sorted(distribution["income"]), sorted(expected)):
            self.assertAlmostEqual(got, exp)
        self.assertAlmostEqual(np.array(distribution["income"]).mean(), posteriors["income"])

if __name__ == "__main__":
    unittest.main()
