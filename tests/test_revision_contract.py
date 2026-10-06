import importlib.util
from pathlib import Path
import sys, unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import revision_contract as contract

class ContractTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame([[1,2,1,1.0,5.0,1],[2,3,0,2.0,np.nan,0]],columns=contract.HISTORY_FIELDS)
    def test_future_fields_removed(self):
        frame=self.history().assign(outcome_label=[0,1],first_future_at=['secret','secret'])
        self.assertEqual(list(contract.policy_input(frame)),contract.HISTORY_FIELDS)
    def test_future_field_rejected_at_scorer(self):
        with self.assertRaises(ValueError):contract.score_policy_input(self.history().assign(outcome_label=1),{})
    def test_future_perturbation_cannot_change_scores(self):
        a=self.history().assign(outcome_label=0,first_future_at='A')
        b=self.history().assign(outcome_label=1,first_future_at='B')
        sa=contract.score_policy_input(contract.policy_input(a),{})
        sb=contract.score_policy_input(contract.policy_input(b),{})
        for name in sa:np.testing.assert_equal(sa[name],sb[name])
    def test_copy_does_not_mutate_caller(self):
        a=self.history();x=contract.policy_input(a);x.iloc[0,0]=9
        self.assertEqual(a.iloc[0,0],1)

if __name__=='__main__':unittest.main()
