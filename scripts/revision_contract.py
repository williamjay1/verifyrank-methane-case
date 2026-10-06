"""Outcome-free policy interface for the revision; no reference fields enter scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
import evaluate_campaigns as base

HISTORY_FIELDS = ['history_detect_count','history_observation_count','latest_label',
                  'days_since_last_detection','max_emission','has_emission']

def policy_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy the permitted history only; references remain in the caller's sidecar."""
    missing=set(HISTORY_FIELDS)-set(frame.columns)
    if missing: raise ValueError(f'Missing historical fields: {sorted(missing)}')
    return frame[HISTORY_FIELDS].copy()

def score_policy_input(history: pd.DataFrame, fitted: dict) -> dict[str,np.ndarray]:
    if list(history.columns) != HISTORY_FIELDS:
        raise ValueError('Policy input must contain only the ordered historical contract fields')
    features=base.feature_matrix(history)
    scores={
        'beta_rate':features['beta_rate'].to_numpy(),
        'last_label':history['latest_label'].to_numpy(float),
        'max_emission':history['max_emission'].where(history['has_emission'].eq(1),-1).fillna(-1).to_numpy(float),
        'uniform_random':np.zeros(len(history))}
    for name in ['logistic','hist_gradient_boosting']:
        scores[name]=fitted[name].predict_proba(features)[:,1] if name in fitted else np.full(len(history),np.nan)
    return scores

def score_campaigns(frame: pd.DataFrame):
    outputs,audit_rows=[],[]
    for scenario,data in frame.groupby('scenario',sort=True):
        for mode in ['external_2023_to_2024','within_campaign_walkforward']:
            test=data.loc[data.campaign.eq('2024')] if mode.startswith('external') else data
            external=None
            if test.empty: continue
            if mode.startswith('external'):
                fit_at=test.decision_at.min()
                external,audits=base.fit_fixed_models(base.training_rows(data,fit_at,'2023'))
                audit_rows += [{'scenario':scenario,'mode':mode,'campaign':'2024','fit_at':fit_at,**a} for a in audits]
            for (campaign,decision),day in test.groupby(['campaign','decision_at'],sort=True):
                if external is None:
                    fitted,audits=base.fit_fixed_models(base.training_rows(data,decision,campaign))
                    audit_rows += [{'scenario':scenario,'mode':mode,'campaign':campaign,'fit_at':decision,**a} for a in audits]
                else: fitted=external
                # The scorer receives neither a target nor a timestamp for a future scene.
                history=policy_input(day)
                scores=score_policy_input(history,fitted)
                # Merge scores with reference metadata only after all policies have been scored.
                scored=day.copy()
                scored['evaluation_mode']=mode
                for name,values in scores.items(): scored[f'score_{name}']=values
                outputs.append(scored)
    return pd.concat(outputs,ignore_index=True),pd.DataFrame(audit_rows)
