# Frozen decision panel and result semantics

`datasets/campaign_snapshots.csv` contains 27,521 candidate-decision-scenario rows: 3,943 from 2023 and 23,578 from 2024. Scenarios reuse observations; rows are not independent field events. The evaluator expands the panel by evaluation mode, producing 51,099 scored rows in the stored prediction file. The primary external 2024 scenario contains 3,921 rows on 14 decision dates: 1,468 have known labels and 2,453 remain unknown.

| Field or group | Meaning |
|---|---|
| `campaign`, `scenario` | Campaign year and frozen quality/radius/horizon/latency setting. Main setting: `qaccepted_r250_h3_l24`. |
| `decision_at`, `window_end`, `window_complete` | UTC decision and outcome-window limit; completeness is administrative, not a guarantee of an observation. |
| `source_id` | Prefix-defined source anchor identifier; this is not an independent physical-facility census identifier. |
| `candidate_known_at` | First candidate availability under the stated lag assumption. |
| `feature_max_observed_at`, `feature_max_available_at` | Latest contributing observation and availability times, retained to audit information cutoffs. |
| `history_detect_count`, `history_observation_count`, `latest_label` | Prefix-only accepted detections, qualified known observations, and latest historical label. |
| `days_since_last_detection`, `max_emission`, `has_emission` | Historical recency and available reported-flux summaries; absent/hidden flux is not zero flux. |
| `history_max_daily_observations` | Maximum number of eligible historical observations within one day. |
| `anchor_longitude`, `anchor_latitude`, `cell_id`, `region` | Fixed source anchor and original projected 5 km block ID. Baseline `region` is the basin label, not the revised survey-polygon partition. |
| `horizon_days` | Derived by the evaluator: fixed outcome-window length for that scenario. |
| `outcome_scene`, `outcome_observed_at`, `outcome_available_at` | First qualifying future reference scene and its acquisition/assumed availability time, when present. |
| `outcome_label` | 1: accepted detection; 0: qualified catalog nondetection; blank: unknown. A catalog nondetection does not establish physical zero emissions. |
| `first_future_scene`, `first_future_status`, `first_future_at` | Audit fields retaining reference selection and unknown reason; an unresolved first reference is not replaced by a later detection. |
| `evaluation_mode`, `score_*` | Added by evaluation. Modes distinguish fixed 2023-to-2024 transfer and within-campaign walk-forward fits. Scores are frozen before evaluation. |

All timestamps are UTC. A 24 h availability lag is an explicit retrospective assumption, not a measured historical release timestamp. The source data were retrieved later; publication timing cannot be inferred from current API availability.

## Reference evidence and completion

Known positive, known catalog nondetection, unresolved outcome of an actual qualified scene, and no qualifying future scene are distinct states. Administrative incomplete windows are excluded according to construction rules. Unknown labels remain blank/NaN throughout; they are not background zeros.

For the actual joint completion analysis, a shared outcome key consists of campaign, source ID, and actual reference scene. Different sources within the same scene are different outcomes. A no-scene slot additionally includes the decision date and is never merged across dates. Filling these no-scene binary score slots is an explicitly hypothetical evaluation extension, not recovery of a historical emission status.

`joint_bounds_summary.csv` reports equal-date macro completion intervals after valid shared-outcome constraints. `per_day_bounds_summary.csv` and `unknown_reason_width_summary.csv` retain daily and reason-specific widths. The intervals are not confidence intervals. `spatial_bootstrap_summary.csv` instead reports conditional fixed-model percentile uncertainty for verified-subset metrics; it has a different target and interpretation.

## Controlled masking and offline revelation

The controlled population contains the 1,468 originally known primary rows. Their 632 unique campaign/source/scene outcomes include 475 reused across decision dates. The original genuinely unknown population is excluded. Each date freezes its own ceil(0.25 × Nknown) budget before synthetic hiding. Uniform, history-weighted, and structured masks are diagnostics, not identified replicas of actual survey missingness.

`mask_reveal_curve_summary.csv` uses one common exact-size half-group mask, hiding 316 unique outcomes per replicate. It reports all integer revelation budgets over 200 masks for three comparisons. Ordering uses |Cg| only; hidden labels are used solely to evaluate the simulated revelation path. Direction certification requires the complete interval to lie above +0.02 or below −0.02. If the complete truth lies within that band, the direction-eligible denominator is zero. Width targets are 0.20, 0.10, and 0.05 precision-difference units, equivalent to 20, 10, and 5 percentage points.

Uniform expected remaining width is analytic under equal-cost perfect revelation without replacement. Its crossing count is not an expected random stopping time and does not yield a direction probability. No future flight is assumed to reveal an earlier emission state.
