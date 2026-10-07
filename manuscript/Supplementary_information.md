# Supplementary Information

## Guide to the evidence

This supplement accompanies the VerifyRank software and evaluation article. S0 retains the full methane adapter, original evaluation procedures and analysis design; A documents source reconstruction, B controlled masks, C unique-reference demand, D preserved supplementary figures, E the new generic software and planning validation. The original archive and analyses are preserved. The generic core is validated separately from the application-specific reconstruction. References to equations in S0 concern that local methodological exposition, rather than the new main-text numbering.

## S0. Complete methane adapter and benchmark specification


#### Evaluation object and evidence contract

Consider a daily decision at time \(t\), a candidate inventory \(I_t\) of size \(N_t\), and an information set containing only observations acquired and available before \(t\). A policy assigns scores to that inventory. The evaluation adapter supplies a separate future reference, chosen by a declared observation rule within \((t,t+H]\). A valid replay specifies source identity, information availability, reference qualification, candidate formation and budget before comparing policies. Acquisition, availability and reference time are distinct fields; none may substitute for another simply because a catalogue lacks a timestamp.

We use four evidence states described in the main input schema. An accepted detection supplies label 1. A qualified catalogue nondetection supplies label 0, meaning that the accepted-detection criterion was not met under that observation, rather than that physical emissions were zero. An unresolved qualified scene is an actual observation with insufficient catalogue evidence for a binary label. A source with no qualified scene has no realized reference under this observation rule. For the last state, the completion calculation introduces a hypothetical binary score slot, not an unobserved response to an actual flight. Administratively incomplete windows are excluded before candidate evaluation and form neither of the unknown states.

The main symbols are \(m_t\), the effective source budget; \(w^p_{it}\), the expected inclusion weight of policy \(p\); \(K_t\) and \(U_t\), known and unknown records; \(c_{it}\), the normalized difference between two policies' weights; and \(C_g\), the macro-average coefficient of a unique reference group \(g\). The outcome \(y\) denotes the accepted-detection endpoint, never methane mass. The new main Figure 1 shows the generic software architecture. The methane history interface remains a separate application adapter.


#### Fixed lists, conditional summaries and paired completion

For a fixed source count \(k\), \(m_t=\min(k,N_t)\); for fraction \(q\), \(m_t=\min(\lceil qN_t\rceil,N_t)\). Ties at the ranking threshold receive equal inclusion weights, with \(0\leq w^p_{it}\leq1\) and \(\sum_i w^p_{it}=m_t\). Uniform selection is represented by \(w_{it}=m_t/N_t\), eliminating favourable random draws. The full-population score and pairwise contrast are

\[
P^p_t(y)=\frac{1}{m_t}\sum_{i\in I_t}w^p_{it}y_{it},\qquad
\Delta_t(y)=P^a_t(y)-P^b_t(y)=\sum_i c_{it}y_{it},\quad
c_{it}=\frac{w^a_{it}-w^b_{it}}{m_t}.\tag{1}
\]

Scores, lists and budgets in Equation (1) are frozen before conditioning on verification. By contrast, the primary conditional precision reranks the known candidates using the original full-population budget, capped by \(|K_t|\). It describes ranking within the verified subset; it is not the score of the frozen full-population recommendation, or selection of the same fraction of known candidates. We retain this familiar diagnostic but name its population and budget explicitly.

For binary unknown outcomes, define \(A_t=\sum_{i\in K_t}c_{it}y_{it}\). The paired completion interval is

\[
L_t=A_t+\sum_{i\in U_t}\min(0,c_{it}),\qquad
R_t=A_t+\sum_{i\in U_t}\max(0,c_{it}),\qquad
W_t=R_t-L_t=\sum_{i\in U_t}|c_{it}|.\tag{2}
\]

An unknown selected identically by the two policies cancels. Subtracting their separately constructed completion intervals allows incompatible completions of the same outcome and generally widens the result. Equation (2) is sharp over the stated binary completion set for a single date; it requires no stochastic independence assumption. It applies established incomplete-outcome reasoning (Manski, 1989; Lakkaraju et al., 2017), rather than proposing a new missing-data identification theorem. The interval is an evidence-completion range, not a confidence interval. It excludes measurement errors within the known labels and supplies no outcome identification for an alternative flight schedule.

For \(T\) equally weighted dates, the same actual source–scene outcome may enter several windows. Group these occurrences by campaign, fixed source and qualified scene, setting

\[
C_g=\frac{1}{T}\sum_{(i,t)\in g}c_{it},\quad
A=\frac{1}{T}\sum_t\sum_{i\in K_t}c_{it}y_{it},\quad
L=A+\sum_{g\in U}\min(0,C_g),\quad
R=A+\sum_{g\in U}\max(0,C_g).\tag{3}
\]

Each no-scene slot receives its own source–decision group. Sharing an unspecified no-scene label across dates would impose an unsupported constraint on different hypothetical references. Equation (3) is sharp for the declared joint completion space. The average of Equation (2) is a valid outer interval and can be wider when coefficients for the same actual unknown reference change sign across dates. Joint grouping is therefore checked, not assumed to tighten every comparison. Splitting \(\sum_g|C_g|\) by evidence state attributes width to absent coverage or ambiguous actual references.

#### Verification demand in unique reference outcomes

Suppose that an actual unresolved reference group can be resolved without error at equal unit cost. Revealing it removes \(|C_g|\) from joint width, regardless of its eventual label. For a budget of \(b\) unique revelations, ordering groups by decreasing \(|C_g|\) gives

\[
W_{\min}(b)=W_0-\sum_{j=1}^{b}|C|_{(j)},\qquad
E[W_{\mathrm{uniform}}(b)]=W_0(1-b/G),\tag{4}
\]

where \(G\) is the number of hidden unique reference outcomes, including zero-coefficient groups. The second expression is the expected remaining width under a uniform fixed-size sample. Its threshold crossing is not the expected stopping time of a random sequential acquisition policy. Equation (4) is an equal-cost finite-set diagnostic; unequal costs, imperfect recovery and operational feasibility require additional inputs. Untied deterministic lists can produce equal coefficients over a disagreement set, giving no finer prioritization within that set.

In this recovery diagnostic, \(W_0\) sums only hidden actual reference groups; no-scene slots are outside the recoverable population. If an application includes irrecoverable score slots, their residual width must be retained separately and cannot be reduced by revealing an actual reference.

We assess width targets of 0.20, 0.10 and 0.05 and directional evidence beyond a difference threshold \(\epsilon=0.02\). Direction is certified only if the entire remaining interval lies above \(+\epsilon\) or below \(-\epsilon\). When the complete finite-population contrast lies inside this band, it is ineligible for a directional certification rate. Width and direction are separate requirements: knowing that two policies are close can narrow width without certifying superiority. Verification simulations below hide and reveal actual observed outcomes, never recover nonexistent historical scenes or create new physical observations.

#### Methane archive adapters and chronological reconstruction

The test case comprises two GAO campaigns in the Permian Basin. The 2023 Carbon Mapper annotated plume and scene snapshot was retrieved on 5 October 2026 for acquisitions from 16–24 August 2023, ending at 00:00 UTC on 25 August. It contains 718 published methane plume records and 145 scenes. The 2024 archive accompanying Cusworth et al. (2026) contains 1,422 distinct plume IDs and 272 scenes for 30 April–17 May, with an administrative endpoint of 00:00 UTC on 18 May. These are short annual observation campaigns from one basin, not independent random samples of oil and gas facilities.

We projected plume locations to an azimuthal equidistant system centred at 32° N, 103° W. An accepted plume joins the nearest existing fixed anchor within 250 m or creates a new anchor at its reported location. Anchors never move or merge. Accepted records are processed first at a shared timestamp, with stable identifiers resolving order. Excluded records may join existing anchors and mark uncertain evidence but cannot discover candidates. This prefix-based identity avoids importing future plume locations into historical source clusters. Anchors approximate emitting locations, without independently verified equipment or facility identities. Association radii of 100 and 500 m provide sensitivity checks.

In 2024, acceptance requires status `valid` or `publish_ready` and plume quality `good` or `questionable`; a strict variant requires `good`. A scene must be assessed, non-cloudy, free of flagged low signal-to-noise and geolocation errors, and cover the fixed anchor using its exact polygon. The 2023 public plume-quality fields are missing, so publication status defines the accepted set without an invented strict-quality equivalent. Its scenes require a non-cloudy flag and cloud cover no greater than 25%; four returned scenes fail this qualification. Because those responses supply bounding boxes, coverage is obtained from the service's exact-point intersection query for each required anchor, not from bounding-box containment. All 274 required points were successfully queried.

At a qualified source–scene record, any accepted associated plume establishes detection. Associated plumes that are all excluded establish unknown evidence. No associated plume establishes a qualified catalogue nondetection only if scene-level catalogue completeness is supported. A 2024 source-archive ID is absent from both plume tables; absence-based outcomes in its scene are therefore unresolved while directly supported detections remain known. The retrospective archive supplies this completeness check, never candidate centroids or memberships. A historical source–scene flux sum is used only when every associated member is accepted and has a visible nonmissing emission rate. This avoids converting incomplete quantification into a low flux predictor.

These rules concern catalogue reference validity, not perfect physical detection. Source emissions can fall below instrumental sensitivity despite a qualified scene (Ayasse et al., 2023, 2024; Thorpe et al., 2024). We impose neither a universal flux threshold guaranteeing detection nor a probability-of-detection curve from another instrument. Qualified nondetection remains distinct from physical non-emission.

#### Decision time, policies and fitting

Daily decisions occur at 00:00 UTC. Historical observations require both acquisition and assumed availability strictly before the decision. The primary availability delay is 24 h; alternatives are 0 and 48 h. A source joins the candidate inventory after its first accepted detection becomes available. Unknown historical observations remain in the archive but do not contribute to binary counts. Future references use the first geometrically and scene-quality qualified observation in \((t,t+H]\), with \(H=3\) days primarily and seven days in sensitivity analysis. If that first scene is ambiguous, a later positive cannot replace it. The endpoint is a variable-time next qualified observation inside the window, not activity exactly \(H\) days later or any activity during the window.

Availability delays are hypothetical information scenarios. The earliest recorded 2023 public plume release occurred more than 37 days after acquisition; the 2024 archive was released after its campaign and lacks usable in-campaign public availability timestamps. The replay does not reconstruct actual real-time public access. Current metadata may also reflect subsequent processing revisions. Its daily recommendations do not alter archived history, remove sources after hypothetical visits or optimize routes. The equal unit source budget is not a flight-hour or repair-cost budget.

Four reference policies rank by Laplace-smoothed historical detection rate \((d+1)/(n+2)\), latest known state, maximum complete historical flux, or uniform random selection. The rate is an auditable history rule, without asserting independent repeated Bernoulli trials. Two interchangeable learners use six features: the smoothed rate, \(\log(1+n)\), latest label, days since last accepted detection, \(\log(1+\text{maximum flux})\), and flux availability. Missing flux is zero only in a transformed predictor with a missingness indicator. Logistic regression uses training-fitted standardization and L2 regularization with \(C=1\). Histogram gradient boosting uses 100 iterations, seven maximum leaves, minimum leaf size 20, learning rate 0.1, L2 regularization 1, no early stopping, and fixed seed 20261005. Neither learner is chosen or tuned on 2024 outcomes.

The primary transfer fit uses eligible 2023 examples and freezes models before 2024 scoring. Expanding-window fits are evaluated separately within each campaign. Training requires a known outcome, a completely elapsed target window and reference availability strictly before fitting. At least 20 distinct training sources and five examples per class are numerical fitting guards, not power claims. The project began with exploratory inspection of 2024 data; neither transfer nor this revision is preregistered or independently confirmatory. The revision freezes the panel and scores before comparing estimators and verification rules. All scenarios and budgets—10, 25 and 50 sources, and 10% and 25% fractions—are retained, including skipped learned fits.

#### Controlled masking and fair evaluation controls

For masking, each day's originally known 2024 references define a complete restricted population. Scores and a budget \(m_t=\lceil0.25|K_t|\rceil\) are frozen before hiding labels. This target differs from the main full-inventory score. No models are refitted. Uniform and history-weighted sampling without replacement retain \(\lceil r|K_t|\rceil\) labels for \(r=0.25,0.50,0.75\), with 200 seeded repetitions per setting. History weights equal one plus historical observation count and never use future outcomes. Additional half-retention designs share masks across dates by unique source–scene outcome, entire scene or acquisition day. Group sampling retains half the groups rounded upward; its realized row retention need not equal half.

We compare five point procedures under each applicable mask: reranking retained candidates with the original count; reranking with the original fraction of retained candidates; the verified portion of each frozen list normalized by its own verified selection mass; uniform-design Horvitz–Thompson (HT) estimation; and zero filling under frozen weights. The frozen conditional pair is undefined if either selected verified mass is zero. For uniform row sampling, \(\pi_{it}=\lceil r|K_t|\rceil/|K_t|\) is the exact inclusion probability and

\[
\widehat\Delta_{\mathrm{HT},t}=\sum_{i\in K_t}\frac{R_{it}c_{it}y_{it}}{\pi_{it}},\tag{5}
\]

with \(R_{it}\) the retention indicator (Horvitz and Thompson, 1952). We do not clip this estimator, derive inverse-probability estimates from normalized history sampling weights, or apply row-design HT to grouped masks. Its design-unbiasedness is assessed on the complete uniform mask population. Conditioning on masks where every point method is defined changes the sampling distribution and removes that guarantee. Fair common-valid error comparisons therefore emphasize the four non-HT methods; full-design HT results are reported separately.

Absolute error and signed error are measured against the original frozen complete-label pair. We preserve method-specific availability and also compare methods on common-valid masks. Reversal uses dates whose absolute complete-label difference is at least 0.02; estimated ties and undefined estimates remain separate. Fixed effect bins are [0, 0.02), [0.02, 0.05), [0.05, 0.10), [0.10, 0.20) and [0.20, infinity), with counts of unique decision dates alongside mask repetitions. Synthetic repetitions are not independent campaigns or evidence for field significance tests.

Masking error summaries weight each defined date–mask equally. Common-valid comparisons use the same surviving masks for the four non-HT procedures within each pair; separate equal-date macro summaries are retained in the supplement. Surviving mask populations can differ between pairs.

A separate macro verification experiment masks 316 of 632 unique known source–scene outcomes, consistently across all appearances on 14 dates, in 200 repetitions. Equation (3) supplies one joint interval per policy pair and repetition. Revelation order uses coefficients alone, with stable source–scene keys breaking ties; complete labels are consulted only after choosing a group. Every integer revelation budget from 0 to 316 is retained. This avoids charging several times for one reused outcome or drawing different hidden labels for its repeated appearances.

#### Uncertainty, data checks and software reproduction

For the campaign ranking evaluation, daily performance is macro-averaged with equal date weights; paired estimates require dates on which both policies are available. Combined within-campaign summaries give campaigns equal weight. Brier error and fixed-width calibration bins describe probability quality only on known outcomes (Gneiting and Raftery, 2007). Learned coefficients are descriptive predictors, not physical or causal mechanisms. For primary conditional differences, 1,000 replicates resample spatial blocks with all constituent sources and dates, recomputing budgets and ties without refitting models. The primary grid is 5 km; 2.5 and 10 km grids and half-cell shifts in both coordinate axes check sensitivity. Leave-one-decision-date-out estimates examine concentration in individual dates. These intervals condition on fitted models, campaign selection and grid structure, and do not capture every shared flight-day disturbance (Roberts et al., 2017).

We reconcile the 2024 funnel at unique plume-ID level and independently reconstruct raw classifications. A separate scalar implementation checks 80 seeded records across detection, nondetection, missing-scene and ambiguous states, all nine association-boundary records 225–275 m from an existing anchor, and the complete accepted and strict anchor sets. This is independent metadata/code checking, not human image interpretation or a calibrated label-error study. Archived East, West and full-domain polygons define mutually exclusive spatial strata with overlaps and out-of-domain membership retained. Region-specific ranking reuses frozen scores but defines a separate regional 25% budget; it is exploratory and not an additive decomposition of global policy performance.

The implemented scorer accepts only six historical raw fields. It has no label, first-reference scene, future timestamp, regional membership or spatial-block input; predictions are joined to the reference sidecar only after scoring. Tests perturb future metadata, reject extra scorer columns and check prefix invariance, elapsed training windows, tie budgets, completion extrema, shared-reference grouping, mask inclusion probabilities and unique verification cost. A fresh virtual environment with independently installed locked dependencies re-executes fitting and evaluation from the frozen panel. Separately, immutable raw responses rebuild the entire panel in a new directory. Timing reports distinguish these stages, exclude network acquisition, and identify the analytic process used for memory measurement. Native PDF/SVG graphics and direct 1200 dpi TIFF exports use the GitHub SciencePlots Nature style with final-size sans-serif typography; numerical source tables accompany every data figure. Graphics quality checking covers text collisions, clipping, line intersections and visual inspection.

**Algorithm 1. Verification-aware comparison.** Inputs are versioned observations, qualification and availability rules, decision dates, ranking functions and budgets. (1) Process observations in acquisition order to establish fixed source anchors and their qualified histories. (2) At each decision, admit only candidates and predictors acquired and available before that time, excluding incomplete administrative windows. (3) Fit on eligible earlier examples where required; pass historical fields to each scorer and freeze the full-inventory inclusion weights. (4) Build the first-qualified-reference sidecar and its four evidence states, then join it to the frozen recommendations. (5) Calculate the named conditional diagnostic and full-inventory paired endpoints, retaining common reference identities across dates. (6) Group actual repeated references, decompose joint width by evidence state, and compute recovery demand only for the actual recoverable-reference population. (7) Export denominators, intervals, availability failures and all declared sensitivity settings. Outputs are comparison evidence and reference demand, not an adaptive acquisition policy.



## A1. Data sources and units

The 2024 inputs are the authors' version 1 archive accompanying Cusworth et al., *Duration of super-emitting oil and gas methane sources* ([article](https://doi.org/10.1038/s41467-026-68804-7); [data and code](https://doi.org/10.5281/zenodo.17862304)). The archive provides two plume CSVs, scene geometries, source associations, infrastructure attribution, survey-region polygons and an analysis notebook. The campaign interval used here is 30 April–17 May 2024, with the right endpoint at 00:00 UTC on 18 May. The archived scene catalog contains 272 scenes; 267 meet our joint scene criteria: assessed, not cloudy, no low-signal flag and no geolocation-error flag.

The 2023 inputs are frozen responses from the publicly accessible Carbon Mapper annotated plume and scene endpoints, retrieved in October 2026 for the interval from 00:00 UTC on 16 August through, but not including, 00:00 UTC on 25 August 2023. The catalog contains 718 methane plumes and 145 scenes. Of these scenes, 141 satisfy `not_cloudy=true` and cloud cover at most 25%; 17 plumes belong to the four excluded scenes. Plume quality grades are unavailable for all 718 records. The accepted 2023 cohort is therefore a currently published, scene-qualified cohort and is not equivalent to the 2024 strict good-quality cohort. Visible emission estimates exist for 617 catalog records, before the other eligibility conditions are applied.

Three counting units must remain distinct. A **plume record** is a unique reported plume ID. A **source–scene record** combines plume evidence, or qualified absence of such evidence, at a fixed chronological anchor in one acquisition. A **candidate–decision record** is an anchor eligible for ranking on one decision date. Repeated source–scene or candidate–decision records are not independent physical sources. The prediction target is a qualified observed detection at the first eligible future scene, rather than emission status at all times or proof of physical non-emission.

## A2. Exact reconciliation of the 2024 plume counts

The original plume files contain 1,422 rows and 1,422 unique plume IDs; no duplicate plume rows are removed. All records lie within the campaign interval. Their status counts are 1,221 `publish_ready`, 157 `valid` and 44 `hidden`; their quality counts are 1,011 `questionable`, 404 `good` and seven `bad`.

We reproduced the relevant steps in cells 4–10 of `AnalysisCode_Cusworth_Permian.ipynb`. Its accepted-status filter yields 1,378 unique IDs. Joining the archived 250 m source table preserves 1,378 rows. Joining infrastructure attribution then produces 1,380 rows, because the attribution file contains repeated entries for `GAO20240506t184846p0000-C` and `GAO20240514t160455p0000-K`. Each contributes one additional joined row. Applying the notebook's infrastructure-specific source identifier construction reproduces 529 sources. The archived notebook's count is therefore explained by

\[
1{,}422-44+2=1{,}380\text{ joined rows},
\]

with 1,378 unique accepted plume IDs. This is a reconciliation of the archived files and notebook, rather than an assertion about an unobserved upstream database or a comprehensive reassessment of the original study. Our chronological anchors also differ from the original study's retrospective, infrastructure-split source definitions.

Table A1 reports the monotone unique-ID funnel used by our accepted-quality panel. Quality and scene exclusions are sequential: the three rejected bad-quality IDs are those remaining after status screening, not all seven bad-quality records in the original files.

**Table A1. Unique plume-ID funnel for the accepted 2024 panel.**

| Stage | Retained unique IDs | Removed at this stage |
|---|---:|---:|
| Raw CSV concatenation | 1,422 | — |
| Deduplication | 1,422 | 0 |
| Campaign interval | 1,422 | 0 |
| Accepted status | 1,378 | 44 |
| Good or questionable quality | 1,375 | 3 |
| Qualified acquisition scene | 1,354 | 21 |
| Fixed anchor covered by the qualified scene polygon | 1,350 | 4 |

The retained 1,350 accepted plume members yield 1,274 positive source–scene records. The complete accepted-radius-250 m observation table contains 2,164 source–scene records: 1,274 positive, 886 non-detection and four unknown. Its chronological inventory contains 496 anchors. Not every anchor becomes a candidate on a decision date: the primary 2024 ranking panel contains 399 distinct candidates across 3,921 candidate–decision records.

The association stream contains 1,373 plume members in total, including rejected members assigned to existing anchors. Those rejected members can make a label unknown or an emission sum incomplete; 1,373 is consequently not a stage in the accepted-record funnel. The source archive also lists `GAO20240503t190640p0001-A`, which is absent from both plume CSVs. The panel preserves directly supported positives in this acquisition but marks absence-based labels in the affected scene unknown. It does not locate the missing plume using a retrospective source centroid.

The full ID-level audit is in `data_audit/plume_funnel_by_id.csv`, with raw filename, original row number, each eligibility flag, assigned anchor and first exclusion reason. Aggregate counts are in `plume_funnel.csv` and `funnel_summary.json`. The author-notebook comparison and duplicate attribution rows are in `author_notebook_reconciliation.csv` and `author_join_duplicate_records.csv`, respectively.

## A3. Survey geography and regional evaluation

### S3.1. Original polygons and mutually exclusive membership

We used the archived `NM Permian 2024 - E.geojson`, `NM Permian 2024 - W.geojson` and `WideMap.geojson`. Their SHA-256 checksums and filenames are recorded in `data_audit/region_polygon_manifest.csv`. Membership is determined by whether the original polygon covers the fixed anchor point, including its boundary. Areas computed using the local azimuthal equidistant projection are 1,224.0543 km² for east, 1,209.5907 km² for west and 11,792.0076 km² for the supplied full-domain polygon.

The original geometries are neither exactly disjoint nor strictly nested. East and west overlap by 0.0219632 km², but no primary candidate lies in that overlap. Their union extends 50.3358 km² outside the supplied full polygon. We therefore retain all three membership indicators and assign mutually exclusive strata in this order: east–west overlap, east only, west only, full-domain residual, and outside all three polygons. Here, “east only” and “west only” distinguish the two intensive polygons; they can include points also in the full-domain polygon. The full-domain residual excludes both intensive polygons. No supplied boundary was silently trimmed or redrawn.

**Table A2. Primary 2024 candidate coverage within the original survey design.**

| Stratum | Candidate–decision records | Distinct candidate anchors | Dates with candidates | Known labels | Positive | Non-detection | Known / all candidate records |
|---|---:|---:|---:|---:|---:|---:|---:|
| East only | 1,587 | 124 | 14 | 803 | 327 | 476 | 50.5986% |
| West only | 920 | 105 | 13 | 593 | 290 | 303 | 64.4565% |
| Full-domain residual | 1,323 | 161 | 14 | 40 | 23 | 17 | 3.0234% |
| Outside all polygons | 91 | 9 | 14 | 32 | 20 | 12 | 35.1648% |
| Total | 3,921 | 399 | 14 | 1,468 | 660 | 808 | 37.4394% |

The unrepresented east–west overlap stratum contains zero candidates. The 2,453 unknown primary outcomes comprise 2,451 cases with no qualified future scene and two ambiguous first-scene outcomes. Table A2 describes pooled candidate–decision coverage; it is not a mean across equally weighted dates. Candidate membership and stratum counts are available in `region_candidate_membership.csv` and `regions.csv`.

### S3.2. Budget, tie handling and denominators

Regional evaluation reuses the frozen externally trained 2023-to-2024 scores. For stratum \(r\) on date \(d\), the budget is

\[
B_{rd}=\lceil 0.25N_{rd}\rceil,
\]

where \(N_{rd}\) is the number of all eligible regional candidates, including candidates with unknown outcomes. Each region is ranked separately. This defines an exploratory regional allocation task; its results do not add up to the global policy value, and no model is refitted within a region. Dates without regional candidates are omitted rather than assigned a value of zero.

Scores alone determine selection. At a tied cutoff, the unfilled slots are distributed uniformly in expectation across all tied candidates. Let \(w_{ird}^{(m)}\) denote the resulting inclusion weight for policy \(m\), summing to \(B_{rd}\). Full-population precision endpoints divide by \(B_{rd}\). Conditional precision reranks only candidates with known outcomes, using \(\min(B_{rd},N_{rd}^{\mathrm{known}})\) slots and that selected known count as its denominator. A date with no known outcomes supplies no conditional precision estimate. All policies are compared over the same eligible dates within each metric.

For policies \(a\) and \(b\), define \(c_i=(w_i^{(a)}-w_i^{(b)})/B_{rd}\). If \(K\) and \(U\) identify known and unknown outcomes, respectively, the shared-completion difference interval is

\[
\left[
\sum_{i\in K}c_i y_i+\sum_{i\in U}\min(0,c_i),\quad
\sum_{i\in K}c_i y_i+\sum_{i\in U}\max(0,c_i)
\right].
\]

Date-specific endpoints are computed first and then averaged with equal date weights. The two policies share each candidate's unknown binary outcome; unknown candidates with equal inclusion weights cancel. Separate policy intervals are also retained for comparison. Averaging valid date-specific intervals gives valid outer bounds, but the average need not be jointly sharp when overlapping windows share the same unresolved acquisition outcome. These are finite-sample completion bounds, not confidence intervals, causal effects or measured deployment benefits. No bootstrap uncertainty statement is made for this supplementary subgroup analysis.

### S3.3. Regional results

**Table A3. Conditional precision by region, expressed as percentages.** All policies within a column use the same conditional dates and budgets.

| Policy | East only | West only | Full-domain residual | Outside all polygons |
|---|---:|---:|---:|---:|
| Beta rate | 67.8298 | 74.5109 | 40.8225 | 76.6667 |
| Latest label | 60.3696 | 62.4763 | 40.8225 | 72.3333 |
| Maximum emission | 55.4043 | 51.3358 | 31.7316 | 60.0000 |
| Uniform random expectation | 50.8424 | 49.5723 | 37.7922 | 67.6667 |
| Logistic regression | 67.3446 | 74.1052 | 40.8225 | 66.6667 |
| Histogram gradient boosting | 56.5688 | 58.7534 | 40.8225 | 70.0000 |
| Dates supplying conditional precision | 12 | 10 | 11 | 10 |
| Dates supplying full-policy bounds | 14 | 13 | 14 | 14 |

**Table A4. Shared-completion bounds for regional precision differences, in percentage points.** The named first policy is compared with the second policy.

| Stratum | Logistic − beta | Gradient boosting − beta | Beta − uniform random |
|---|---:|---:|---:|
| East only | [−10.1492, 7.0766] | [−33.4456, 12.6591] | [−17.0516, 48.4627] |
| West only | [−11.9116, 10.0385] | [−29.4520, 8.9576] | [−5.6243, 40.7012] |
| Full-domain residual | [−28.0475, 27.8677] | [−49.3458, 44.0160] | [−48.4292, 50.8453] |
| Outside all polygons | [−15.2778, −6.1508] | [−28.9683, 15.2778] | [−32.4943, 48.7528] |

The intensive regions supply substantially more evaluation coverage than the full-domain residual. This geographical association accords with the documented survey objectives but does not identify the cause of missing labels. In the residual stratum, only 40 of 1,323 candidate–decision outcomes are known. Its similar conditional scores therefore should not be interpreted as evidence of equivalent model performance or reliable broad-area deployment. The signed logistic–beta interval outside the polygons is retained transparently, but it concerns only nine distinct anchors and is not treated as a primary finding. All six policies, per-date denominators, full-policy endpoints and paired endpoints are retained in `region_policy_daily.csv`, `region_policy_summary.csv`, `region_paired_daily.csv` and `region_paired_summary.csv`.

## A4. Separate metadata and geometry checks

A separate scalar implementation read the original plume CSVs and scene geometries, reconstructed chronological fixed-anchor associations and classified source–scene evidence without calling the production builder's classification functions. It reproduced the IDs, anchor plume IDs and coordinates of all 496 accepted and 212 strict 250 m anchors. Future plumes did not relocate or merge previously created anchors in this reconstruction.

The prespecified revision audit seed was 20261006. From the primary 2024 panel, we sampled 20 positive, 20 non-detection and 20 no-qualified-scene candidate–decision records. Both primary ambiguous records were examined; 18 randomly selected strict-quality ambiguous records completed the 20-record ambiguity stratum. For each sampled record, the independent implementation checked raw plume status and quality, scene assessment flags, point-in-polygon coverage, associated member IDs and the earliest eligible future acquisition inside the evaluation window. It retained an ambiguous first acquisition instead of skipping ahead to a later known outcome. All 80 classifications and scene identifiers matched the frozen panel.

The separate radius-boundary pool included every plume record whose nearest previously created anchor lay 225–275 m away. Only nine records met this criterion, so all nine were examined, rather than claiming a 20-record sample. All nine matched the 250 m association rule. Raw member completeness was also checked when constructing emission evidence: an accepted positive can remain valid while its emission sum is unknown if any member is rejected, hidden or lacks a usable emission estimate.

The 2023 coverage audit read all 286 retained point-query caches, including superseded requests preserved for provenance. Their 1,584 point–scene memberships were checked against catalog scene UUIDs and timestamps. The current required subset comprises 274 point queries, with 1,534 memberships involving 135 distinct scenes. All 274 current queries succeeded and had nonempty results. Query points match the original anchor plume points to within \(10^{-12}\) degrees; the tiny differences reflect CSV floating-point round trips. Scene bounding boxes were not substituted for point coverage, and missing or failed queries were not converted into successful empty responses.

Evidence files are `independent_record_audit.csv`, `radius_boundary_record_audit.csv`, `independent_anchor_comparison.csv`, `point_query_uuid_audit.csv` and `record_audit_summary.json`. This is an independent AI-agent audit of raw metadata and geometry. It is not human expert review, facility verification, plume-image interpretation or physical validation of zero emissions. The stratified checks do not estimate a population-wide error rate.

## A5. Raw-input reconstruction and computational cost

The reconstruction starts from the immutable downloaded responses and archive files. It verifies 291 raw 2023 response hashes—five catalog pages and 286 point-query responses—and all 11 files in the 2024 download manifest. The 2023 catalog lists and point-coverage mapping are rebuilt directly from these responses. A copy of the existing panel builder, unchanged, then writes a new output directory. Neither the existing normalized 2023 catalog lists nor the frozen combined panel is used as the builder's input.

Both completed reconstructions reproduced all 16 compared CSVs byte for byte, including the 27,521-row combined panel. The final timed repetition required 65.3767 s for raw integrity checking, reconstruction and output comparison, of which 65.0210 s was spent in the builder. The observed Windows peak working set of the builder process was 158,695,424 bytes, or 151.34375 MiB. Memory was sampled every 0.05 s using the Windows process API, including its reported peak working set. These figures exclude acquisition network time and the supervisory interpreter's memory; they describe this reconstruction stage, not the full model training and evaluation workflow.

The combined panel SHA-256 is `651b7e0734b1ebe4a4a197bcc1d64684a1a78855693b4d080e2b214651fe2b68`. File-level comparisons, resource measurements and environment versions are in `raw_rebuild_hash_comparison.csv`, `raw_rebuild_report.json` and `audit_environment.json`. Core package versions were NumPy 2.5.1, pandas 3.0.5, pyproj 3.7.2 and Shapely 2.1.2. Four synthetic edge-case tests passed, covering overlap partitioning, incomplete-catalogue absence, retention of the first eligible scene, and shared-unknown cancellation.

From the project root on the recorded Windows environment, the following command creates a new reconstruction. The subdirectory name must not identify a previously completed run:

```powershell
python -B scripts/revision_data_audit.py --rebuild-only --rebuild-subdir raw_rebuild_next
```

Manifest paths must resolve to the unchanged raw files on the reproducing machine. The Windows memory monitor is platform-specific; this command is not asserted to work unchanged on Linux. Raw hash checks and reconstruction results are separate from the question of whether the live API will return identical data on a later date.

## A6. Availability, provenance and license boundaries

The 2024 Zenodo record is a fixed public version whose metadata identifies CC BY 4.0; its notebook additionally links to Carbon Mapper's terms. The 2023 inputs are a dated snapshot of a mutable service, not a reconstruction of its historically available catalog. A simulated publication delay does not establish that any record was actually available on the historical decision date.

The [Carbon Mapper terms effective 13 January 2026](https://carbonmapper.org/terms) permit publication and redistribution of data and derivatives subject to noncommercial, purpose, attribution and equivalent redistribution conditions. Accordingly, a mixed-year derived panel must not be described as unrestricted CC BY data. The public case repository preserves source-specific terms and explicit Carbon Mapper attribution; its derived data are not MIT-licensed. The generic software and case materials are now available in separate GitHub repositories listed in E4. New Zenodo deposits remain pending the author's action; no new DOI, special provider permission or human scientific sign-off is claimed. Official-page retrieval records and the detailed provenance note are in `official_source_snapshots.json` and `sources_license_notes.md`.


## B1. Experimental population and frozen comparison target

The revision-stage experiment diagnoses how evaluation procedures behave when labels are removed from a population whose reference outcomes are already known. It uses the primary accepted-quality, 250 m anchor, three-day horizon and 24 h information-delay scenario, with the fixed models trained on the 2023 activity and scored on the 2024 activity. The models, scores, source definitions and reference labels were not refitted, tuned or revised for this experiment.

For each decision date (d), the experimental population consists of the (N_d) candidates with an originally known future reference outcome. The combined population contains 1,468 source–decision records across 14 decision dates, representing 254 source identifiers, 632 unique source–reference-scene outcomes, 117 reference scenes and 10 acquisition days. The 2,453 originally unknown records are excluded from this controlled experiment. It therefore cannot reveal their outcomes or identify the actual archive missingness mechanism. Repeated source–decision records are not independent facilities or emission events.

The frozen budget is (m_d=\lceil0.25N_d\rceil). Each policy receives label-independent inclusion probabilities (w_{Aid}\in[0,1]), with (\sum_i w_{Aid}=m_d), from the scores available before masking. Ties at the selection threshold are allocated equal inclusion probabilities. Uniform random selection is represented by its analytic inclusion probabilities. For policies (A) and (B), define

\[
c_{id}=\frac{w_{Aid}-w_{Bid}}{m_d},\qquad
\Delta_d=\sum_{i=1}^{N_d}c_{id}y_{id}.
\]

Here (y_{id}) is the complete, originally observed binary catalogue outcome. The known Δ is the reference target for all error calculations. It is not the original full-candidate policy difference from the main analysis: that analysis bases its budget on all daily candidates, whereas this controlled experiment bases its budget on each day's originally known population. Nor is Δ a quantity of methane emitted or mitigated.

## B2. Row-level masking designs and reproducibility

The two original row-level mechanisms were retained without changing their random streams. For nominal retention (r\in\{0.25,0.50,0.75\}), exactly (k_d=\lceil rN_d\rceil) labels are retained without replacement on each date. Uniform masking assigns equal inclusion probabilities. History-weighted masking supplies draw weights proportional to (1+h_{id}), where (h_{id}) is the historical qualified-observation count available before the decision. Neither mechanism uses the hidden labels to select retained cases. The same mask is applied to both policies in each comparison.

Each setting has 200 reproducible repetitions. The NumPy seed sequence is `[20261005, sorted decision index, mechanism index, retention index, repetition]`, with mechanism order uniform then history weighted and retention order 0.25, 0.50, 0.75. The original label-free mask generator is reused. This reproduces all 50,400 original pair–mask evaluations, with maximum numerical difference in original-count estimates below (10^{-16}). In particular, the original logistic–historical-rate reversal counts at nominal half retention remain 835/2,000 under uniform masking and 751/2,000 under history weighting.

Ceiling the retained count makes realized retention slightly larger than its nominal value on some dates. Across dates, the mean retained row fractions are 0.261136, 0.503150 and 0.757986 for nominal 0.25, 0.50 and 0.75, respectively. The uniform and history-weighted designs have identical retained counts; they differ in which labels are retained. All calculations use actual counts, including the Horvitz–Thompson inclusion probability.

## B3. Evaluation procedures and their distinct targets

Let (R_{id}) indicate that a label is retained. All procedures use the same scores and complete target Δ, but they do not estimate that target under the same assumptions. The following distinctions are essential when interpreting their errors.

**Original-count reranking.** The retained population is reranked separately by each policy, using the count budget (\min(m_d,\sum_iR_{id})). Its precision difference describes the newly selected lists in the retained population. It changes population, lists and selection fraction. At nominal 25% retention, the retained count equals the original budget by construction, so both policies select every retained case and their estimated difference is identically zero. This is a structural consequence of the evaluation rule, not evidence of equivalent policies.

**Original-fraction reranking.** The retained population is reranked using (\lceil0.25\sum_iR_{id}\rceil) selections. This preserves the nominal selection fraction rather than the original count, but still changes the population and selected lists. It is a mechanism-separation diagnostic, not a deployable policy that can identify future verified cases in advance. Either reranking procedure is undefined when no labels remain.

**Frozen verified conditional precision.** The original selection weights remain fixed. Each policy's observed weighted positive mass is divided by its own observed selected mass:

\[
\widehat\Delta_{\mathrm{FVC},d}
=\frac{\sum_iR_{id}w_{Aid}y_{id}}{\sum_iR_{id}w_{Aid}}
-\frac{\sum_iR_{id}w_{Bid}y_{id}}{\sum_iR_{id}w_{Bid}}.
\]

If either denominator is zero, the difference is undefined and recorded as missing. The two denominators are not forced to be equal. This contrast preserves the policy lists but conditions on verification within each list. It is a ratio-based diagnostic, not generally an unbiased estimator of the frozen full-population difference.

**Uniform fixed-size Horvitz–Thompson.** For the uniform row-level design, the first-order inclusion probability is known exactly: (\pi_{id}=k_d/N_d>0). The estimator is

\[
\widehat\Delta_{\mathrm{HT},d}
=\sum_i\frac{R_{id}c_{id}y_{id}}{\pi_{id}}.
\]

Its expectation over the complete specified masking design is Δ, because (E(R_{id})=\pi_{id}); label independence across units is not required for this expectation. This is the finite-population design principle of Horvitz and Thompson (1952). The estimator is not clipped to [−1,1]: clipping would change its expectation. Design unbiasedness does not guarantee small variance, low mean absolute error, or correct signs in individual masks.

The normalized history weights supplied to sequential weighted sampling without replacement are **not** the first-order inclusion probabilities. The retained probability of a unit depends on the entire design and is generally not its normalized draw weight. Accordingly, no history-weighted HT result is reported. HT was also not included for the three new grouped mechanisms under the frozen revision configuration. This restriction is a scope choice; it does not imply that valid group-design inclusion probabilities or estimators cannot be constructed.

**Zero filling.** Keeping the original weights and budget, the estimate is (\sum_iR_{id}c_{id}y_{id}). It treats every hidden contribution as zero and is a plug-in diagnostic with no general unbiasedness claim. In uniform fixed-size masking, HT is this estimate divided by a common positive π. Thus, the two estimates have the same sign for each mask, although their magnitudes and errors differ. The identical reversal counts reported for those two methods are expected.

**Paired completion bounds.** The interval is calculated from the original coefficients and retained labels:

\[
L_d=\sum_iR_{id}c_{id}y_{id}+\sum_{i:R_{id}=0}\min(0,c_{id}),
\qquad
U_d=\sum_iR_{id}c_{id}y_{id}+\sum_{i:R_{id}=0}\max(0,c_{id}).
\]

It is an exact per-decision interval over unrestricted binary completions, not a point estimate or confidence interval. The controlled complete target is necessarily contained when the implementation is correct. Successful containment tests the calculation, not physical measurement accuracy or the realism of a missingness mechanism. The reported daily-average intervals in this experiment remain distinct from the separately implemented cross-day joint-completion calculation.

## B4. Common-valid comparisons, full-design HT and denominator accounting

Two summary populations are provided. `method_defined` reports all finite estimates for each method, preserving that method's own validity count. `common_point_defined` restricts comparisons to the same masks on which original-count reranking, original-fraction reranking, frozen verified conditional precision and zero filling are all defined. This is the appropriate matched support for comparing those four diagnostics without confounding their error summaries by different missing-estimate patterns.

Conditioning on this shared support changes the masking distribution. HT's unbiasedness statement therefore applies to the **full uniform design** in the `method_defined` summary, not to its additional row in `common_point_defined`. A common-valid HT row is available as a descriptive comparison, but should not be presented as retaining full-design unbiasedness. The primary matched-method display can exclude HT and show its full-design results separately.

Errors are estimated difference minus complete frozen difference. Absolute errors, squared errors, sign reversals and ties are recorded for every defined estimate. A sign-reversal analysis is eligible when |Δ|≥0.02; estimated ties are kept separately. Floating-point comparisons use a tolerance of (10^{-12}). Eligibility comes from the complete target, not the masked estimate. In each mechanism and retention setting, the full eligibility denominators are 2,000 masks on 10 dates for logistic versus historical rate, 2,400 on 12 dates for boosting versus historical rate, and 2,600 on 13 dates for historical rate versus random. Each pair has 2,800 total date–mask evaluations before eligibility or validity restrictions.

For the half-retention uniform logistic comparison, the common-valid population has 2,674 evaluations, of which 1,963 meet the sign threshold. Table B1 uses this same support for all four methods. Its MAE values weight each surviving date–mask equally. Because validity restrictions can leave unequal numbers of masks across dates, the outputs also provide equal-date macro errors and their available-date counts. These are not silently substituted for the date–mask means.

**Table B1. Matched-validity comparison under nominal 50% uniform retention, logistic versus historical rate.** All methods use 2,674 common-valid evaluations for MAE and 1,963 eligible common-valid evaluations for reversals and ties.

| Diagnostic | MAE | Reversals | Reversal fraction | Eligible ties |
|---|---:|---:|---:|---:|
| Original-count reranking | 0.06652 | 808 | 41.16% | 246 |
| Original-fraction reranking | 0.05823 | 478 | 24.35% | 275 |
| Frozen verified conditional | 0.05172 | 391 | 19.92% | 133 |
| Zero filling | 0.06407 | 515 | 26.24% | 173 |

A second display retains all 2,000 eligible masks and gives undefined results their own category (Table B2). In particular, 391/1,963 cannot be relabelled as a 19.92% rate among all 2,000 masks. Among all 2,800 uniform half-retention evaluations, frozen verified conditional precision is undefined on 126; 37 of these occur among the 2,000 eligible masks.

**Table B2. Complete eligibility denominator under nominal 50% uniform retention, logistic versus historical rate.** Each row sums to 2,000 masks, including undefined estimates.

| Diagnostic | Reversal | Tie | Undefined | Same-sign remainder |
|---|---:|---:|---:|---:|
| Original-count reranking | 835 | 256 | 0 | 909 |
| Original-fraction reranking | 505 | 285 | 0 | 1,210 |
| Frozen verified conditional | 391 | 133 | 37 | 1,439 |
| Uniform HT | 515 | 183 | 0 | 1,302 |
| Zero filling | 515 | 183 | 0 | 1,302 |

Across the complete uniform half-retention design, logistic HT has mean signed error −0.00214 and MAE 0.12658. The corresponding original-count, original-fraction and zero-filled MAEs are 0.07073, 0.06746 and 0.06869. HT is retained despite its higher observed MAE. Finite-sample variance and small original selection budgets can make an unbiased estimator noisy. These results do not establish a universal ranking of evaluation procedures, particularly because the conditional diagnostics describe quantities different from the frozen target.

## B5. Magnitude strata and the amount of distinct date information

Before generating revision results, absolute complete differences were assigned to fixed left-closed, right-open bins: [0,0.02), [0.02,0.05), [0.05,0.10), [0.10,0.20), and [0.20,∞). The binning uses the frozen complete difference, not a masked or model-selected estimate. No cutpoint was changed in response to the results. Error, reversal and tie summaries retain unique decision-date counts and effective validity denominators for each bin.

**Table B3. Unique decision dates in each predeclared magnitude bin.** Counts describe distinct decisions, not independent events; different dates can reuse the same reference outcomes.

| Comparison | [0,0.02) | [0.02,0.05) | [0.05,0.10) | [0.10,0.20) | [0.20,∞) |
|---|---:|---:|---:|---:|---:|
| Logistic − historical rate | 4 | 5 | 3 | 2 | 0 |
| Boosting − historical rate | 2 | 1 | 3 | 3 | 5 |
| Historical rate − random | 1 | 0 | 0 | 4 | 9 |

The formal |Δ|≥0.02 reversal statistic is not defined for the first bin. The full outputs separately retain descriptive strict reversals among nonzero complete targets, without using them as replacements for the prespecified thresholded statistic. Empty bins remain empty. Mask repetitions do not multiply the number of distinct dates supporting a bin.

The bins should not be used to claim a monotonic error–magnitude relationship. For example, original-count logistic reversal rates under uniform half retention are 40.00%, 39.00% and 50.25% in its three nonempty eligible bins. The final value is supported by only two decision dates. Differences across policy pairs, small daily populations and repeated references preclude interpreting that pattern as an independently estimated general function of effect size.

## B6. Consistent masks for shared references, scenes and acquisition days

The revision adds three structural sensitivity analyses at nominal 50% retention, each with 200 repetitions and the same three policy pairs. A global mask is generated over the primary activity and projected onto all decisions. The seed sequence is `[20261006, 900, group mechanism index, repetition]`, with order unique reference, scene, acquisition day.

The unique-reference identifier is **campaign–source–scene**. A scene alone is not an outcome identifier: two sources in one scene can have different labels. All occurrences of one reference key receive the same mask across decisions. The inputs were checked for conflicting labels or acquisition times within these keys; no conflicts were found. Scene groups use campaign–scene, and acquisition-day groups use campaign–UTC acquisition date. For each mechanism, exactly (\lceil0.5G\rceil) of its (G) distinct groups are sampled uniformly without replacement. Group selection does not inspect outcomes.

Equal retained group counts do not imply equal retained row counts. Table B4 reports realized retention over all 1,468 repeated rows within each global replicate; its mean and range are calculated over the 200 global masks. The counts of empty and fully observed decision-masks refer to 14 decisions × 200 repetitions, before multiplication by policy pairs.

**Table B4. Group sizes, realized row retention and all-or-none decision cases.**

| Group mechanism | Total/retained groups | Mean global row retention | Range across global masks | Empty decision-masks | Fully observed decision-masks |
|---|---:|---:|---|---:|---:|
| Unique reference outcome | 632/316 | 0.499837 | 0.482970–0.521798 | 6/2,800 | 4/2,800 |
| Reference scene | 117/59 | 0.505245 | 0.416894–0.607629 | 119/2,800 | 105/2,800 |
| Acquisition day | 10/5 | 0.483648 | 0.225477–0.758856 | 816/2,800 | 853/2,800 |

These global row-weighted retention figures differ from equal-date retention averages, which are 0.499012, 0.500332 and 0.504696 for the three mechanisms, respectively. Both quantities are saved. At the individual decision level, all three grouped mechanisms can produce retention from zero to one; group counts and actual fractions accompany each date–mask record.

Only ten acquisition days contribute to this population, and five of the fourteen decisions refer to a single acquisition day. Day-group masking can consequently reveal every label for a decision or hide them all. Fully observed decisions have zero error for the non-HT point contrasts; empty decisions have undefined reranking and frozen conditional contrasts. They are retained as explicit states. An apparently small conditional error after excluding empty cases must not be interpreted as evidence that grouped missingness is easier to handle or that the method performs better under realistic missingness.

The structural masks preserve different kinds of dependence but are not estimates of actual flight selection, weather loss or publication processes. They complement, rather than replace, the original row-level masks. No transfer from the originally known population to originally unobserved locations is asserted.

## B7. Complete reporting, software checks and computation

The frozen matrix contains 25,200 decision–mask combinations and 75,600 policy-pair comparisons. Five point-method records per comparison yield 378,000 rows, including explicit inapplicable HT entries for nonuniform mechanisms. A separate table contains 75,600 paired completion intervals. The 15 synthetic tests cover HT unbiasedness by enumeration, actual rather than nominal inclusion probability, exclusion of invalid weighted-design HT, policy-specific verified denominators, undefined zero masses, selection-fraction budgets, shared-reference identity and consistency, empty grouped masks, interval containment and validity-aware summary denominators. Synthetic fixtures are used for software tests; the masking experiment operates on real observed reference records.

The full revision run completed in 51.65 s wall time and 51.41 s process CPU time, with a Windows process peak working set of approximately 826 MiB. This memory measurement includes Python imports. There were no paired containment failures. Timing excludes input acquisition, original model fitting, manuscript preparation and human review. No GPU, new model training or test-set tuning was used.

All mechanisms, retentions, policy pairs and five diagnostics remain available, including poor HT absolute-error results, structural ties at 25% original-count retention, undefined frozen conditional estimates, empty groups and sparsely populated magnitude bins. Comparisons were not selected for large reversal rates or favourable method rankings. These are documented revision-stage exploratory analyses, not preregistered or new external validation.

The reproducibility outputs provide individual estimates, eligibility and validity flags, two summary populations, per-date summaries, magnitude-bin summaries, paired intervals, group membership, global mask hashes, realized retention and the legacy reproduction check. Plot-ready tables retain their explicitly stated summary population; figures using common-valid support should filter the complete summary tables to `common_point_defined`. Full-design HT should use `method_defined` under uniform masking. No confidence intervals or field-population significance tests are based on the repeated artificial masks.

## Reference

Horvitz, D. G., and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47(260), 663–685. [https://doi.org/10.1080/01621459.1952.10483446](https://doi.org/10.1080/01621459.1952.10483446). Bibliographic metadata were verified against the publisher record and the December 1952 JSTOR issue; the publisher's later online-posting date does not replace the original publication year.

## C. Joint reference grouping and verification targets

Joint grouping uses the key (campaign, fixed source, actual qualified reference scene). In the 2024 primary replay there are 3,084 unique reference-or-score-slot groups, including 2,452 unknown groups: 2,451 decision-specific no-scene slots and one ambiguous actual reference reused by two decisions. The main comparison is computed on all 14 dates. For the controlled unique-reference experiment, all 1,468 known source–decision records are retained, representing 632 actual reference outcomes; 475 appear on more than one decision. A global mask hides 316 of those outcomes, with the same retention indicator at every appearance. No-scene slots are absent from this recoverable population.

Group coefficients are sums of the daily normalized policy differences divided by fourteen, rather than averages over the number of appearances of the group. Known repeated labels are accounted for by their full coefficient. One recovered outcome is charged once, irrespective of its number of appearances. Source–scene keys, rather than labels, break equal-coefficient ties. Reference grouping can reduce macro completion width only if coefficients for a shared unknown actual reference have opposite signs; the true primary campaign supplies no additional joint narrowing to numerical precision.

**Table C1. Verification demand at fixed width targets.** Entries are mean first integer revelation counts over 200 consistent half-reference masks. Uniform entries average the per-mask first integer crossing of that mask's expected-width curve. They are neither the crossing of the pooled mean curve nor random sequential stopping times.

| Pair | Width target | Targeted mean | Uniform expected-width crossing mean |
|---|---:|---:|---:|
| Logistic − history | 0.20 | 4.820 | 96.470 |
| Logistic − history | 0.10 | 18.325 | 206.420 |
| Logistic − history | 0.05 | 34.340 | 261.445 |
| Boosting − history | 0.20 | 37.975 | 203.460 |
| Boosting − history | 0.10 | 63.095 | 260.005 |
| Boosting − history | 0.05 | 84.330 | 288.260 |
| History − random | 0.20 | 79.235 | 205.960 |
| History − random | 0.10 | 143.260 | 261.255 |
| History − random | 0.05 | 196.275 | 288.855 |

The complete restricted-population macro differences are −0.0139586 (logistic–history), −0.1035799 (boosting–history) and 0.2466145 (history–random). The directional threshold is ±0.02. Logistic–history has zero eligible masks for certifying a direction beyond that band, not a directional success rate of zero. All 200 eligible boosting and history–random masks eventually certify their respective directions, with mean targeted demands 54.985 and 34.955. See `bounds/mask_threshold_summary.csv` for dispersion and every threshold, and `bounds/mask_reveal_curve_summary.csv` for every integer revelation count. Per-day coefficients, reason-specific width and group membership are generated by the accompanying code; they do not require assigning outcomes to missing scenes.

## D. Supplementary figures

![Figure S1](figures/Figure_S1_overlap.svg)

**Figure S1. Policy overlap and comparison information.** Each point is one policy pair, decision date and declared budget in the primary externally scored 2024 replay. Overall overlap is shared inclusion mass divided by budget. Relative reduction is one minus paired width divided by independent-policy width. Points with zero independent width are undefined and excluded. The plot shows a finite algebraic diagnostic rather than an independently estimated relationship, and supplies no field confidence intervals. Source: `figures/source_data/Figure_S1_overlap.csv`.

![Figure S2](figures/Figure_S2_effect_magnitude.svg)

**Figure S2. Masking error by complete-label contrast magnitude.** Panels show uniform half-retention date–mask summaries for each policy pair, using fixed effect bins. Each method is summarized over its own defined estimates; HT uses the full uniform design. Counts beneath bins denote distinct decision dates, not independent events. Empty bins are retained without fabricated values, and missing bin values are not connected. Conditional-method validity counts and common-valid errors are reported separately in section B and the numerical outputs. The y-axis includes the 0.375 HT error in the smallest boosting–history bin. Source: `figures/source_data/Figure_S2_effect_magnitude.csv`.

![Figure S3](figures/Figure_S3_grid_sensitivity.svg)

**Figure S3. Spatial resampling grid sensitivity.** Points and horizontal bars show conditional precision differences and 95% percentile intervals from 1,000 fixed-model replicates for each of six grid scale/origin combinations. Row labels give side length in kilometres and origin displacement as a fraction of one cell in both coordinate axes. Circles represent the unshifted grid and squares the half-cell shift. Each block carries its sources and repeated dates; budgets and ties are recomputed. These are conditional resampling intervals, not binary completion ranges or model-refitting uncertainty. Source: `figures/source_data/Figure_S3_grid_sensitivity.csv`.



## E. Generic software and scenario validation

The generic package is VerifyRank 0.2.0, separate from the earlier campaign-analysis release. Its core accepts frozen policy weights and a reference table; it does not call the methane scorer or infer a missing physical event. The Python API, command-line entry point, precise schema and installation tutorial are distributed with the software. Tests check specified computational properties; they are not field replications or independent physical validation.

### E1. Cost, recoverability and label-error assumptions

Verification costs are relative planning units. Equal unit costs and deterministic costs from 1 to 5 are declared scenarios, not observed annotation, flight or repair costs. Recovery probability means that an attempted check returns a usable qualified binary reference. It is not a label-accuracy estimate; correctness is assumed separately in the primary calculation. Probabilities of 1, 0.8 and 0.5, and the heterogeneous-probability scenario, are assumptions used to inspect sensitivity. Failed attempts remain unknown. A lower expected width does not guarantee that a particular realized set of checks will reach a target.

The planner sorts absolute coefficient divided by cost, optionally multiplied by recovery probability. It accepts a task only when its whole cost fits the remaining budget. For indivisible tasks this rule is not generally optimal; the synthetic counterexample and exhaustive small-budget checks make that limitation explicit. Irrecoverable references contribute to a retained floor. A real absent follow-up is not a candidate for archival resolution.

An additional sensitivity allows at most a specified number of supplied unique-reference labels to be wrong. For supplied value z and coefficient C, a flip changes the known contribution by C(1−2z). Summing up to r most negative changes and, separately, up to r most positive changes gives an exact error-count extension of the completion range, while unknown-reference terms remain unchanged. The constraint counts unique outcomes, not repeated appearances. This is a declared bounded-error scenario, not an estimated label-error rate; it is separate from recovery probability. Exact small enumerations verify the implemented range.

An equivalence stopping rule is computed after each returned label by asking whether the entire current range lies within ±0.02. The hidden complete contrast is never used for ordering or stopping. Directional superiority, within-tolerance comparison, requested width and unattainable floor are distinct report fields.

### E2. Internal reconstruction and external validity

The existing campaign reconstruction, fresh refitting and 99-test campaign suite remain the evidence described in the first revision. The new package adds an independent generic interface and a clean-install command-line test, with parity checks against the preserved full-inventory group coefficients. Its Linux GitHub Actions workflow has also completed successfully. Docker execution remains untested. These are internal computational reconstructions and software checks. No third-party reproduction, user-adoption study, operational deployment or cross-domain empirical validation has been carried out. Small synthetic datasets test structural reuse and exact arithmetic under the stated schema.

### E3. Graphics provenance

The plotting code reuses the GitHub SciencePlots package (2.2.2; science/nature/no-latex styles), the reviewed K-Dense scientific-visualization skill and Nature's official artwork guidance. The main figures are redesigned around the article's three questions. Native PDF and SVG exports contain vector marks and text; 1200 dpi RGB TIFF companions are rendered directly at the final size. Vectors have no intrinsic dpi. The graphics manifest records dimensions, raster resolution, text bounding-box collisions, clipping and line intersections. Figure data are exported as CSV. Preserved supplementary graphics retain the same source data as the validated first revision.

### E4. Release and author information

VerifyRank version 0.2.0 is publicly available at [VerifyRank v0.2.0](https://github.com/williamjay1/verifyrank/releases/tag/v0.2.0). The generic code, documentation and synthetic fixtures have an MIT licence; raw provider records and the mixed-year derived panel are excluded. The published software commit is [8d15fe30c4ffd961bccd8eb2ba1c89395d0aad13](https://github.com/williamjay1/verifyrank/commit/8d15fe30c4ffd961bccd8eb2ba1c89395d0aad13), and its [Linux GitHub Actions run](https://github.com/williamjay1/verifyrank/actions/runs/37425443624) completed successfully. The container configuration has not been executed with Docker.

The methane case materials, version 2026.10.06, are publicly available separately at [verifyrank-methane-case v2026.10.06](https://github.com/williamjay1/verifyrank-methane-case/releases/tag/v2026.10.06), commit [c0eaa7abc3efa5e845e192ddca83dac65fd5ed7a](https://github.com/williamjay1/verifyrank-methane-case/commit/c0eaa7abc3efa5e845e192ddca83dac65fd5ed7a). Their data-terms file preserves Carbon Mapper attribution, applicable noncommercial and redistribution conditions, and the stated CC BY 4.0 licence of the 2024 source archive. The software MIT licence does not relicense these data. New Zenodo deposits remain pending the author's action; [10.5281/zenodo.17862304](https://doi.org/10.5281/zenodo.17862304) identifies the provider's existing archive, not either new project release.

junjiezhang is the sole author and human contributor. OpenAI Codex provided the substantive assistance disclosed in the main article; no additional human contributors or human scientific sign-off are claimed. The author remains responsible for the submitted work. Affiliations, corresponding-author details, funding, conflicts and submission approval have not been invented; the applicable declarations and journal-specific requirements remain for the author to complete at submission.

### E5. Executed scenario results

## Cost-aware verification and label-error sensitivity

These analyses were added during the second revision and are exploratory, not preregistered. They reuse the primary 2024 external-transfer comparison and the originally known reference population: 1,468 source-decision rows, 14 decision dates and 632 unique campaign/source/actual-scene outcomes. Before masking, each date's policy inclusion weights were fixed at a budget of ceil(0.25 × originally known candidates); dates receive equal macro weights. A reference's coefficient is the sum of its date-weighted policy differences over every genuine reuse of that outcome. All policy scores remain fixed.

### Reproducible scenarios and planning rules

Reference IDs are the prefix `reference:` followed by compact JSON encoding of the campaign, source ID and actual scene ID, all strings. IDs are sorted lexicographically. NumPy's `default_rng(20261006)` selects exactly 316 of the 632 IDs without replacement as retained in each repetition; the other 316 are hidden. The same 200 masks are used across the three policy pairs, both cost scenarios and every recovery scenario, matching the earlier unique-reference masking experiment. Masking is independent of reference labels.

Equal costs assign one relative unit to every reference. For bounded variable costs, let `h` be the hexadecimal SHA256 digest of the UTF-8 reference ID. The deterministic assignment is `1 + int(h[0:8], 16) % 5`, yielding costs from 1 to 5 relative units. These values are hypothetical, not estimated monetary, aircraft or annotation costs. Uniform recovery scenarios assign probability 1, 0.8 or 0.5 to every eligible reference. An additional heterogeneous sensitivity assigns `[1, 0.8, 0.5][int(h[8:16], 16) % 3]` under the same variable costs. These probability assignments are independent of labels, are not estimated from the archive, and concern recovery of a usable qualified result rather than label accuracy. The primary calculation separately treats a returned binary label as correct.

The two scores are absolute aggregate coefficient divided by cost, and absolute coefficient multiplied by recovery probability divided by cost. Ties are broken by reference ID. Uniform probabilities multiply all scores by the same constant, so only the first ordering needs to be evaluated in those scenarios. Both orderings are retained in the heterogeneous sensitivity. The planner scans that order and accepts an indivisible reference only when its complete cost fits the remaining budget. It never reads a hidden label and is not a knapsack optimizer. Figure 4 evaluates this actual rule independently at every integer cost budget from 0 to 400; selected sets need not be nested. The compact summary also evaluates cost budgets equal to the floor of 0%, 10%, 25%, 50%, 75% or 100% of the total hidden-reference cost. Given a selected set, hypothetical expected residual width subtracts the sum of probability × absolute coefficient from the initial width. It is not an achieved completion interval. Failed, uncertain or nonbinary returns remain unknown.

Sequential stopping is a separate diagnostic: follow the complete label-free priority order, return one qualified label at a time, and recompute the entire interval after each return. The first whole-interval entry into [−0.02, 0.02] defines practical-equivalence stopping. Positive and negative assessments require the lower endpoint to exceed +0.02 or the upper endpoint to fall below −0.02, respectively. At a prefix's exact cumulative cost, the budget-feasible API selects that same prefix with no remaining budget; nevertheless, its crossing cost need not be the lowest successful fixed budget or the knapsack optimum. The stored complete contrast is used for containment checks, never to choose requests or stop before labels return. Mean crossing counts or costs refer to the reached repetitions, with their denominators retained. The width-target summaries concern first crossings of 0.10 along this ordered prefix; under incomplete recovery they describe an expected-width curve, not an expected random stopping cost.

### Wrong-label sensitivity

The error budget permits at most r currently supplied unique-reference labels to be wrong, with r fixed to 0, 1 or 3. It covers both labels retained at the start and subsequently returned labels. With an observed value z and coefficient C, flipping that reference changes the contrast by C(1−2z). Starting from the nominal completion interval, its lower endpoint adds the r most negative available changes, while its upper endpoint adds the r largest positive changes; fewer than r changes are used if the relevant sign has fewer candidates. Unreturned references retain unrestricted binary completion. The resulting endpoints are exact under this specified finite error-count constraint, as checked by exhaustive small-case enumeration. No error rate is estimated, and the calculation does not physically validate catalogue detections or nondetections.

**Table S-P1. Logistic–history practical-equivalence stopping under ideal successful recovery.** Full-return intervals are common to all masks and both cost orders, apart from numerical rounding. Means refer to the first interval entirely within ±0.02; outcomes are unique references, and costs are relative units. Every scenario contains 200 masks.

| Maximum wrong supplied labels | Full-return interval | Equivalence reached | Mean returns, equal costs | Mean returns, variable costs | Mean variable cost |
|---:|---:|---:|---:|---:|---:|
| 0 | [−0.013959, −0.013959] | 200/200 | 55.125 | 55.640 | 162.385 |
| 1 | [−0.037768, 0.033660] | 0/200 | Not reached | Not reached | Not reached |
| 3 | [−0.073482, 0.081279] | 0/200 | Not reached | Not reached | Not reached |

The practical-equivalence finding therefore depends on treating the supplied catalogue labels as correct. Even one possible wrong unique label prevents this conclusion after all hidden references have been returned. In the first controlled mask, the budget-feasible variable-cost plan at cost 100 selects 36 references and yields [−0.036785, 0.018531] after ideal returns: narrower than 0.10, but not wholly within ±0.02. Along the prescribed order, the first equivalence interval is [−0.019511, 0.003577] after 52 returned references at relative cost 160.

**Table S-P2. Other policy pairs under the same wrong-label sensitivities.** Counts and costs use the variable-cost sequential order. A negative or positive assessment means separation beyond the declared ±0.02 tolerance, not merely exclusion of zero. Full-return intervals are independent of the masking order.

| Contrast | Maximum wrong supplied labels | Full-return interval | Assessment at ±0.02 | Classified masks | Mean returns to classification | Mean relative cost |
|---|---:|---:|---|---:|---:|---:|
| Boosting − history | 0 | [−0.103580, −0.103580] | Negative | 200/200 | 61.745 | 158.400 |
| Boosting − history | 1 | [−0.127389, −0.055961] | Negative | 200/200 | 87.020 | 239.780 |
| Boosting − history | 3 | [−0.175009, −0.008749] | Unresolved at tolerance | 0/200 | Not reached | Not reached |
| History − random | 0 | [0.246615, 0.246615] | Positive | 200/200 | 41.585 | 77.540 |
| History − random | 1 | [0.216853, 0.264472] | Positive | 200/200 | 58.705 | 116.595 |
| History − random | 3 | [0.157329, 0.300186] | Positive | 200/200 | 92.730 | 201.855 |

The actual full-candidate archive has a different information limit. Only one ambiguous unique actual reference is potentially recoverable; absent-followup slots are not historical audit requests. For logistic–history, ideal recovery of that ambiguous result can remove at most 0.001532 of width, leaving a floor of 0.315348. A 0.10 target cannot be met by recovering existing references in that archive. A revised monitoring design can collect new evidence but cannot retroactively supply a nonexistent historical observation.

### Computational verification

The reusable planning component uses the Python standard library. The analysis script reuses the existing evaluator and requires its numerical dependencies. The final 200-mask run took 26.112 seconds of wall time and 25.875 seconds of process CPU time on the executing environment. It performed no model refitting or new data download. Nine planning unit tests passed; output checks covered 28,800 fixed-budget curve records, 2,406 figure records and 3,600 stopping records. The vectorized figure calculation was compared with the public API at five budgets in every applicable mask/scenario. An additional 420 path comparisons matched the exact public error-bound function. Structural synthetic examples establish binary sharpness, cancellation, separation of absent-followup slots and a greedy counterexample; they are not a second environmental case study or independent field validation.


### E6. Software validation record

The supplied software validation report records the exact tests, actual clean-install command-line run and methane-adapter parity checks. Details and output values are included in the accompanying artifact rather than represented as an external reproducibility study.

## F. Additional benchmark sensitivities

#### Sensitivity and executable reproducibility

All six spatial grid scale/origin combinations preserve the primary qualitative conditional conclusions. History–random intervals exclude zero positively, boosting–history intervals exclude it negatively, and logistic–history intervals include it. Across grids, logistic lower endpoints range from −0.0249 to −0.0225 and upper endpoints from 0.0071 to 0.0102 (Figure S3). Leaving out one decision date gives logistic–history conditional estimates from −0.0133 to −0.0063; history–random ranges from 0.0863 to 0.1035 and boosting–history from −0.0511 to −0.0422. This checks grid and date dependence without claiming that fourteen dates represent extensive independent campaign replication.

Delay and endpoint sensitivities remain scientifically material. At 0, 24 and 48 h assumed availability, the 2023 training counts are 585, 415 and 278; logistic–history conditional differences are 0.0016, −0.0102 and −0.0688. All three three-day full-population completion intervals include zero. The seven-day endpoint leaves only 46 eligible training records from one 2023 decision date; logistic–history completion bounds [−0.2842, −0.0073] and boosting–history [−0.3500, −0.0056] are negative. These results are retained, with their changed endpoint and weak training support, rather than used to claim general learner inferiority. Association radii of 100 and 500 m leave logistic–history conditional differences −0.0004 and 0.0048 and completion intervals spanning zero. Strict-quality external learning remains unavailable because 2023 lacks comparable quality fields.

Within-campaign learned fits cover only two of five 2023 dates and eight of fourteen 2024 dates. On the eight common primary 2024 dates, logistic conditional precision is 0.5511 versus history's 0.5507, and boosting–history difference is −0.0321. Strict 2024 common-date differences are −0.0039 and −0.0212 with completion intervals crossing zero. Region-specific completion bounds likewise cross zero in East, West and the full-domain residual. The nine-anchor outside-polygon stratum has a negative logistic–history bound [−0.1528, −0.0615], retained as a small exploratory result rather than a generalization claim. Supplementary methods preserve every region's candidate, known-outcome and date denominator and the separately defined regional budget.

The independent metadata reconstruction matches all 496 accepted and 212 strict anchors. No discrepancy is found in the 80 stratified reference records or nine radius-boundary cases. Raw-input rebuilding verifies 291 original 2023 response files and eleven 2024 archived files, then reproduces all sixteen CSV outputs byte for byte, including the combined 27,521-row scenario panel. The timed raw rebuild takes 65.38 s and the builder's measured peak working set is 151.34 MiB. These are software and metadata checks, without claiming independent physical validation of nondetections. Fresh-environment refitting and all-scenario evaluation reproduce the original numerical outputs. The profiled evaluation takes 33.43 s wall time (55.44 s summed process CPU time) with peak working set 2,079.36 MiB, including construction of the full selection and unresolved-priority exports; its locked environment is supplied with the software package. The expanded masking calculation takes 51.65 s with peak working set 825.9 MiB; the grouped completion, verification and grid sensitivity stage takes 16.08 s. Such computation is inexpensive on a prepared catalogue. Acquisition, operational access and scientific review costs are outside these timings.


