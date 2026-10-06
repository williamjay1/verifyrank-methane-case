"""Re-run the original evaluation through an outcome-free policy interface."""
import evaluate_campaigns
from revision_contract import score_campaigns
if __name__=='__main__':
    evaluate_campaigns.score_campaigns=score_campaigns
    raise SystemExit(evaluate_campaigns.main())
