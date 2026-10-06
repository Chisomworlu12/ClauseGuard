from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    clause_text: str
    expected_risk_level: str
    expected_category: str
    expected_reference_ids: tuple[str, ...]
    ambiguous: bool = False


EVAL_CASES = [
    EvalCase("termination-001", "Either party may terminate this agreement at any time without cause or notice.", "high", "termination", ("termination-redflag-03",)),
    EvalCase("termination-002", "Either party may terminate for material breach after thirty days' written notice and an opportunity to cure.", "low", "termination", ("termination-norm-01", "termination-norm-03")),
    EvalCase("termination-003", "The customer may terminate for convenience on ninety days' written notice.", "medium", "termination", ("termination-norm-01",)),
    EvalCase("termination-004", "This agreement lasts for twelve months and contains no renewal or termination language.", "unable_to_assess", "other", (), True),
    EvalCase("liability-001", "Each party's liability is limited to the fees paid under this agreement during the prior twelve months.", "low", "liability", ("liability-norm-01",)),
    EvalCase("liability-002", "The provider is liable for all losses of every kind, including unlimited consequential damages.", "high", "liability", ("liability-redflag-01", "liability-redflag-03")),
    EvalCase("liability-003", "Neither party will be liable for indirect or consequential damages, except for fraud or willful misconduct.", "low", "liability", ("liability-norm-02",)),
    EvalCase("liability-004", "The parties agree that liability will be allocated as commercially reasonable under applicable law.", "unable_to_assess", "liability", (), True),
    EvalCase("auto-renewal-001", "This agreement automatically renews for successive one-year terms unless either party gives sixty days' notice.", "low", "auto_renewal", ("auto-renewal-norm-01", "auto-renewal-norm-03")),
    EvalCase("auto-renewal-002", "The contract renews automatically for five years and may not be cancelled during the renewal period.", "high", "auto_renewal", ("auto-renewal-redflag-01", "auto-renewal-redflag-04")),
    EvalCase("auto-renewal-003", "Renewal requires written agreement signed by both parties at least thirty days before expiry.", "low", "auto_renewal", ("auto-renewal-norm-02",)),
    EvalCase("auto-renewal-004", "The term is twelve months and the parties may discuss renewal if business conditions permit.", "unable_to_assess", "auto_renewal", (), True),
    EvalCase("payment-001", "Client shall pay each undisputed invoice within thirty days of receipt.", "low", "payment_terms", ("payment_terms-norm-01", "payment_terms-norm-02")),
    EvalCase("payment-002", "Client may delay payment indefinitely until it is satisfied with the services.", "high", "payment_terms", ("payment_terms-redflag-01",)),
    EvalCase("payment-003", "Client shall pay Provider within sixty days after receiving an invoice.", "medium", "payment_terms", ("payment_terms-norm-01", "payment_terms-norm-02")),
    EvalCase("payment-004", "Payment will be made promptly in accordance with the parties' agreed commercial practice.", "unable_to_assess", "payment_terms", (), True),
    EvalCase("ip-assignment-001", "Each party retains ownership of its pre-existing intellectual property.", "low", "ip_assignment", ("ip_assignment-norm-01",)),
    EvalCase("ip-assignment-002", "All intellectual property created by the provider, including its pre-existing tools, belongs exclusively to the client.", "high", "ip_assignment", ("ip_assignment-redflag-01", "ip_assignment-redflag-02")),
    EvalCase("ip-assignment-003", "The provider grants the client a perpetual, worldwide, non-exclusive licence to use the deliverables.", "medium", "ip_assignment", ("ip_assignment-norm-02",)),
    EvalCase("ip-assignment-004", "Ownership of inventions and materials will be determined later by the parties in writing.", "unable_to_assess", "ip_assignment", (), True),
    EvalCase("non-compete-001", "For six months after termination, the consultant will not provide competing services within the city of Lagos.", "medium", "non_compete", ("non-compete-norm-01",)),
    EvalCase("non-compete-002", "The employee may not work for any business anywhere in the world for five years after leaving.", "high", "non_compete", ("non-compete-redflag-01", "non-compete-redflag-03")),
    EvalCase("non-compete-003", "The restriction applies only to direct competitors and only to services the employee performed for the company.", "low", "non_compete", ("non-compete-norm-02", "non-compete-norm-03")),
    EvalCase("non-compete-004", "The parties will agree on reasonable post-termination restrictions when the engagement ends.", "unable_to_assess", "non_compete", (), True),
    EvalCase("indemnification-001", "The supplier will indemnify the customer for third-party claims caused by the supplier's negligence.", "low", "indemnification", ("indemnification-norm-01",)),
    EvalCase("indemnification-002", "The supplier must indemnify the customer against every claim, loss, or expense arising from any event whatsoever.", "high", "indemnification", ("indemnification-redflag-01", "indemnification-redflag-03")),
    EvalCase("indemnification-003", "An indemnified party must promptly notify the indemnifying party and allow it to control the defence.", "low", "indemnification", ("indemnification-norm-02",)),
    EvalCase("indemnification-004", "The parties will share responsibility for claims in a manner they later determine in good faith.", "unable_to_assess", "indemnification", (), True),
]