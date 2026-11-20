from src.eval.y5_diagnostics import by_source, diagnostics


def test_diagnostics_counts_bias_and_inherited_errors():
    preds = [{"id": "s1", "pred_label": "supported"}, {"id": "s1_B", "pred_label": "contradicted"},
             {"id": "s1_A", "pred_label": "contradicted"}, {"id": "s1_B_A", "pred_label": "contradicted"},
             {"id": "s2", "pred_label": "contradicted"}, {"id": "s2_A", "pred_label": "contradicted"},
             {"id": "s2_B_A", "pred_label": "contradicted"}]
    gold = {"s1": {"orig": "supported", "B": "contradicted", "A": "supported", "B_A": "contradicted"},
            "s2": {"orig": "supported", "A": "supported", "B_A": "contradicted"}}
    en, ar = diagnostics(by_source(preds), gold)
    assert en["supported_recall"] == 0.5 and en["paired_esr"] == 1.0 and en["paired_n"] == 1
    assert ar["supported_recall"] == 0.0 and ar["paired_n"] == 0
    # two wrong _A verdicts: s1 English right (Arabic-only), s2 English wrong (inherited)
    assert ar["errors_n"] == 2 and ar["errors_inherited"] == 0.5
