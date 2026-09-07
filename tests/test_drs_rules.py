import pytest
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall, PitchingZone, ImpactZone, WicketsResult

def test_drs_impact_outside_off():
    """
    Test a trajectory where ball impact is 10 cm outside off-stump.
    Assert Impact: OUTSIDE OFF, Decision: NOT OUT.
    """
    engine = DRSEngine()
    # Stumps half width = 0.1143m, ball radius = 0.036m
    # 10 cm outside off for RHB => X = -(0.1143 + 0.10) = -0.2143m
    bounce_pt = (-0.05, 12.0, 0.0)
    pad_impact = (-0.2143, 18.8, 0.45)
    stump_impact = (-0.05, 20.12, 0.40)

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT
    )

    assert verdict.impact == ImpactZone.OUTSIDE_OFF
    assert verdict.final_verdict == "NOT OUT"

def test_drs_wickets_umpires_call_clipping():
    """
    Test a trajectory where ball clips outer edge of off-stump by 1.5 cm.
    Assert Wickets: UMPIRE'S CALL.
    """
    engine = DRSEngine()
    # Stump edge = 0.1143m. Ball center at 0.1143 + 0.015m (1.5 cm past outer edge)
    # Ball radius = 0.036m, so it overlaps the stump edge by 2.1 cm (> 50% margin)
    bounce_pt = (0.02, 11.5, 0.0)
    pad_impact = (0.05, 18.8, 0.45)
    stump_impact = (-0.1293, 20.12, 0.50)  # Clips off-stump outer margin

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.OUT
    )

    assert verdict.pitching == PitchingZone.IN_LINE
    assert verdict.impact == ImpactZone.IN_LINE
    assert verdict.wickets == WicketsResult.UMPIRES_CALL
    # With on-field call OUT, verdict stands as OUT
    assert verdict.final_verdict == "OUT"

def test_drs_pitching_outside_leg():
    """
    Test pitching outside leg stump for a right-handed batsman.
    Assert Pitching: OUTSIDE LEG, Decision: NOT OUT.
    """
    engine = DRSEngine()
    # For RHB, leg-side is X > +0.1143m
    bounce_pt = (0.18, 10.5, 0.0)  # Pitches outside leg
    pad_impact = (0.05, 18.8, 0.45)
    stump_impact = (0.0, 20.12, 0.35)  # Hitting middle stump

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.OUT
    )

    assert verdict.pitching == PitchingZone.OUTSIDE_LEG
    assert verdict.final_verdict == "NOT OUT"
    assert "Pitching outside leg" in verdict.reasons

def test_drs_three_reds_lbw_out():
    """
    Test classic Three Reds LBW: In-line pitching, In-line impact, and Hitting middle stump.
    """
    engine = DRSEngine()
    bounce_pt = (-0.03, 12.0, 0.0)
    pad_impact = (-0.02, 18.8, 0.45)
    stump_impact = (-0.01, 20.12, 0.42)

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT
    )

    assert verdict.pitching == PitchingZone.IN_LINE
    assert verdict.impact == ImpactZone.IN_LINE
    assert verdict.wickets == WicketsResult.HITTING
    assert verdict.final_verdict == "OUT"
