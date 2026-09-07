import pytest
from core.drs_engine import DRSEngine, BatsmanHand, OnFieldCall, PitchingZone, ImpactZone, WicketsResult

def test_drs_impact_outside_off_shot_offered():
    """
    Test a trajectory where ball impact is 10 cm outside off-stump with shot offered.
    Assert Impact: OUTSIDE OFF, Decision: NOT OUT.
    """
    engine = DRSEngine()
    bounce_pt = (-0.05, 12.0, 0.0)
    pad_impact = (-0.2143, 18.8, 0.45)
    stump_impact = (-0.05, 20.12, 0.40)

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.NOT_OUT,
        shot_offered=True
    )

    assert verdict.impact == ImpactZone.OUTSIDE_OFF
    assert verdict.final_verdict == "NOT OUT"

def test_drs_impact_outside_off_no_shot_offered():
    """
    Law 36.1(e): If NO shot is offered (padded away), impact outside off is OUT
    if ball goes on to hit the stumps!
    """
    engine = DRSEngine()
    bounce_pt = (-0.05, 12.0, 0.0)
    pad_impact = (-0.2143, 18.8, 0.45)
    stump_impact = (0.0, 20.12, 0.40)  # Hitting middle stump

    verdict = engine.evaluate_lbw(
        bounce_point=bounce_pt,
        pad_impact_point=pad_impact,
        predicted_stump_point=stump_impact,
        batsman_hand=BatsmanHand.RIGHT_HAND,
        on_field_call=OnFieldCall.OUT,
        shot_offered=False  # No shot played
    )

    assert verdict.impact == ImpactZone.OUTSIDE_OFF
    assert verdict.wickets == WicketsResult.HITTING
    assert verdict.final_verdict == "OUT"

def test_drs_wickets_umpires_call_clipping():
    """
    Test a trajectory where ball clips outer edge of off-stump by 1.5 cm.
    Assert Wickets: UMPIRE'S CALL.
    """
    engine = DRSEngine()
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
    assert verdict.final_verdict == "OUT"

def test_drs_pitching_outside_leg():
    """
    Test pitching outside leg stump for a right-handed batsman.
    Assert Pitching: OUTSIDE LEG, Decision: NOT OUT.
    """
    engine = DRSEngine()
    bounce_pt = (0.18, 10.5, 0.0)  # Pitches outside leg for RHB
    pad_impact = (0.05, 18.8, 0.45)
    stump_impact = (0.0, 20.12, 0.35)

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
