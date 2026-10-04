import pytest

from genai_lab.garment_topology import (
    evaluate_topology_diagnostic,
    resolve_garment_topology,
    topology_guidance,
)


def test_bikini_is_two_piece_with_exposed_abdomen():
    contract = resolve_garment_topology(("striped_bikini",))

    assert contract["status"] == "CONFIRMED"
    assert contract["topology"] == "two_piece"
    assert contract["component_presence"] == {
        "upper": "present", "lower": "present", "one_piece": "unknown",
    }
    assert contract["required_exposed_regions"] == ["abdomen"]
    assert contract["forbidden_connections"] == ["upper_to_lower"]
    assert "do not join" in topology_guidance(contract)
    assert "abdomen" in topology_guidance(contract)


def test_generic_swimsuit_stays_unresolved():
    contract = resolve_garment_topology(("swimsuit",))

    assert contract["status"] == "UNRESOLVED"
    assert contract["topology"] == "unresolved"
    assert contract["ambiguous_tags"] == ["swimsuit"]
    assert topology_guidance(contract) == ""


def test_missing_lower_component_does_not_become_absent():
    contract = resolve_garment_topology(("bikini_top",))

    assert contract["topology"] == "upper_only"
    assert contract["component_presence"]["upper"] == "present"
    assert contract["component_presence"]["lower"] == "unknown"


def test_reviewed_two_piece_requires_connection_rule():
    with pytest.raises(ValueError, match="연결 금지"):
        resolve_garment_topology((), reviewed_contract={
            "topology": "two_piece",
            "components": ["upper", "lower"],
            "required_exposed_regions": ["abdomen"],
            "forbidden_connections": [],
        })


def test_topology_diagnostic_never_enforces_retry():
    contract = resolve_garment_topology(("bikini",))
    diagnostic = evaluate_topology_diagnostic(
        contract, {"bodysuit": .81, "bikini": .10}
    )

    assert diagnostic["status"] == "DIAGNOSTIC"
    assert diagnostic["reason"] == "possible_topology_mismatch"
    assert diagnostic["enforced"] is False


def test_generic_output_evidence_is_unresolved_not_failure():
    contract = resolve_garment_topology(("bikini",))
    diagnostic = evaluate_topology_diagnostic(contract, {"swimsuit": .72})

    assert diagnostic["status"] == "UNRESOLVED"
    assert diagnostic["reason"] == "component_evidence_incomplete"
    assert diagnostic["component_checks"]["upper"]["status"] == "UNRESOLVED"
    assert diagnostic["component_checks"]["lower"]["status"] == "UNRESOLVED"
    assert diagnostic["absence_is_not_failure"] is True


def test_upper_and_lower_are_reported_separately():
    contract = resolve_garment_topology(("bikini",))
    diagnostic = evaluate_topology_diagnostic(
        contract, {"bikini_top": .78, "bikini_bottom": .69, "bikini": .64}
    )

    assert diagnostic["status"] == "PASS"
    assert diagnostic["component_checks"]["upper"]["score"] == .78
    assert diagnostic["component_checks"]["lower"]["score"] == .69


def test_single_component_topology_is_not_forced_through_connection_check():
    contract = resolve_garment_topology(("shirt",))
    diagnostic = evaluate_topology_diagnostic(contract, {"shirt": .80})

    assert diagnostic["status"] == "NOT_EVALUATED"
    assert diagnostic["reason"] == "component_topology_has_no_connection_rule"
    assert diagnostic["enforced"] is False
