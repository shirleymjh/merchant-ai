from merchant_ai.services.planning_tooling import json_object_candidates, parse_json_object


def test_parse_json_object_reads_fenced_payload_without_pattern_matching() -> None:
    assert parse_json_object('```json\n{"status":"UNDERSTOOD","nested":{"ok":true}}\n```') == {
        "status": "UNDERSTOOD",
        "nested": {"ok": True},
    }


def test_parse_json_object_ignores_braces_inside_quoted_strings() -> None:
    payload = 'provider prefix {"message":"literal } and escaped \\"{\\"","value":3} trailing text'

    assert parse_json_object(payload) == {
        "message": 'literal } and escaped "{"',
        "value": 3,
    }


def test_json_object_candidates_fail_closed_for_unbalanced_protocol_text() -> None:
    assert json_object_candidates('prefix {"status":"UNDERSTOOD"') == []
    assert parse_json_object('```json\n{"status":"UNDERSTOOD"') == {}
