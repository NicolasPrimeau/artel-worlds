from automata.llm import parse_note, parse_genome


def test_parse_note_extracts_the_tribe_voice_line():
    text = '{"note": "fleeing the toxic core, pushing west to fresh ground", "regulators": {}, "behaviors": []}'
    assert parse_note(text) == "fleeing the toxic core, pushing west to fresh ground"


def test_parse_note_is_blank_when_absent_or_unparseable():
    assert parse_note('{"regulators": {}, "behaviors": []}') == ""
    assert parse_note("not json at all") == ""


def test_note_does_not_break_genome_parsing():
    text = '{"note": "grow now", "regulators": {}, "behaviors": [{"cond1": {"variable": "my_energy", "op": ">", "threshold": 10}, "verb": "divide", "target": "random"}]}'
    g = parse_genome(text, 8)
    assert g is not None and len(g.behaviors) >= 1
