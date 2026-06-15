from spinegen.naming import slugify, unique_name
from spinegen.tags import parse_tags, strip_tags


def test_slugify_has_ascii_fallback() -> None:
    assert slugify("头发 [bone]", fallback="layer") == "bone"
    assert slugify("!!!", fallback="layer") == "layer"


def test_unique_name_suffixes_duplicates() -> None:
    used = {"head"}
    assert unique_name("head", used) == "head_2"
    assert unique_name("head", used) == "head_3"


def test_parse_and_strip_tags() -> None:
    assert parse_tags("left arm [bone][skin:armor]") == {"bone": True, "skin": "armor"}
    assert strip_tags("left arm [bone][skin:armor]") == "left arm"

