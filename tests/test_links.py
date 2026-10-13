from gspaenli.links import strip_unsourced_links

DATA = '{"source_url": "https://www.priminfo.admin.ch/de/praemien", "map": "https://kvgmapper26.github.io/maps/a.html"}'


def test_links_from_the_data_are_kept():
    text = "See https://www.priminfo.admin.ch/de/praemien. Or [the map](https://kvgmapper26.github.io/maps/a.html)."
    assert strip_unsourced_links(text, DATA) == (text, [])


def test_invented_links_are_removed():
    # Regression: Apertus answered an honest "not found" but added www.bern.ch/de/abfall.
    text = "Schauen Sie unter www.bern.ch/de/abfall nach. Oder [hier](https://example.ch/x)."
    cleaned, removed = strip_unsourced_links(text, "")
    assert "bern.ch" not in cleaned and "example.ch" not in cleaned
    assert "hier" in cleaned
    assert removed == ["https://example.ch/x", "www.bern.ch/de/abfall"]


def test_bare_link_label_is_dropped_with_its_link():
    cleaned, _ = strip_unsourced_links("Source: [https://x.ch/y](https://x.ch/y)", "")
    assert "x.ch" not in cleaned
