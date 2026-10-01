import pytest
from conftest import load

import ergmx
from ergmx import datasets


@pytest.mark.parametrize("name", ["flomarriage", "samplk3", "faux.mesa.high", "faux.dixon.high"])
def test_bundled_networks_are_the_test_networks(name):
    formula = "edges + nodecov('wealth')" if name == "flomarriage" else "edges + gwesp(0.5, fixed=TRUE)"
    expected = ergmx.summary_stats(load(name), formula)
    assert ergmx.summary_stats(datasets.load(name), formula) == expected
    assert ergmx.summary_stats(datasets.load(name, backend="networkx"), formula) == expected


def test_every_dataset_loads():
    for name in datasets.names():
        g = datasets.load(name)
        assert g.vcount() > 0 and "name" in g.vs.attributes() and "id" not in g.vs.attributes()
        assert datasets.describe(name)


def test_dataset_errors():
    with pytest.raises(KeyError, match="no dataset 'florentine'"):
        datasets.load("florentine")
    with pytest.raises(ValueError, match="backend"):
        datasets.load("samplk3", backend="network")
