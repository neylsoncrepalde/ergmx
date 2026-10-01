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
        loaded = datasets.load(name)
        for g in loaded if isinstance(loaded, list) else [loaded]:
            assert g.vcount() > 0 and "name" in g.vs.attributes() and "id" not in g.vs.attributes()
        assert datasets.describe(name)


def test_households_are_ergm_multis():
    """Goeyvaerts, as in R: 318 networks, of which the 225 included weekday ones
    have 1,300 contacts (R's summary of N(~edges))."""
    graphs = datasets.load("Goeyvaerts")
    assert len(graphs) == 318
    weekday = [g for g in graphs if g["included"] and g["weekday"]]
    assert len(weekday) == 225 and sum(g.ecount() for g in weekday) == 1300
    assert set(graphs[0].vs["role"]) == {"Father", "Mother", "Child"}
    nx_graphs = datasets.load("Goeyvaerts", backend="networkx")
    assert [g.number_of_edges() for g in nx_graphs] == [g.ecount() for g in graphs]
    assert nx_graphs[0].graph == {"included": True, "weekday": True}
    assert ergmx.summary_stats(ergmx.Networks(nx_graphs[:5]), "N(~edges + nodematch('role'))") == \
        ergmx.summary_stats(ergmx.Networks(graphs[:5]), "N(~edges + nodematch('role'))")


def test_dataset_errors():
    with pytest.raises(KeyError, match="no dataset 'florentine'"):
        datasets.load("florentine")
    with pytest.raises(ValueError, match="backend"):
        datasets.load("samplk3", backend="network")
