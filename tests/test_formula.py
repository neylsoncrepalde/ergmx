import pytest

import ergmx
from ergmx import FormulaError, edges, gwesp, nodematch, parse_formula, triangle


def test_string_and_terms_give_the_same_formula():
    parsed = parse_formula("edges + nodematch('Grade') + gwesp(0.5, fixed=TRUE)")
    built = edges() + nodematch("Grade") + gwesp(0.5, fixed=True)
    assert repr(parsed) == repr(built) == "edges() + nodematch('Grade') + gwesp(0.5, fixed=True)"


def test_r_syntax():
    assert repr(parse_formula("net ~ edges + triangle")) == "edges() + triangle()"
    assert repr(parse_formula("gwesp(decay = 0.25, fixed = T)")) == "gwesp(0.25, fixed=True)"


def test_terms_combine_with_plus():
    formula = edges() + triangle() + "nodematch('Race')"
    assert len(formula) == 3


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("edges + kstar(2)", "unknown term 'kstar'"),
        ("edges + nodematch(attr)", "must be literals"),
        ("edges - triangle", "can't parse"),
        ("edges +", "can't parse the formula"),
        ("nodematch()", "nodematch: "),
    ],
)
def test_bad_formulas(formula, message):
    with pytest.raises(FormulaError, match=message):
        parse_formula(formula)


def test_unsupported_options_are_errors_not_silent():
    with pytest.raises(NotImplementedError, match="fixed=True"):
        gwesp(0.5)
    with pytest.raises(NotImplementedError, match="diff=True"):
        nodematch("Grade", diff=True)


def test_formula_must_not_be_empty():
    with pytest.raises(TypeError):
        ergmx.summary_stats(None, "edges")
