"""The case registry: TOML loading, defaults, and error paths."""

import pytest

from flowkit.cases import Case, CASES_FILE, case, cases, load_registry, registry

MINIMAL = """
[defaults]
patch = "airfoil"
u_inf = [1.0, 0.0]

[cases.alpha]
path = "/tmp/alpha"

[cases.beta]
path  = "/tmp/beta"
patch = "cylinder"
u_inf = [0.5, 0.5]
nc    = "/tmp/beta.nc"
notes = "  spaced  "
"""


@pytest.fixture
def case_file(tmp_path):
    f = tmp_path / "cases.toml"
    f.write_text(MINIMAL)
    return f


def test_loads_every_case(case_file):
    reg = load_registry(case_file)
    assert sorted(reg) == ["alpha", "beta"]
    assert all(isinstance(c, Case) for c in reg.values())


def test_defaults_apply_when_a_case_is_silent(case_file):
    alpha = load_registry(case_file)["alpha"]
    assert alpha.patch == "airfoil"
    assert alpha.u_inf == (1.0, 0.0)


def test_a_case_overrides_the_defaults(case_file):
    beta = load_registry(case_file)["beta"]
    assert beta.patch == "cylinder"
    assert beta.u_inf == (0.5, 0.5)
    assert beta.aoa == pytest.approx(45.0)
    assert beta.notes == "spaced"


def test_explicit_nc_wins(case_file):
    assert str(load_registry(case_file)["beta"].netcdf) == "/tmp/beta.nc"


def test_missing_file_names_the_path_and_the_format(tmp_path):
    with pytest.raises(FileNotFoundError, match="No case file at"):
        load_registry(tmp_path / "absent.toml")


def test_case_without_path_is_rejected(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[cases.broken]\npatch = "airfoil"\n')
    with pytest.raises(KeyError, match="no 'path' key"):
        load_registry(f)


def test_unknown_key_is_rejected(tmp_path):
    """A typo'd key should fail loudly, not be silently ignored."""
    f = tmp_path / "c.toml"
    f.write_text('[cases.oops]\npath = "/tmp/x"\nu_infinity = [1, 0]\n')
    with pytest.raises(KeyError, match="unknown key"):
        load_registry(f)


def test_empty_file_gives_an_empty_registry(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("")
    assert load_registry(f) == {}


# ------------------------------------------------- the repo's own cases.toml

def test_repo_registry_parses():
    assert CASES_FILE.is_file(), f"expected a case file at {CASES_FILE}"
    assert "re500_aoa30" in cases()


def test_unknown_name_lists_what_exists():
    with pytest.raises(KeyError) as e:
        case("definitely_not_a_case")
    msg = str(e.value)
    assert "re500_aoa30" in msg
    assert "cases.toml" in msg


def test_registry_is_cached_but_reloadable():
    assert registry() is registry()
    assert registry(reload=True) is not None


def test_moving_mesh_case_is_registered_and_flagged():
    """step_input_aoa must carry its hazard in the notes -- see ISSUES #33."""
    c = case("step_input_aoa")
    assert c.path.exists()
    assert "MOVING MESH" in c.notes
