"""Sphinx configuration for the ergmx documentation.

Build locally, after installing ergmx (it needs Rust to build):

    pip install -e ".[docs]"
    sphinx-build -W --keep-going -d docs/_build/doctrees docs docs/_build/html
"""

from importlib.metadata import version as _version

project = "ergmx"
author = "Neylson Crepalde"
copyright = "2026, Neylson Crepalde"
release = _version("ergmx")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "figures"]

# -- MyST Markdown and executable pages --------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "attrs_inline"]
myst_heading_anchors = 3
# Every code cell runs on each build: a broken example fails the build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600
nb_merge_streams = True

# -- API reference ------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "none"  # Types are documented in the numpy-style docstrings.
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "callable": ":term:`callable`",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "igraph": ("https://python.igraph.org/en/stable/api", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "ergmx"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/neylsoncrepalde/ergmx",
    "navbar_align": "left",
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_sidebars = {
    "index": [], "api": [], "coming-from-r": [], "validation": [], "terms": [], "changelog": [],
}
copybutton_exclude = ".linenos, .gp, .go"
