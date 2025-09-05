import os, sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('../../'))  # import radjax

project = "RadJAX"
author = "TCIG-CIPS"
copyright = f"{datetime.now():%Y}, {author}"

# Version string (optional)
try:
    from radjax import __version__ as release
except Exception:
    release = ""

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}

extensions = [
    "myst_parser",
    "nbsphinx",                  
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    #"sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    #"sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# Sphinx 8 hardening
root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Don't execute notebooks (render only)
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"  # don’t try to resolve every hint in the signature
typehints_fully_qualified = False

# Mock heavy deps so imports don’t fail during docs build
autodoc_mock_imports = [
    "jax", "jaxlib", "flax", "chex", "optax", "matplotlib", "astropy",
]


# Intersphinx for Sphinx 8+
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable/", None),
    "jax":    ("https://jax.readthedocs.io/en/latest/", None),
}
