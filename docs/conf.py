# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Add the src directory so autodoc/autoapi can find FRAME_FM
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -- Project information -----------------------------------------------------

project = "FRAME-FM"
author = "Adam Ward, National Oceanography Centre"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"
version = "0.0.0"
release = version + "a"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "numpydoc",
    # "autoapi.extension",  # Uncomment if you want autoapi instead of manual autodoc
    "myst_parser",
    "sphinx_last_updated_by_git",
    "sphinx_codeautolink",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = False

# -- AutoAPI configuration (optional) ----------------------------------------
# If you want Sphinx to automatically generate API docs from src/FRAME_FM

autoapi_dirs = [str(SRC_DIR / "FRAME_FM")]
autoapi_root = "api"
autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "no-private-members",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "groundwork"
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"

# IMPORTANT: change this to your actual GitHub Pages URL
html_baseurl = "https://NERC-EDS.github.io/FRAME-FM/"


pygments_style = "friendly"
pygments_dark_style = "monokai"

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "IPython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "python": ("https://docs.python.org/3/", None),
}

# -- Autodoc / Napoleon ------------------------------------------------------

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "fullqualname": True,
}

add_module_names = True

napoleon_include_init_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_custom_sections = None
napoleon_attr_annotations = True

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "smartquotes",
]

myst_heading_anchors = 3

# -- nbsphinx configuration --------------------------------------------------

# For local builds you might want "always", but for CI it's usually safer to avoid execution
# to keep builds fast and avoid missing-dependency issues.
nbsphinx_execute = os.environ.get("NBSPHINX_EXECUTE", "never")
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

nbsphinx_thumbnails = {
    "gallery/thumbnail-from-conf-py": "gallery/a-local-file.png",
    "gallery/*-rst": "images/notebook_icon.png",
    "orphan": "_static/favicon.svg",
}

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
