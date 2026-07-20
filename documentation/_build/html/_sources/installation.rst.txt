Installation
===============================

Dependencies
------------

CSI is a pure Python package. Core runtime dependencies are declared in ``pyproject.toml`` and include numpy, scipy, shapely, pyproj, matplotlib and cartopy.

Optional external tools:

- ``okada4py`` for some dislocation workflows (available on `GitHub <https://github.com/jolivetr/okada4py>`_)
- EDKS (external software by Luis Rivera) for layered Green's functions used by some advanced workflows

Install CSI (recommended)
-------------------------

Clone the repository, then install in editable mode:

.. code-block:: bash

	pip install -e .

Install documentation dependencies:

.. code-block:: bash

	pip install -e ".[docs]"

Build the documentation
-----------------------

From the ``documentation`` folder:

.. code-block:: bash

	make html

The generated site is written to ``documentation/_build/html``.

Legacy fallback
---------------

If editable installation is not available in your environment, CSI can still be used by adding the repository root to ``PYTHONPATH``.

.. code-block:: bash

	export PYTHONPATH=/path/to/csi:$PYTHONPATH

