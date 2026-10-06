Fault class
===============================

``GetCumDis`` computes and returns cumulative along-strike distance from the
first point of the fault trace. Set ``discretized=True`` to use the discretized
trace; set ``recompute=False`` to retrieve the cached ``cumdis`` values.

.. code-block:: python

    cumulative_distance = fault.GetCumDis(discretized=True)

``saveGFs(outputDir=...)`` creates the output directory when it does not
already exist. The same behavior applies to the ``saveGFs`` methods on
``Block``, ``MultiBlock`` and ``Pressure``; ``transformation`` also creates
its output directory when saving Green's functions.

.. autoclass:: csi.Fault
    :members:

