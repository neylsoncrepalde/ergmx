# API reference

```{eval-rst}
.. currentmodule:: ergmx

Fitting and simulating
----------------------

.. autosummary::
   :toctree: generated/

   ergm
   simulate
   summary_stats
   Control

Results
-------

.. autosummary::
   :toctree: generated/

   ErgmFit
   FitSummary

Checking and comparing models
-----------------------------

.. autosummary::
   :toctree: generated/

   gof
   GofResult
   GofTable
   McmcDiagnostics
   compare
   ModelComparison

Terms
-----

Call the functions to build a formula (``edges() + gwesp(0.5, fixed=True)``),
or write it as a string. See the :doc:`term reference <terms>` for the
statistics.

.. autosummary::
   :toctree: generated/

   edges
   mutual
   edgecov
   kstar
   istar
   ostar
   gwdegree
   gwidegree
   gwodegree
   triangle
   ttriple
   ctriple
   gwesp
   gwdsp
   nodematch
   nodefactor
   nodeifactor
   nodeofactor
   nodecov
   nodeicov
   nodeocov
   absdiff

Formulas
--------

.. autosummary::
   :toctree: generated/

   parse_formula
   Formula
   Term

Errors
------

.. autosummary::
   :toctree: generated/

   DegeneracyError
   FormulaError

Datasets
--------

.. autosummary::
   :toctree: generated/

   datasets.load
   datasets.names
   datasets.describe
```
