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
   asymmetric
   edgecov
   sender
   receiver
   sociality
   kstar
   istar
   ostar
   degree
   idegree
   odegree
   isolates
   concurrent
   twopath
   gwdegree
   gwidegree
   gwodegree
   triangle
   ttriple
   ctriple
   transitive
   cycle
   gwesp
   gwdsp
   gwnsp
   esp
   dsp
   nsp
   nodematch
   nodemix
   nodefactor
   nodeifactor
   nodeofactor
   nodecov
   nodeicov
   nodeocov
   absdiff
   absdiffcat

Bipartite terms
---------------

For bipartite networks (``bipartite=`` in :func:`ergm`): ``b1`` terms are
about the first mode, ``b2`` terms the second.

.. autosummary::
   :toctree: generated/

   b1star
   b2star
   b1degree
   b2degree
   gwb1degree
   gwb2degree
   b1concurrent
   b2concurrent
   b1factor
   b2factor
   b1cov
   b2cov
   b1nodematch
   b2nodematch
   b1dsp
   b2dsp
   gwb1dsp
   gwb2dsp

Operators
---------

.. autosummary::
   :toctree: generated/

   offset
   F
   terms.Curved

Constraints
-----------

Give constraints to :func:`ergm`, :func:`simulate` and :func:`gof` as strings
in R syntax, ``"bd(maxout=4) + blocks('level', levels2=2)"``. See the
:doc:`user guide <user-guide/constraints>`.

.. autosummary::
   :toctree: generated/

   constraints.parse_constraints
   constraints.Bd
   constraints.Blocks
   constraints.Degrees
   constraints.ODegrees
   constraints.IDegrees

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
