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
   san
   SanControl

Bayesian ERGMs
--------------

.. autosummary::
   :toctree: generated/

   bergm
   BergmFit
   BergmSummary
   bergmC
   evidence
   ModelEvidence
   ergm_apl
   AdjustedPL

Egocentric data
---------------

.. autosummary::
   :toctree: generated/

   EgoData
   ego_stats
   ergm_ego
   EgoFit

Several networks, and networks over time
----------------------------------------

.. autosummary::
   :toctree: generated/

   Networks
   NetSeries
   Layer
   tergm
   EgmmeFit
   simulate_dynamic
   btergm
   BtergmFit
   memory
   delrecip
   timecov
   DynamicSimulation

Results
-------

.. autosummary::
   :toctree: generated/

   ErgmFit
   FitSummary
   load_fit

Interpreting and reporting results
----------------------------------

.. autosummary::
   :toctree: generated/

   predict
   TiePredictions
   table
   ResultsTable
   NumericTable

Checking and comparing models
-----------------------------

.. autosummary::
   :toctree: generated/

   gof
   GofResult
   GofTable
   gofN
   GofNResult
   GofNTable
   GofNSummary
   lm_gofN
   LmFit
   McmcDiagnostics
   compare
   ModelComparison

Terms
-----

Call the functions to build a formula (``edges() + gwesp(0.5, fixed=True)``),
or write it as a string. See the :doc:`term reference <terms>` for the
statistics, and for terms written in Python:

.. autosummary::
   :toctree: generated/

   UserTerm
   register_term

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
   density
   meandeg
   dyadcov
   hamming
   attrcov
   mm
   diff
   smalldiff
   nodecovrange
   nodeicovrange
   nodeocovrange
   nodefactordistinct
   nodeofactordistinct
   nodeifactordistinct
   degrange
   idegrange
   odegrange
   degree1_5
   idegree1_5
   odegree1_5
   concurrentties
   isolatededges
   degcor
   degcrossprod
   altkstar
   triadcensus
   balance
   intransitive
   simmelian
   nearsimmelian
   simmelianties
   transitiveties
   cyclicalties
   threetrail
   opentriad
   tripercent
   localtriangle
   m2star
   desp
   ddsp
   dnsp
   dgwesp
   dgwdsp
   dgwnsp

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
   b1degrange
   b2degrange
   b1mindegree
   b2mindegree
   b1sociality
   b2sociality
   b1starmix
   b2starmix
   b1twostar
   b2twostar
   b1covrange
   b2covrange
   b1factordistinct
   b2factordistinct
   coincidence

Operators
---------

.. autosummary::
   :toctree: generated/

   offset
   F
   S
   N
   Form
   Persist
   Diss
   Cross
   Change
   terms.Curved

Statistics of tie ages
----------------------

tergm's durational statistics: targets of the EGMME, monitors of dynamic
simulations and terms of their models (see the :doc:`user guide
<user-guide/temporal>`); in formula strings, by their R names (``mean.age``,
``degree.mean.age``, ``EdgeAges``...).

.. autosummary::
   :toctree: generated/

   edge_ages
   mean_age
   edges_ageinterval
   edgecov_ages
   edgecov_mean_age
   nodefactor_mean_age
   nodemix_mean_age
   degree_mean_age
   degrange_mean_age

Multilevel terms
----------------

MPNet's configurations of two-level networks; see the
:doc:`user guide <user-guide/multilevel>`.

.. autosummary::
   :toctree: generated/

   star2ax
   star2bx
   axs1a
   axs1b
   aas1x
   abs1x
   aaaxs
   abaxs
   txax
   txbx
   atxax
   atxbx
   l3xax
   l3xbx
   l3axb
   c4axb
   exta
   extb
   asaxasb
   in2starax
   in2starbx
   out2starax
   out2starbx
   axs1ain
   axs1bin
   axs1aout
   axs1bout
   aains1x
   abins1x
   aaouts1x
   abouts1x
   txaxarc
   txbxarc
   txaxreciprocity
   txbxreciprocity
   atxaxarc
   atxbxarc
   atxaxreciprocity
   atxbxreciprocity
   l3xaxreciprocity
   l3xbxreciprocity
   l3axbin
   l3axbout
   l3axbpath
   l3bxapath
   c4axbentrainment
   c4axbexchange
   c4axbexchangeareciprocity
   c4axbexchangebreciprocity
   c4axbreciprocity
   ainasxainbs
   aoutasxaoutbs
   ainasxaoutbs
   aoutasxainbs

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
   constraints.Edges
   constraints.B1Degrees
   constraints.B2Degrees
   constraints.Fixedas
   constraints.Fixallbut
   constraints.Observed
   constraints.Blockdiag
   constraints.DyadsConstraint

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
   ErgmDifferenceWarning

Datasets
--------

.. autosummary::
   :toctree: generated/

   datasets.load
   datasets.names
   datasets.describe
```
