"""Post-hoc analysis pipeline (aggregation -> statistics -> plots -> LaTeX tables).

Having this as a package lets the scripts be run as modules from the repository root::

    python -m analysis.run_analysis_summary_clean --experiment v9_7 --trial1 4 --trial2 5

It is deliberately *not* part of the installed distribution (the analysis dependencies live
in the ``analysis`` extra), so run it from the checkout root.
"""
