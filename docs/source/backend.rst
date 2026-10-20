.. currentmodule:: dpsql

Backend
======================
The DPSQL+ Backend handles the low-level operations required to execute differentially private SQL queries.

We provide backends for different data storage systems, including PySpark, SQLite, and DuckDB.

NULL, NaN, and infinity
-----------------------

The final private aggregation differs from ordinary SQL in the following ways.
These rules apply after CTE execution; released aggregates also include noise.

- **NULL privacy units:** The entire row is removed before contribution bounding
  and group selection.
- **Key types:** Floating-point privacy unit and ``GROUP BY`` columns raise
  ``ExecutionBackendError``.
- **All-NULL sums:** A retained group's ``SUM(x)`` starts from 0 before noise,
  whereas ordinary SQL returns NULL.
- **Empty results:** If no group passes selection, DPSQL+ returns no rows even
  without ``GROUP BY``.
- **NaN and infinity:** Numeric aggregates clip inputs to finite bounds
  ``[L, U]`` before computing sums, squares, or products: ``+Inf`` becomes ``U``
  and ``-Inf`` becomes ``L``. Spark and DuckDB also map NaN to ``U``. SQLite
  converts NaN to NULL during storage, so it is excluded instead.

.. automodule:: dpsql.backend
   :members:
   :undoc-members:
