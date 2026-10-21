.. currentmodule:: dpsql

Validator
======================

The DPSQL+ Validator parses SQL strings (in a Spark/SQLite-like dialect) and performs static checks:

* Registration of private tables (``ALTER PRIVATE_TABLE ...``)
* Uniqueness and propagation of the privacy unit column in CTEs (``WITH`` clause)
* Allowed join constraints (only equality joins on privacy unit columns)
* Column and aggregation constraints in the final ``SELECT``
* Extraction of privacy parameters (``PRIVATE_QUERY OPTIONS ...``)
* Detection of unsupported constructs (e.g. ``PIVOT`` / ``UNPIVOT`` / ``LATERAL VIEW``)

If you already understand the syntax, see the raw grammar: :download:`sqlgrammar.lark <../../src/dpsql/validator/sqlgrammar.lark>`.

Validation Pipeline & Privacy Constraints
-----------------------------------------
1. ALTER PRIVATE_TABLE

  * Capture (optional) database name, table name, and exactly one ``PRIVACY_UNIT_COLUMN``.
  * Reject duplicate registrations.

2. WITH (CTEs)

  * Parse each CTE ``SELECT`` core.
  * If private tables are referenced, enforce exactly one privacy unit column.
  * For set operations (e.g. ``UNION`` / ``INTERSECT``), the privacy unit column name must match on both sides.

3. FROM / JOIN in CTEs

  * Disallow ``CROSS JOIN`` between private tables.
  * Allow only equality predicates (``table_a.col = table_b.col``) chained with ``AND`` for private tables.
  * Forbid duplicate equality predicates and self-joins on the same privacy unit column.

4. PRIVATE_QUERY OPTIONS

  * Extract privacy parameter literals (``EPSILON``, ``DELTA``, ``MIN_FREQUENCY``, ``CONTRIBUTION_BOUND``).
  * Validate ranges (``EPSILON > 0``, ``0 < DELTA <= 1``, ``MIN_FREQUENCY >= 1``, ``CONTRIBUTION_BOUND >= 1``).

5. Final SELECT

  * Forbid projecting the raw privacy unit column.
  * Forbid including the privacy unit column in ``GROUP BY``.
  * Forbid using the privacy unit column inside aggregation arguments.
  * ``DISTINCT`` only allowed inside aggregation functions (e.g. ``COUNT(DISTINCT col)``).
  * Disallow ``*`` wildcard.

6. Final Consistency

  * Ensure a single lineage of the privacy unit column remains.

Primary Exception Classes
-------------------------
Each may include a ``context`` dict and a corrective ``hint``.

* ``QueryParseError`` - Grammar / token / structural ordering errors
* ``PrivacyConstraintError`` - Privacy constraint violations (raw projection, ``GROUP BY`` usage, disallowed ``JOIN``)
* ``ValidationError`` - General logical/state validation failures (missing private table, unmatched counts)
* ``UnsupportedQueryError`` - Explicitly unsupported constructs (e.g. ``PIVOT``)

PRIVATE_QUERY OPTIONS Parameters
--------------------------------
Supported literals:

* ``EPSILON``: Positive float (e.g. ``1.0``)
* ``DELTA``: Float in ``(0, 1]`` (e.g. ``1e-5``)
* ``MIN_FREQUENCY`` (optional): Positive integer (e.g. ``10``)
* ``CONTRIBUTION_BOUND``: Positive integer (e.g. ``1``)

Aggregation Argument Form
-------------------------
* Clipping bounds are supplied inline for certain aggregations where required: ``SUM(column, lower_bound, upper_bound)``, ``AVG(column, lower_bound, upper_bound)``, ``STDDEV_SAMP(column, lower_bound, upper_bound)``, ``VAR_SAMP(column, lower_bound, upper_bound)``, ``COVAR_SAMP(column_a, column_b, lower_bound, upper_bound, lower_bound, upper_bound)``.
* Bounds must satisfy ``lower_bound <= upper_bound``.
* ``COUNT`` / ``COUNT(DISTINCT ...)`` take no clipping parameters.

Examples
--------
Simple private aggregation:

.. code-block:: sql

   ALTER PRIVATE_TABLE sales
     OPTIONS PRIVACY_UNIT_COLUMN(sales.user_id); -- Equivalent to dpsql.engine.Engine.register_database.

   WITH filtered AS (
     SELECT user_id, amount
     FROM sales
     WHERE amount > 0
   )
   SELECT PRIVATE_QUERY OPTIONS (
     EPSILON = 1.0,
     DELTA = 1e-5,
     CONTRIBUTION_BOUND = 1
   ) -- Equivalent to epsilon, delta, and contribution_bound in dpsql.dp_params.DPParams.
   COUNT(DISTINCT user_id) AS user_cnt,
       SUM(amount, 0, 15) AS total_amount -- Equivalent to clipping_thresholds in dpsql.dp_params.DPParams.
     FROM filtered
     ORDER BY total_amount DESC;

Allowed join (equality):

.. code-block:: sql

   ALTER PRIVATE_TABLE users
     OPTIONS PRIVACY_UNIT_COLUMN(users.uid); -- Equivalent to dpsql.engine.Engine.register_database.
   ALTER PRIVATE_TABLE sessions
     OPTIONS PRIVACY_UNIT_COLUMN(sessions.uid); -- Equivalent to dpsql.engine.Engine.register_database.

   WITH joined AS (
     SELECT s.uid, s.duration
     FROM users AS u
     JOIN sessions AS s
       ON u.uid = s.uid
   )
   SELECT PRIVATE_QUERY OPTIONS (
     EPSILON = 1.0,
     DELTA = 1e-5,
     MIN_FREQUENCY = 5,
     CONTRIBUTION_BOUND = 1
   ) -- Equivalent to dpsql.dp_params.DPParams.
   COUNT(DISTINCT uid) AS active_users,
       SUM(duration, 0, 100) AS total_duration -- Equivalent to clipping_thresholds in dpsql.dp_params.DPParams.
     FROM joined;

Violation (``GROUP BY`` includes privacy unit column):

.. code-block:: sql

   ALTER PRIVATE_TABLE joined
     OPTIONS PRIVACY_UNIT_COLUMN(joined.uid); 
   
   SELECT PRIVATE_QUERY OPTIONS ( 
     EPSILON = 1.0,
     DELTA = 1e-5,
     MIN_FREQUENCY = 5,
     CONTRIBUTION_BOUND = 1
   ) uid, SUM(duration, 0, 100)
     FROM joined
     GROUP BY uid;  -- INVALID

Raises ``PrivacyConstraintError`` (e.g. "Privacy unit column in `GROUP BY` is disallowed").
