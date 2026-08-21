Quick start
===========

This quick start guide helps you get up and running with DPSQL+.

Tutorial
--------
We recommend following the tutorial in the ``examples`` directory to better understand how to use DPSQL+.


How to Use
----------

To begin, install the project by following :doc:`./installation`.

To use DPSQL+, prepare the following:

- A standard SQL query where the final ``SELECT`` clause consists of aggregation expressions
- Privacy budget parameters (epsilon, delta)
- A Spark session object with Python bindings (SQLite3 and DuckDB backends are also supported)
- The database and table names used in the query
- The names of the columns representing user IDs (e.g., ``privacy_unit``) for each table, where applicable



1. **Initialize components**:

    Import and initialize the necessary components. For detailed information about each component, refer to the :doc:`./overview`.

    .. code-block:: python

        from dpsql.backend import SparkSQLBackend
        from dpsql.engine import Engine
        from dpsql.validator import Validator
        from dpsql.accountant import RenyiAccountant

        sql_backend = SparkSQLBackend(spark)
        validator = Validator()
        accountant = RenyiAccountant(epsilon, delta)
        engine = Engine(accountant, sql_backend, validator)

2. **Register databases**:

    Register the databases and tables you want to query with the engine, and specify privacy unit columns:

    .. code-block:: python

        engine.register_database("TITANIC", privacy_unit_columns={"passengers_features": "id", "passengers_survived": "id"})
        engine.register_database("SEX", privacy_unit_columns={})        

3. **Set query parameters**:

    Define the parameters for the query:

    .. code-block:: python

        from dpsql.dp_params import DPParams

        contribution_bound = 1
        min_frequency = 10
        epsilon_per_query = 0.5
        delta_per_query = 5e-5
        clipping = 1

        dpparams = DPParams(
            contribution_bound=contribution_bound, 
            min_frequency=min_frequency, 
            epsilon=epsilon_per_query, 
            delta=delta_per_query, 
            clipping_thresholds=[None, [(0, clipping)]]
        )
4. **Run the query**:

    Execute the query with the specified parameters:

    .. code-block:: python

        sql = """
        WITH combined_data_tmp AS (
            SELECT 
                e.id,
                e.who,
                s.survived
            FROM 
                TITANIC.passengers_features AS e
            JOIN 
                TITANIC.passengers_survived AS s ON e.id = s.id
        ), combined_data AS (
            SELECT
                c.id,
                c.who,
                c.survived,
                w.adult_male
            FROM
                combined_data_tmp AS c
            JOIN
                SEX.is_adult_male AS w ON c.who = w.who
        )
        SELECT 
            adult_male, COUNT(adult_male), SUM(survived)
        FROM 
            combined_data
        GROUP BY 
            adult_male
        """

        result = engine.execute_query(sql, dpparams)
        result.show()


Additional Usage
-----------
- **Specify sigma values directly**:

    You can specify sigma values directly instead of epsilon and delta:

    .. code-block:: python

        from dpsql.dp_params import DPParams

        contribution_bound = 1
        min_frequency = 10
        tau = 100
        sigma = 20
        sigma_for_thresholding = sigma

        dpparams = DPParams(contribution_bound=contribution_bound, min_frequency=min_frequency, tau=tau, sigma_for_thresholding=sigma_for_thresholding, sigmas=[sigma], clipping_thresholds=[None])
