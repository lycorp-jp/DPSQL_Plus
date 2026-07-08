Overview
======================

Architecture
-------------------

Main Components
~~~~~~~~~~~~~~~~~~

:doc:`./engine` is the main component of the system. 
It receives queries from users and sends each query to the Validator to check whether it satisfies the privacy constraints.
If the query is valid, the Engine inquires the Accountant to check whether sufficient privacy budget remains.
If the budget is sufficient, the Engine sends the query to the Backend for execution on the database.

:doc:`./validator` is the component that validates the query.
It receives the query from the Engine and checks if it satisfies the privacy constraints.

:doc:`./accountant` is the component that manages the privacy budget.
Before the system starts, the Admin sets the total privacy budget, which corresponds to the acceptable privacy leakage. 
The Accountant calculates the privacy budget consumed each time a query is executed and checks if it exceeds the limit.

:doc:`./backend` is the component that executes the query on the database.
It executes the query received from the Engine and adds noise to the results to ensure the differential privacy.


Diagram
~~~~~~~~~~~~~~~~~~

.. mermaid::
   
   graph LR
      User([👤 User]) --> Engine[⚙️ Engine]
      Engine --> Validator[✅ Validator]
      Engine --> Accountant[🔐 Accountant]
      Engine --> Backend[🔌 Backend]

      subgraph db["Spark / SQLite3 / DuckDB"]
         Backend --> Database[(Database)]
      end

      subgraph pbm["Privacy&nbsp;Budget&nbsp;Management"]
         Admin([🛠️ Admin]) --> Accountant
      end
      
Flow of the System
-------------------
.. mermaid::

   sequenceDiagram
      participant Admin as 🛠️ Admin
      participant User as 👤 User
      participant Engine as ⚙️ Engine
      participant Validator as ✅ Validator
      participant Accountant as 🔐 Accountant
      participant Backend as 🔌 Backend

      Admin ->> Engine: Register tables and privacy units
      Admin ->> Accountant: Set privacy budget
      User ->> Engine: Execute query
      Engine ->> Validator: Validate query
      Validator ->> Validator: Check whether it satisfies privacy constraints
      Validator -->> Engine: Validation result
      Engine ->> Accountant: Update privacy budget
      Accountant ->> Accountant: Check whether privacy budget is sufficient
      Accountant -->> Engine: Check result
      Engine ->> Backend: Execute query
      Backend ->> Backend: Create intermediate tables
      Backend ->> Backend: Execute final select
      Backend ->> Backend: Add noise
      Backend -->> Engine: Results
      Engine -->> User: Results

Query Syntax
-------------------
DPSQL+ admits a subset of SQL queries.
Specifically, queries must adhere to the following syntax:

.. code-block::

   query_statement ::= [cte, ...] final_select
   cte ::= WITH cte_name AS (select_statement)
   select_statement ::= SELECT expr [,...] FROM table_reference [WHERE expr] [GROUP BY expr [,...]] [HAVING expr] 
   final_select ::= SELECT [PRIVATE_QUERY OPTIONS (...)] [agg_expr, ...] FROM table_name | cte_name [GROUP BY column_name]
   table_reference ::= table_name | cte_name | table_reference JOIN table_reference ON condition
   agg_expr ::= agg_func([DISTINCT] expr)
   agg_func ::= COUNT | SUM | AVG ... etc
   
Particular attention should be given to the following points:

- The query must start with a sequence of Common Table Expressions (CTEs) followed by a final select statement.
- CTEs must emit the privacy unit if they involve data that should be protected. 
- In CTEs, joins between tables that need to be protected are only permitted as equi-joins.
- The final select statement can only include aggregation functions such as ``COUNT`` and ``SUM`` to protect the privacy of the data.

For detailed syntax and privacy constraints, see :doc:`./validator`.