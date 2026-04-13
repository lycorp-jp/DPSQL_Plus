import pytest

from dpsql.errors import (
    PrivacyConstraintError,
    QueryParseError,
    UnsupportedQueryError,
)
from dpsql.validator.privparser import PrivSQLParser


def test_error_ctes_private_analyze():
    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT AVG(col) FROM ptbl WHERE pcol = 1)
        SELECT COUNT(tbl.col), COUNT(DISTINCT tbl.col2)
         FROM ptbl2 AS tbl;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: CTE projects no privacy unit column
        # (required exactly one for private base table)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT pcol, pcol, AVG(col) FROM ptbl
         WHERE pcol = 1 GROUP BY pcol)
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: CTE projects the privacy unit column twice
        # (must be exactly one distinct column)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT pcol AS col, AVG(col) FROM
          ptbl WHERE pcol = 1 GROUP BY pcol HAVING AVG(pcol) > 3 )
        SELECT col, COUNT(DISTINCT col2) FROM ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: FINAL SELECT groups by the privacy unit column
        # (grouping key cannot be privacy unit)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT AVG(col) FROM
          ptbl WHERE pcol = 1 GROUP BY pcol HAVING AVG(pcol) > 3 )
        SELECT col, COUNT(DISTINCT col2) FROM ptbl2 AS tbl;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Privacy unit column omitted in CTE output (must appear exactly once)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT DISTINCT AVG(col) FROM ptbl)
        SELECT COUNT(tbl.col), COUNT(DISTINCT tbl.col2)
         FROM ptbl2 AS tbl;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: DISTINCT aggregate only, still missing
        # required privacy unit column in CTE
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT ALL AVG(col) FROM ptbl)
        SELECT COUNT(tbl.col), COUNT(DISTINCT tbl.col2)
         FROM ptbl2 AS tbl;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: ALL aggregate only, no privacy unit column projected
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_ctes_private_analyze_join():
    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT ptbl.pcol1, AVG(tbl.col) FROM
          ptbl JOIN tbl ON ptbl.pcol = tbl.pcol)
        SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
          ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: CTE projection uses non-privacy column
        # (pcol1) instead of privacy unit column pcol
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT ptbl.pcol1, AVG(tbl.col) FROM
          tbl JOIN ptbl ON ptbl.pcol = tbl.pcol)
        SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
          ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Same as above; join order reversed,
        # privacy unit column still not projected
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (SELECT ptbl.pcol, AVG(tbl.col) FROM
          ptbl JOIN ptbl ON ptbl.pcol = ptbl.pcol)
        SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
          ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Self-join of same private table on
        # privacy unit column not allowed (leaks multiplicity)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (
            SELECT ptbl.pcol AS pcol1, ptbl.pcol AS pcol2, AVG(tbl.col)
            FROM tbl JOIN ptbl ON ptbl.pcol = tbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Duplicate occurrences of privacy
        # unit column (pcol1, pcol2) in CTE output
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl2 AS (
            SELECT hash(ptbl.pcol)
            FROM tbl JOIN ptbl ON ptbl.pcol = tbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl2 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Privacy unit column wrapped in a
        # scalar function (cannot appear in transformed form)
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_ctes_private_analyze_multi_table():
    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl2 ON ptbl.pcol1 = ptbl2.pcol2
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Join does not equi-join the declared
        # privacy unit columns (pcol vs pcol2)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl2 ON ptbl2.pcol1 = ptbl.pcol3
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Another non-equi privacy unit
        # mapping (wrong columns used)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl2 ON ptbl2.pcol2 != ptbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(QueryParseError):
        # Reason: Non-equality join operator (!=) rejected by parser/validator
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl2 ON pcol2 = ptbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(QueryParseError):
        # Reason: Missing table qualifiers in join
        # condition (ambiguous column references)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl2 ON ptbl.pcol = ptbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Join compares privacy unit column
        # to itself (not linking different tables)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol
            FROM ptbl JOIN ptbl ON ptbl.pcol = ptbl.pcol
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Self-join of same privacy table (prohibited)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol AS pcol, ptbl2.pcol2 AS pcol2
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol = ptbl2.pcol2 and ptbl2.a = ptbl.a
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Extra predicate after equi-join duplicates
        # privacy unit linkage (duplicate join pattern)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol AS pcol
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol = ptbl2.pcol2 or ptbl2.a = ptbl.a
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(QueryParseError):
        # Reason: OR operator in join condition unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol AS pcol
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol AND ptbl2.pcol2 = ptbl2.a AND ptbl.a
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(QueryParseError):
        # Reason: Malformed boolean expression
        # (dangling identifiers without comparisons)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol AS pcol
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol = ptbl2.pcol2 = ptbl2.a AND ptbl.a
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(QueryParseError):
        # Reason: Chained equality (a = b = c) unsupported in join parsing
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl.pcol AS pcol
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol = ptbl2.pcol2 and ptbl2.pcol2 = ptbl.pcol
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Duplicate symmetric equi-join on privacy unit columns
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        WITH ptbl3 AS (
            SELECT ptbl2.pcol2 AS pcol2
            FROM ptbl JOIN ptbl2
            ON ptbl.pcol = ptbl2.pcol2 and ptbl2.a = ptbl.a
        )
        SELECT pcol2, COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3 AS tbl GROUP BY tbl.pcol2;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Privacy unit column appears in FINAL SELECT GROUP BY
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        with ptbl3 as (
            select ptbl.pcol
            from ptbl join ptbl2
            on ptbl.pcol = ptbl2.pcol2 and ptbl2.a = ptbl.a
        ),
        ptbl4 as (select * from ptbl3)
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl4;
    """
    with pytest.raises(QueryParseError):
        # Reason: SELECT * in downstream private CTE chain
        # is disallowed (wildcard expansion)
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        with ptbl3 as (
            select ptbl.pcol
            from ptbl join ptbl2
            using (pcol2, a)
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: USING clause omits the shared privacy unit column pcol
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        with ptbl3 as (
            select ptbl.pcol2
            from ptbl join ptbl2
            using (pcol, a)
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Privacy unit column not projected in CTE output after USING join
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        with ptbl3 as (
            select ptbl.pcol
            from ptbl join ptbl2
            using (pcol, pcol)
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Duplicate privacy unit column listed twice in USING clause
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        with ptbl3 as (
            select ptbl.pcol
            from ptbl join ptbl
            using (pcol)
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: Self-join using privacy unit through USING clause
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        with ptbl3 as (
            select ptbl.pcol
            from ptbl cross join ptbl2
            using (pcol)
        )
        SELECT COUNT(col), COUNT(DISTINCT col2)
        FROM ptbl3;
    """
    with pytest.raises(PrivacyConstraintError):
        # Reason: CROSS JOIN between private tables breaks
        # required privacy-preserving equi-join constraint
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_ctes_private_analyze_hints():
    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /* REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a))
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(QueryParseError):
        # Reason: Unsupported inline comment style / hint token in grammar
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_ctes_private_analyze_spark_extensions():
    # Each Spark extension construct is mapped to UnsupportedQueryError when detected
    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a) PIVOT (
            SUM(age) AS a, AVG(class) AS c
            FOR name IN ('John' AS john, 'Mike' AS mike)
          )
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: PIVOT clause unsupported for private validation
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        PIVOT (
              SUM(age) AS a, AVG(class) AS c
              FOR (name, age) IN (('John', 30) AS c1, ('Mike', 40) AS c2)
          )
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: PIVOT with tuple target unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        UNPIVOT EXCLUDE NULLS (
            (first_quarter, second_quarter)
            FOR half_of_the_year IN (
                (q1, q2) AS H1,
                (q3, q4) AS H2
            )
          )
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: UNPIVOT EXCLUDE NULLS unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        UNPIVOT INCLUDE NULLS (
            sales FOR quarter IN (q1 AS Q1, q2 AS Q2, q3 AS Q3, q4 AS Q4)
          ) AS up
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: UNPIVOT INCLUDE NULLS unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        UNPIVOT (
            sales FOR quarter IN (q1, q2, q3, q4)
          )
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: Basic UNPIVOT unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        LATERAL VIEW EXPLODE(ARRAY(30, 60)) tableName AS c_age
        LATERAL VIEW EXPLODE(ARRAY(40, 80)) AS d_age
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: LATERAL VIEW EXPLODE unsupported
        parser = PrivSQLParser()
        parser.analyze(query)

    query = """
        ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
          FROM ptbl JOIN ptbl2 using (pcol, a)
        PIVOT (
              SUM(age) AS a, AVG(class) AS c
              FOR (name, age) IN (('John', 30) AS c1, ('Mike', 40) AS c2)
          )
        UNPIVOT (
            sales FOR quarter IN (q1, q2, q3, q4)
          )
        LATERAL VIEW EXPLODE(ARRAY(30, 60)) tableName AS c_age
        LATERAL VIEW EXPLODE(ARRAY(40, 80)) AS d_age
        )
        SELECT col, COUNT(col), COUNT(DISTINCT col2)
          FROM ptbl3 AS tbl GROUP BY tbl.col;
    """
    with pytest.raises(UnsupportedQueryError):
        # Reason: Combination of unsupported Spark extensions
        # (PIVOT + UNPIVOT + LATERAL VIEW)
        parser = PrivSQLParser()
        parser.analyze(query)
