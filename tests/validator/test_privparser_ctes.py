from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.validator.privparser import PrivSQLParser


def test_ctes_private_analyze():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 1
                                  UNION ALL
                                  SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 2)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
                      ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 1
                                  EXCEPT
                                  SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 2)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
                      ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 1)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
                      ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert (
        parser.common_table_expressions
        == "WITH ptbl2 AS (SELECT pcol, AVG(col) FROM ptbl WHERE pcol = 1)"
    )
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT pcol, AVG(col) FROM
                      ptbl WHERE pcol = 1 GROUP BY pcol)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
                      ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
               WITH ptbl2 AS (SELECT pcol, AVG(col) FROM ptbl
                              WHERE pcol = 1 GROUP BY pcol HAVING AVG(pcol) > 3)
               SELECT col, COUNT(col), COUNT(DISTINCT col2)
                FROM ptbl2 AS tbl GROUP BY tbl.col;"""
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None


def test_ctes_private_analyze_join():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT ptbl.pcol, AVG(tbl.col) FROM
                      ptbl JOIN tbl ON ptbl.pcol = tbl.pcol)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                     FROM ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl2 AS (SELECT ptbl.pcol, AVG(tbl.col)
                      FROM tbl LEFT JOIN ptbl ON ptbl.pcol = tbl.pcol)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2) FROM
                      ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE db1.ptbl OPTIONS
                      (PRIVACY_UNIT_COLUMN = ptbl.pcol);
                    WITH ptbl2 AS (SELECT ptbl.pcol, AVG(tbl.col)
                      FROM tbl RIGHT JOIN db1.ptbl ON ptbl.pcol = tbl.pcol)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl2 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl2"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None


def test_ctes_private_analyze_multi_table():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl LEFT JOIN ptbl2 ON ptbl.pcol = ptbl2.pcol2)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl JOIN ptbl2 ON ptbl.pcol = ptbl2.pcol2)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl RIGHT JOIN ptbl2 ON ptbl.pcol = ptbl2.pcol2
                      where ptbl.pcol in (1,2,3))
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl JOIN ptbl2 ON ptbl.pcol = ptbl2.pcol2
                        and ptbl2.a = ptbl.a)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    with ptbl3 as (select ptbl.pcol
                      from ptbl join ptbl2 on ptbl.pcol = ptbl2.pcol2
                        and ptbl2.a = ptbl.a GROUP BY ptbl.pcol
                          HAVING AVG(ptbl.pcol) > 1),
                    ptbl4 as (select pcol from ptbl3 where pcol = 1)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl4 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl4"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
                    with ptbl3 as (select ptbl.pcol
                      from ptbl join ptbl2 on ptbl.pcol = ptbl2.pcol2
                        and ptbl2.a = ptbl.a
                        WHERE ptbl.pcol = 1 GROUP BY ptbl.pcol
                          HAVING AVG(ptbl.pcol) > 1),
                    ptbl4 as (select pcol from ptbl3 where pcol = 1)
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl4 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl4"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None


def test_ctes_private_analyze_using():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a))
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol))
                    SELECT col, COUNT(col) AS a, COUNT(DISTINCT col2) AS b
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"], "a"),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"], "b"),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol)),
                    ptbl4 AS (SELECT hoge FROM a)
                    SELECT col, COUNT(col) AS a, COUNT(DISTINCT col2) AS b
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"], "a"),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"], "b"),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None


def test_ctes_private_analyze_hints():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a))
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"]),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"]),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REBALANCE */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol))
                    SELECT col, COUNT(col) AS a, COUNT(DISTINCT col2) AS b
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"], "a", []),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"], "b", []),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REBALANCE(3, c) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol)),
                    ptbl4 AS (SELECT hoge FROM a)
                    SELECT col, COUNT(col) AS a, COUNT(DISTINCT col2) AS b
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"], "a", []),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"], "b", []),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ SHUFFLE_HASH(t1), MERGE(t1,t2) */
                    ptbl.pcol FROM ptbl JOIN ptbl2 using (pcol)),
                    ptbl4 AS (SELECT hoge FROM a)
                    SELECT col, COUNT(col) AS a, COUNT(DISTINCT col2) AS b
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "ptbl3"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["col"]),
        AggregationColumn(Aggregation.COUNT, ["col"], "a", []),
        AggregationColumn(Aggregation.COUNT_DISTINCT, ["col2"], "b", []),
    ]
    assert parser.group_by_columns == ["col"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None


def test_ctes_private_analyze_spark_extension():
    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a PIVOT (
                        SUM(age) AS a, AVG(class) AS c
                        FOR name IN ('John' AS john, 'Mike' AS mike)
                      )
                    )
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
                    PIVOT (
                          SUM(age) AS a, AVG(class) AS c
                          FOR (name, age) IN (('John', 30) AS c1, ('Mike', 40) AS c2)
                      )
                    )
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
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
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
                    UNPIVOT INCLUDE NULLS (
                        sales FOR quarter IN (q1 AS Q1, q2 AS Q2, q3 AS Q3, q4 AS Q4)
                      ) AS up
                    )
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
                    UNPIVOT (
                        sales FOR quarter IN (q1, q2, q3, q4)
                      )
                    )
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
                    LATERAL VIEW EXPLODE(ARRAY(30, 60)) tableName AS c_age
                    LATERAL VIEW EXPLODE(ARRAY(40, 80)) AS d_age
                    )
                    SELECT col, COUNT(col), COUNT(DISTINCT col2)
                      FROM ptbl3 AS tbl GROUP BY tbl.col;
                    """
    parser = PrivSQLParser()
    assert parser.analyze(query)

    query = """
                    ALTER PRIVATE_TABLE ptbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    ALTER PRIVATE_TABLE ptbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
                    WITH ptbl3 AS (SELECT /*+ REPARTITION(3) */ ptbl.pcol
                      FROM ptbl JOIN ptbl2 using (pcol, a)),
                    ptbl4 AS (SELECT hoge FROM a
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
    parser = PrivSQLParser()
    assert parser.analyze(query)
