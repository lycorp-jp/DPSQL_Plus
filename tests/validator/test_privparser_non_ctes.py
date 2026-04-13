from dpsql.aggregation import Aggregation, AggregationColumn
from dpsql.validator.privparser import PrivSQLParser


def test_non_ctes_non_private_analyze():
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT pcol FROM tbl2;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT * FROM tbl2;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT SUM(*), pcol FROM tbl2;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT a, b, c, SUM(*), SUM(pcol), COUNT(*), COUNT(pcol),
              COUNT(DISTINCT pcol), AVG(*), AVG(pcol), pcol FROM tbl2;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            alter private_table tbl options (privacy_unit_column = pcol);
            select a, b, c, sum(*), sum(pcol), count(*), count(pcol),
              count(distinct pcol), avg(*), avg(pcol), pcol from tbl2;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)


def test_non_ctes_non_private_analyze_group_by():
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT pcol FROM tbl2 group by pcol;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT * FROM tbl2 group by a;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT a, SUM(*) AS b, pcol FROM tbl2 group by a;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT a, b, c, SUM(*), SUM(pcol), COUNT(*), COUNT(pcol),
              COUNT(DISTINCT pcol), AVG(*), AVG(pcol), pcol FROM tbl2 group BY a, b, c;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT a, SUM(*), pcol FROM tbl2 GROUP BY a;
            """
    parser = PrivSQLParser(False)
    assert parser.analyze(query)


def test_non_ctes_private_analyze():
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(*), SUM(price) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["*"]),
        AggregationColumn(Aggregation.SUM, ["price"]),
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) AS b FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"], "b"),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY tbl.age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl ORDER BY age ASC LIMIT 10 OFFSET 5;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == [{"column_name": "age", "order": "ASC"}]
    assert parser.limit == 10
    assert parser.offset == 5
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl
            GROUP BY tbl.age
            ORDER BY tbl.age ASC
            LIMIT 10 OFFSET 5;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == [{"column_name": "age", "order": "ASC"}]
    assert parser.limit == 10
    assert parser.offset == 5
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl ORDER BY age DESC LIMIT 10;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == [{"column_name": "age", "order": "DESC"}]
    assert parser.limit == 10
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl ORDER BY age, name ASC;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == [
        {"column_name": "age", "order": None},
        {"column_name": "name", "order": "ASC"},
    ]
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""


def test_non_ctes_private_analyze_multi_table():
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(pcol) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["pcol"])
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT COUNT(*), SUM(a) FROM tbl;
            """
    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["*"]),
        AggregationColumn(Aggregation.SUM, ["a"]),
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, COUNT(pcol), SUM(a, 0.0, 5.0) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.COUNT, ["pcol"]),
        AggregationColumn(Aggregation.SUM, ["a"], parameters=[0.0, 5.0]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""

    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT age, AVG(a, 4.0, 10.0), STDDEV(a, 2.4, 6.2) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.AVG, ["a"], parameters=[4.0, 10.0]),
        AggregationColumn(Aggregation.STDDEV_SAMP, ["a"], parameters=[2.4, 6.2]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.common_table_expressions == ""


def test_non_ctes_private_analyze_dpparams():
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT PRIVATE_QUERY OPTIONS(epsilon=3.0, delta=1e-5, contribution_bound=3)
              age, AVG(a, 4.0, 10.0), STDDEV(a, 2.4, 6.2) FROM tbl GROUP BY age;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.NONE, ["age"]),
        AggregationColumn(Aggregation.AVG, ["a"], parameters=[4.0, 10.0]),
        AggregationColumn(Aggregation.STDDEV_SAMP, ["a"], parameters=[2.4, 6.2]),
    ]
    assert parser.group_by_columns == ["age"]
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.privacy_params == {
        "EPSILON": 3.0,
        "DELTA": 1e-05,
        "CONTRIBUTION_BOUND": 3,
    }
    query = """
            ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
            SELECT PRIVATE_QUERY OPTIONS(epsilon=5.0, delta=2e-5, contribution_bound=10)
              COUNT(a), SUM(a, 2.4, 6.2) FROM tbl;
            """

    parser = PrivSQLParser()
    assert parser.analyze(query)
    assert parser.final_privacy_unit_column == "pcol"
    assert parser.final_table_name == "tbl"
    assert parser.final_db_name is None
    assert parser.final_result_columns == [
        AggregationColumn(Aggregation.COUNT, ["a"]),
        AggregationColumn(Aggregation.SUM, ["a"], parameters=[2.4, 6.2]),
    ]
    assert parser.group_by_columns == []
    assert parser.ordering_terms == []
    assert parser.limit is None
    assert parser.offset is None
    assert parser.privacy_params == {
        "EPSILON": 5.0,
        "DELTA": 2e-05,
        "CONTRIBUTION_BOUND": 10,
    }
