import pytest

from dpsql.errors import PrivacyConstraintError, QueryParseError, ValidationError
from dpsql.validator.privparser import PrivSQLParser


def test_error_non_ctes_non_private_analyze():
    # Reason: COUNT(DISTINCT *) is not supported
    # by the grammar / validator -> parsed as syntax issue -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT COUNT(DISTINCT *) FROM tbl2;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: SUM(DISTINCT *) unsupported aggregate argument
    # form -> syntax/parse rejection -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT SUM(DISTINCT *) FROM tbl2;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: AVG(DISTINCT *) unsupported (same rationale as above) -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT AVG(DISTINCT *) FROM tbl2;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: Scalar literals (1,4) not allowed
    # in FINAL SELECT (expects single expr or aggregates) -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT 1, 4 FROM tbl2;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: HAVING clause not supported
    # in final SELECT core (not implemented in validator) -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol, COUNT(*) FROM tbl2 GROUP BY pcol HAVING a > 10;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: Scalar function not supported -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol, HASH(pcol) FROM tbl2 GROUP BY pcol;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)

    # Reason: WHERE clause not supported
    # in FINAL SELECT grammar variant -> QueryParseError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol FROM tbl2 WHERE pcol = 1;
    """
    with pytest.raises(QueryParseError):
        parser = PrivSQLParser(False)
        parser.analyze(query)


def test_error_non_ctes_private_analyze():
    # Reason: Raw privacy unit column together
    # with COUNT(DISTINCT pcol); raw projection
    # of privacy unit disallowed -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT COUNT(DISTINCT pcol), pcol FROM tbl;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Multiple raw non-aggregate columns (pcol, a, b) without
    # GROUP BY -> violation of aggregation rules -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol, a, b FROM tbl;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: SELECT * not allowed in final result
    # columns (explicit columns required) -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT * FROM tbl;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_non_ctes_private_analyze_group_by():
    # Reason: Privacy unit column appears in GROUP BY and raw
    # projection (pcol) -> privacy constraint violation -> PrivacyConstraintError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT COUNT(DISTINCT pcol), pcol FROM tbl GROUP BY pcol;
    """
    with pytest.raises(PrivacyConstraintError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: COUNT(DISTINCT pcol) plus GROUP BY pcol leaks unit
    # via grouping key -> privacy constraint -> PrivacyConstraintError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol, COUNT(DISTINCT pcol) FROM tbl GROUP BY pcol;
    """
    with pytest.raises(PrivacyConstraintError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Projected column b not included in
    # GROUP BY (only a) -> projection/group-by mismatch -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT a, b, COUNT(DISTINCT pcol) FROM tbl GROUP BY a;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: SELECT * with GROUP BY unsupported
    # (`*` forbidden) -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT * FROM tbl group by a;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Raw privacy unit column (pcol) not
    # grouped properly (GROUP BY a) -> privacy
    # + grouping violation -> PrivacyConstraintError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT a, COUNT(DISTINCT pcol), pcol FROM tbl GROUP BY a;
    """
    with pytest.raises(PrivacyConstraintError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Privacy unit column appears raw while
    # GROUP BY excludes it (a, b) -> privacy constraint -> PrivacyConstraintError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT a, b, COUNT(DISTINCT pcol), pcol FROM tbl GROUP BY a, b;
    """
    with pytest.raises(PrivacyConstraintError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Non-privacy columns a,b projected together
    # with COUNT(DISTINCT pcol) and GROUP BY
    # only (a, pcol) -> column b missing in
    # GROUP BY -> PrivacyConstraintError (privacy/raw mismatch precedence)
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol, COUNT(DISTINCT pcol), a, b FROM tbl GROUP BY a, pcol;
    """
    with pytest.raises(PrivacyConstraintError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: GROUP BY column (age) not included
    # in SELECT clause -> projection/group-by mismatch -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT COUNT(pcol) FROM tbl GROUP BY age;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)


def test_error_non_ctes_private_analyze_multi_table():
    # Reason: Mixed private tables; projecting other table's
    # privacy unit (pcol2) with COUNT(DISTINCT pcol) -> cross
    # privacy leakage -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        SELECT COUNT(DISTINCT pcol), pcol2 FROM tbl2;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Alias on raw column (col as c) without aggregate
    # allowed only under aggregate path rules -> alias
    # misuse / raw projection rule -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        SELECT col, COUNT(DISTINCT pcol2) AS b, col as c FROM tbl2 group by col;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Duplicate ALTER PRIVATE_TABLE for same logical
    # table with different privacy unit column
    # (pcol vs pcol2) -> already registered conflict -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol2);
        SELECT a, b FROM tbl;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Duplicate registration with identical privacy
    # unit column -> redundant definition -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT a, b FROM tbl;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Referencing another private table's privacy unit
    # column raw (pcol from tbl2) violates privacy projection rules -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol FROM tbl2;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)

    # Reason: Qualified db.tbl2 still raw privacy unit column
    # projection (pcol) -> same violation -> ValidationError
    query = """
        ALTER PRIVATE_TABLE tbl OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        ALTER PRIVATE_TABLE db.tbl2 OPTIONS (PRIVACY_UNIT_COLUMN = pcol);
        SELECT pcol FROM db.tbl2;
    """
    with pytest.raises(ValidationError):
        parser = PrivSQLParser()
        parser.analyze(query)
