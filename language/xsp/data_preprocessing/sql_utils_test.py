from absl.testing import absltest

from language.xsp.data_preprocessing import sql_utils


class AnonymizeAliasesTest(absltest.TestCase):
    def test_count_function(self) -> None:
        sql = "select count( * ) from paper where paper.year = '1999' ;"
        self.assertEqual(sql, sql_utils.preprocess_sql(sql))

    def test_anonymize_alias(self) -> None:
        sql = 'SELECT DISTINCT WRITESalias0.AUTHORID FROM PAPER AS PAPERalias0 , VENUE AS VENUEalias0 , WRITES AS WRITESalias0 WHERE VENUEalias0.VENUEID = PAPERalias0.VENUEID AND VENUEalias0.VENUENAME = "venuename0" AND WRITESalias0.PAPERID = PAPERalias0.PAPERID ;'.lower()
        expected_sql = "select distinct T1.authorid from paper as T2 , venue as T3 , writes as T1 where T3.venueid = T2.venueid and T3.venuename = 'venuename0' and T1.paperid = T2.paperid ;"
        self.assertEqual(expected_sql, sql_utils.preprocess_sql(sql))

    def test_anonymize_alias_in_function(self) -> None:
        sql = "select count(WRITESalias0.AUTHORID) from paper as PAPERalias0 , writes as WRITESalias0 , where PAPERalias0.paperid = WRITESalias0.paperid ;".lower()
        expected_sql = "select count( T1.authorid ) from paper as T2 , writes as T1 , where T2.paperid = T1.paperid ;"
        self.assertEqual(expected_sql, sql_utils.preprocess_sql(sql))


if __name__ == "__main__":
    absltest.main()
