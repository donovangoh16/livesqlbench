import unittest

from system_agent.tools import (
    _categorical_literals,
    _normalize_plan,
    _unique_inferred_table_aliases,
)


class SafeGeneralizationTests(unittest.TestCase):
    def test_conventional_aliases_are_inferred_only_when_unique(self):
        aliases = _unique_inferred_table_aliases([
            "bank_and_transactions", "credit_and_compliance",
            "employment_and_income",
        ])
        self.assertEqual(aliases["bt"], "bank_and_transactions")
        self.assertEqual(aliases["cc"], "credit_and_compliance")
        self.assertEqual(aliases["ei"], "employment_and_income")

    def test_plan_expressions_resolve_conventional_aliases(self):
        plan, _ = _normalize_plan({
            "operation": "SELECT",
            "source_tables": ["bank_and_transactions", "employment_and_income"],
            "output_columns": [{"name": "id", "expression": "ei.emplcoreref"}],
            "population_constraints": [{
                "phrase": "high usage",
                "predicate": "bt.bankrelscore > 0.7",
            }],
        }, "Query")
        self.assertEqual(
            plan["output_columns"][0]["expression"],
            "employment_and_income.emplcoreref",
        )
        self.assertEqual(
            plan["population_constraints"][0]["predicate"],
            "bank_and_transactions.bankrelscore > 0.7",
        )

    def test_numeric_units_and_regex_are_not_categorical_values(self):
        predicate = (
            "frequency ~ '10\\\\s*hz' AND bandwidth = '22 kHz' "
            "AND ratio > '0.7' AND status = 'Active'"
        )
        self.assertEqual(_categorical_literals(predicate), ["Active"])


if __name__ == "__main__":
    unittest.main()
