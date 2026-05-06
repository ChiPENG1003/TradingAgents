import unittest

import pandas as pd

from cli.main import _weekly_dates, _weekly_trading_dates


class BacktestScheduleTest(unittest.TestCase):
    def test_weekly_dates_advance_by_seven_calendar_days(self):
        self.assertEqual(
            _weekly_dates("2025-01-02", "2025-01-20"),
            ["2025-01-02", "2025-01-09", "2025-01-16"],
        )

    def test_weekly_trading_dates_shift_to_next_open_day(self):
        trading_dates = pd.to_datetime(["2025-01-02", "2025-01-10", "2025-01-16", "2025-01-20"])
        self.assertEqual(
            _weekly_trading_dates("2025-01-02", "2025-01-20", trading_dates),
            ["2025-01-02", "2025-01-10", "2025-01-16"],
        )


if __name__ == "__main__":
    unittest.main()
