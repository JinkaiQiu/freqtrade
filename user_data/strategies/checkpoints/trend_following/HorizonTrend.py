"""Fixed hourly and four-hour adaptations of the daily Turtle comparison."""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import informative
from DailyTrend import DailyTrendBase


def _bar_channels(dataframe: DataFrame) -> DataFrame:
    dataframe["n"] = ta.ATR(dataframe, timeperiod=20)
    dataframe["upper55"] = dataframe["high"].rolling(55).max()
    dataframe["lower20"] = dataframe["low"].rolling(20).min()
    return dataframe


class HorizonTrendBase(DailyTrendBase):
    startup_candle_count = 400
    channel_timeframe = ""
    atr_timeframe = ""
    entry_label = ""
    exit_label = ""

    # Shadow the daily decorator inherited from DailyTrendBase. Only the
    # calendar-horizon control adds daily ATR as an informative dependency.
    def populate_indicators_daily(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        timeframe = self.channel_timeframe
        dataframe["atr"] = dataframe[f"n_{self.atr_timeframe}"]
        # Native informative merging first exposes a completed higher-timeframe
        # candle on the 15m row that closes alongside it. Its channels become
        # eligible at the next 15m row's open.
        dataframe["active_upper"] = dataframe[f"upper55_{timeframe}"].shift(1)
        dataframe["active_lower"] = dataframe[f"lower20_{timeframe}"].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cross = (dataframe["close"] > dataframe["active_upper"]) & (
            dataframe["close"].shift(1) <= dataframe["active_upper"]
        )
        dataframe["enter_long"] = (self._valid(dataframe) & cross).astype(int)
        dataframe["enter_tag"] = self.entry_label
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = (dataframe["close"] < dataframe["active_lower"]).astype(int)
        dataframe["exit_tag"] = self.exit_label
        return dataframe


class HourlyTurtle55(HorizonTrendBase):
    """55/20 completed hourly bars, with a fixed fill-based two-hourly-N stop."""

    channel_timeframe = atr_timeframe = "1h"
    entry_label = "hourly_turtle55"
    exit_label = "hourly_channel20"

    @informative("1h")
    def populate_indicators_hourly(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return _bar_channels(dataframe)


class FourHourTurtle55(HorizonTrendBase):
    """55/20 completed four-hour bars, with a fixed two-four-hour-N stop."""

    channel_timeframe = atr_timeframe = "4h"
    entry_label = "fourhour_turtle55"
    exit_label = "fourhour_channel20"

    @informative("4h")
    def populate_indicators_fourhour(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return _bar_channels(dataframe)


class FourHourCalendar55(HorizonTrendBase):
    """55/20 calendar-day channels sampled at 4h, with daily ATR20 stop."""

    channel_timeframe = "4h"
    atr_timeframe = "1d"
    entry_label = "fourhour_calendar55"
    exit_label = "fourhour_calendar20"

    @informative("4h")
    def populate_indicators_fourhour(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dated = dataframe.set_index("date")
        dataframe["upper55"] = dated["high"].rolling("55D").max().to_numpy()
        dataframe["lower20"] = dated["low"].rolling("20D").min().to_numpy()
        return dataframe

    @informative("1d")
    def populate_indicators_daily(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["n"] = ta.ATR(dataframe, timeperiod=20)
        return dataframe
