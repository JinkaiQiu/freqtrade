"""Four-hour Turtle entry and risk with a completed daily lower-channel exit."""

from pandas import DataFrame

from freqtrade.strategy import informative
from HorizonTrend import FourHourTurtle55


class FourHourDailyExit(FourHourTurtle55):
    """Change only the lower exit channel of the fixed four-hour policy."""

    exit_label = "daily_channel20_fourhour_risk"

    @informative("1d")
    def populate_indicators_daily(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["lower20"] = dataframe["low"].rolling(20).min()
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["active_lower"] = dataframe["lower20_1d"].shift(1)
        return dataframe
