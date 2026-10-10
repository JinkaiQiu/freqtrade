"""Fixed daily trend hypotheses, executed on completed 15-minute closes."""

from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute
from FrozenGuardRemoved import FrozenGuardRemoved


class DailyTrendBase(FrozenGuardRemoved):
    startup_candle_count = 300
    stoploss = -0.99

    @informative("1d")
    def populate_indicators_daily(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["n"] = ta.ATR(dataframe, timeperiod=20)
        dataframe["upper55"] = dataframe["high"].rolling(55).max()
        dataframe["lower20"] = dataframe["low"].rolling(20).min()
        dataframe["ma50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["ma200"] = ta.SMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = dataframe["n_1d"]
        # The native merge exposes day D on D's 23:45 bar, whose close is D+1 midnight.
        # Channel crossings use the threshold already known at the 15m bar's open.
        dataframe["active_upper"] = dataframe["upper55_1d"].shift(1)
        dataframe["active_lower"] = dataframe["lower20_1d"].shift(1)
        dataframe["daily_update"] = dataframe["date_1d"].ne(dataframe["date_1d"].shift(1))
        dataframe["bullish_ma"] = dataframe["ma50_1d"] > dataframe["ma200_1d"]
        return dataframe

    def _valid(self, dataframe: DataFrame) -> pd.Series:
        valid = (dataframe["volume"] > 0) & np.isfinite(dataframe["atr"]) & (dataframe["atr"] > 0)
        valid &= 2 * dataframe["atr"] < 0.99 * dataframe["close"]
        for start, end in self._quality_ranges():
            valid &= ~((dataframe["date"] < end) &
                       (dataframe["date"] + pd.Timedelta(minutes=15) > start))
        return valid

    @staticmethod
    def _boundary(current_time: datetime, candle: pd.Series) -> bool:
        return pd.Timestamp(current_time).floor("15min") == candle["date"] + pd.Timedelta(minutes=15)

    def _distance(self, atr: float, rate: float) -> float:
        return 2 * atr / rate

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        since = pd.Timestamp(current_time).normalize() - pd.Timedelta(microseconds=1)
        if any(t.nr_of_successful_entries > 0 for t in
               Trade.get_trades_proxy(pair=pair, open_date=since.to_pydatetime())):
            return False
        candles = self._closed(pair, current_time)
        if candles.empty or not 0 < self._distance(float(candles["atr"].iat[-1]), rate) < 0.99:
            return False
        return super().confirm_trade_entry(pair, order_type, amount, rate, time_in_force,
                                           current_time, entry_tag, side, **kwargs)

    def _state(self, pair: str, trade: Trade) -> dict | None:
        if trade.amount <= 0:
            return None
        if state := self._read_state(trade):
            return state
        candles = self._closed(pair, trade.open_date_utc)
        if candles.empty:
            return None
        atr = float(candles["atr"].iat[-1])
        if not np.isfinite(atr) or not 0 < 2 * atr < trade.open_rate:
            return None
        state = {"entry_atr": atr, "stop": float(trade.open_rate - 2 * atr),
                 "risk_fraction": float(2 * atr / trade.open_rate + self._costs())}
        self._save_state(trade, state, initial=True)
        return state

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs) -> float | None:
        state = self._state(pair, trade)
        return (stoploss_from_absolute(state["stop"], current_rate, leverage=1)
                if state else None)

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs) -> str | None:
        state = self._state(pair, trade)
        return "protective_stop_gap" if state and current_rate <= state["stop"] else None


class DailyTurtle55(DailyTrendBase):
    """55d/20d System-2-inspired long/cash adaptation; one entry per UTC day."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cross = (dataframe["close"] > dataframe["active_upper"]) & (
            dataframe["close"].shift(1) <= dataframe["active_upper"])
        dataframe["enter_long"] = (self._valid(dataframe) & cross).astype(int)
        dataframe["enter_tag"] = "daily_turtle55"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = (dataframe["close"] < dataframe["active_lower"]).astype(int)
        dataframe["exit_tag"] = "daily_channel20"
        return dataframe


class DailyMA50_200(DailyTrendBase):
    """Conventional fixed 50d/200d SMA hypothesis, retried only at daily updates."""

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = (self._valid(dataframe) & dataframe["daily_update"] &
                                   dataframe["bullish_ma"]).astype(int)
        dataframe["enter_tag"] = "daily_ma50_200"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = (dataframe["ma50_1d"] <= dataframe["ma200_1d"]).astype(int)
        dataframe["exit_tag"] = "daily_ma_bearish"
        return dataframe


class EqualBuyHold(IStrategy):
    """Unmanaged benchmark: 47.5% BTC, 47.5% ETH, 5% cash before fees."""

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False
    startup_candle_count = 300
    stoploss = -0.99
    minimal_roi = {}
    use_exit_signal = True
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = (dataframe["volume"] > 0).astype(int)
        dataframe["enter_tag"] = "equal_buy_hold"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs) -> float:
        stake = 0.475 * float(self.config["dry_run_wallet"])
        return min(stake, max_stake) if stake >= (min_stake or 0) else 0.0
