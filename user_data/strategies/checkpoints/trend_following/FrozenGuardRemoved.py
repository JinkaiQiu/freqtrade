"""Causal 15-minute spot candidates; research knobs live in custom.volatility_research."""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import ndtr

from freqtrade.exceptions import OperationalException
from freqtrade.persistence import Order, Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute

logger = logging.getLogger(__name__)


class FrozenGuardRemoved(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False
    startup_candle_count = 1600
    process_only_new_candles = True
    position_adjustment_enable = False
    use_custom_stoploss = True
    trailing_stop = False
    stoploss = -0.08
    minimal_roi = {}
    use_exit_signal = True
    exit_profit_only = False
    default_engine = "breakout"

    def _params(self) -> dict:
        return self.config.get("custom", {}).get("volatility_research", {})

    @staticmethod
    def _parse_quality_exclusions(exclusions) -> tuple:
        if not isinstance(exclusions, list):
            raise ValueError("quality_exclusions must be a list")
        intervals = []
        for item in exclusions:
            if not isinstance(item, dict) or not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise ValueError("Quality exclusions require a nonempty reason")
            start = pd.Timestamp(item.get("start_utc"))
            ends = [pd.Timestamp(item[name]) for name in ("end_exclusive_utc", "end_exclusive") if name in item]
            if not ends:
                raise ValueError("Quality exclusions require an exclusive end")
            if any(pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0
                   for stamp in (start, *ends)):
                raise ValueError("Quality exclusion boundaries must be explicit UTC timestamps")
            if any(stamp != ends[0] for stamp in ends):
                raise ValueError("Quality exclusion exclusive ends conflict")
            end = ends[0]
            if end <= start or (intervals and start < intervals[-1][1]):
                raise ValueError("Quality exclusions must be positive, sorted, and nonoverlapping")
            intervals.append((start.tz_convert("UTC"), end.tz_convert("UTC")))
        return tuple(intervals)

    def _quality_ranges(self) -> tuple:
        if "_quality_intervals" not in self.__dict__:
            self._quality_intervals = self._parse_quality_exclusions(
                self._params().get("quality_exclusions", [])
            )
        return self._quality_intervals

    def _quality_blocked(self, current_time: datetime) -> bool:
        now = pd.Timestamp(current_time)
        return any(start <= now < end for start, end in self._quality_ranges())

    def _quality_bar_blocked(self, date) -> bool:
        start = pd.Timestamp(date)
        end = start + pd.Timedelta(minutes=15)
        return any(start < excluded_end and end > excluded_start
                   for excluded_start, excluded_end in self._quality_ranges())

    def bot_start(self, **kwargs) -> None:
        self._closed_cache = {}
        self._last_candle_cache = {}
        self._risk_state = {"peak": 0.0, "halted": False}
        self._risk_path = None
        p = self._params()
        self._quality_intervals = self._parse_quality_exclusions(p.get("quality_exclusions", []))
        if p.get("enable_drawdown_guard", False) and self.config.get("runmode") in (
            "live", "dry_run"
        ):
            path = p.get("research_state_path")
            if not path:
                raise OperationalException("Drawdown guard requires research_state_path in live/dry_run")
            self._risk_path = Path(path).expanduser().resolve()
            if self._risk_path.exists():
                state = json.loads(self._risk_path.read_text())
                if not np.isfinite(state["peak"]) or state["peak"] < 0 or not isinstance(
                    state["halted"], bool
                ):
                    raise OperationalException("Invalid persisted drawdown guard state")
                self._risk_state = state

    def _guard_scale(self, equity: float) -> float:
        if not self._params().get("enable_drawdown_guard", False):
            return 1.0
        if "_risk_state" not in self.__dict__:
            self.bot_start()
        state = self._risk_state
        previous = dict(state)
        state["peak"] = max(state["peak"], equity)
        drawdown = 1 - equity / state["peak"] if state["peak"] > 0 else 0.0
        if drawdown >= 0.10 - 1e-12:
            state["halted"] = True
        if state != previous and self._risk_path is not None:
            try:
                self._risk_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._risk_path.with_name(self._risk_path.name + ".tmp")
                temporary.write_text(json.dumps(state, allow_nan=False))
                temporary.replace(self._risk_path)
            except OSError:
                state["halted"] = True
                logger.exception("Drawdown guard persistence failed; blocking new entries")
                return 0.0
        if state["halted"]:
            return 0.0
        return 0.5 if drawdown >= 0.08 - 1e-12 else (
            0.75 if drawdown >= 0.05 - 1e-12 else 1.0
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        p = self._params()
        close = dataframe["close"]
        previous = close.shift(1)
        tr = pd.concat(
            [dataframe["high"] - dataframe["low"],
             (dataframe["high"] - previous).abs(),
             (dataframe["low"] - previous).abs()], axis=1,
        ).max(axis=1)
        dataframe["atr"] = tr.rolling(14).mean().shift(1)
        returns = np.log(close / previous)
        dataframe["rv_fast"] = returns.rolling(int(p.get("rv_fast", 4))).std(ddof=0)
        dataframe["rv_day"] = returns.rolling(96).std(ddof=0).shift(1)
        dataframe["rv_ratio"] = dataframe["rv_fast"] / dataframe["rv_day"].replace(0, np.nan)
        weekly_std = returns.rolling(672).std(ddof=0)
        dataframe["weekly_trend"] = returns.rolling(672).sum() / (
            weekly_std.replace(0, np.nan) * np.sqrt(672)
        )
        signed_pressure = dataframe["volume"] * (
            2 * ndtr(returns / dataframe["rv_day"].replace(0, np.nan)) - 1
        )
        dataframe["pressure"] = signed_pressure.rolling(16).sum()
        squared = returns.pow(2)
        downside = squared.where(returns < 0, 0).rolling(96).sum() / (
            squared.rolling(96).sum().replace(0, np.nan)
        )
        dataframe["downside_scale"] = (0.5 / downside.clip(lower=0.5)).clip(upper=1).fillna(1)
        dataframe["ema_fast"] = close.ewm(
            span=int(p.get("trend_fast", 80)), adjust=False,
            min_periods=int(p.get("trend_fast", 80)),
        ).mean()
        dataframe["ema_slow"] = close.ewm(
            span=int(p.get("trend_slow", 320)), adjust=False,
            min_periods=int(p.get("trend_slow", 320)),
        ).mean()
        dataframe["trend_slope"] = dataframe["ema_slow"].pct_change(16, fill_method=None)
        window = int(p.get("lookback", 24))
        dataframe["channel_high"] = dataframe["high"].rolling(window).max().shift(1)
        dataframe["channel_low"] = dataframe["low"].rolling(
            int(p.get("exit_lookback", 12))
        ).min().shift(1)
        dataframe["volume_mean"] = dataframe["volume"].rolling(24).mean().shift(1)
        dataframe["quote_volume_hour"] = (dataframe["volume"] * close).rolling(4).sum()
        range_window = int(p.get("range_window", 32))
        dataframe["range_mid"] = close.rolling(range_window).mean().shift(1)
        dataframe["range_std"] = close.rolling(range_window).std(ddof=0).shift(1)
        change = close.diff()
        gains = change.clip(lower=0).rolling(14).mean()
        losses = (-change.clip(upper=0)).rolling(14).mean()
        dataframe["rsi"] = 100 - 100 / (1 + gains / losses.replace(0, np.nan))
        dataframe.loc[(losses == 0) & (gains > 0), "rsi"] = 100
        dataframe.loc[(losses == 0) & (gains == 0), "rsi"] = 50
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        p = self._params()
        close = dataframe["close"]
        atr_fraction = dataframe["atr"] / close
        liquid = (dataframe["volume"] > 0) & atr_fraction.between(
            float(p.get("min_atr_fraction", 0.001)),
            float(p.get("max_atr_fraction", 0.04)),
        )
        for start, end in self._quality_ranges():
            liquid &= ~((dataframe["date"] < end) & (
                dataframe["date"] + pd.Timedelta(minutes=15) > start
            ))
        if p.get("enable_weekly_trend", False):
            liquid &= dataframe["weekly_trend"] > 0
        if p.get("enable_pressure", False):
            liquid &= dataframe["pressure"] > 0
        trend = (dataframe["ema_fast"] > dataframe["ema_slow"]) & (
            dataframe["trend_slope"] > float(p.get("min_trend_slope", 0))
        )
        breakout = liquid & trend & (close > dataframe["channel_high"]) & (
            dataframe["volume"] > dataframe["volume_mean"] * float(p.get("volume_ratio", 1.2))
        ) & (dataframe["rv_ratio"] > float(p.get("vol_ratio", 1.1)))
        pullback = liquid & trend & (close > dataframe["ema_slow"]) & (
            dataframe["low"] <= dataframe["ema_fast"] + dataframe["atr"]
            * float(p.get("pullback_atr", 0.5))
        ) & (close > dataframe["open"]) & (close > close.shift(1)) & (
            dataframe["rv_ratio"] < float(p.get("pullback_max_vol_ratio", 2.5))
        ) & (dataframe["rsi"] < float(p.get("pullback_max_rsi", 65)))
        range_revert = liquid & (
            dataframe["trend_slope"].abs() < float(p.get("range_max_slope", 0.002))
        ) & (dataframe["rv_ratio"] < float(p.get("range_max_vol_ratio", 1.2))) & (
            close < dataframe["range_mid"] - dataframe["range_std"]
            * float(p.get("range_z", 1.5))
        ) & (dataframe["rsi"] < float(p.get("range_max_rsi", 38))) & (
            close > dataframe["open"]
        )
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = ""
        engine = p.get("engine", self.default_engine)
        for name, signal in (("range", range_revert), ("pullback", pullback),
                             ("breakout", breakout)):
            if engine in (name, "combined"):
                dataframe.loc[signal, ["enter_long", "enter_tag"]] = (1, name)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe

    def _closed(self, pair: str, current_time: datetime) -> DataFrame:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return dataframe
        cutoff = pd.Timestamp(current_time).floor("15min") - pd.Timedelta(minutes=15)
        key = (cutoff, dataframe["date"].iat[-1], len(dataframe))
        cache = self.__dict__.setdefault("_closed_cache", {})
        previous = cache.get(pair)
        if previous is None or previous[0] != key:
            cache[pair] = (key, dataframe.loc[dataframe["date"] <= cutoff])
        return cache[pair][1]

    def _last_candle(self, pair: str, candles: DataFrame) -> pd.Series:
        cache = self.__dict__.setdefault("_last_candle_cache", {})
        previous = cache.get(pair)
        if previous is None or previous[0] is not candles:
            cache[pair] = (candles, candles.iloc[-1])
        return cache[pair][1]

    def _read_state(self, trade: Trade) -> dict | None:
        state = trade.get_custom_data("volatility_state")
        return dict(state) if state else None

    def _save_state(self, trade: Trade, state: dict, initial: bool = False) -> None:
        trade.set_custom_data("volatility_state", dict(state))

    @staticmethod
    def _boundary(current_time: datetime, candle: pd.Series) -> bool:
        now = pd.Timestamp(current_time)
        return now.floor("15min") == candle["date"] + pd.Timedelta(minutes=15) and (
            now.minute % 15 == 0
        )

    def _distance(self, atr: float, rate: float) -> float:
        return float(np.clip(float(self._params().get("stop_atr", 2.5)) * atr / rate,
                             0.005, abs(self.stoploss)))

    def _costs(self) -> float:
        fee = self.config.get("fee", 0.001)
        return 2 * float(0.001 if fee is None else fee) + float(
            self._params().get("adverse_cost", 0.002)
        )

    def _liquidity_cap(self, candle: pd.Series) -> float | None:
        participation = self._params().get("max_hour_participation")
        if participation is None:
            return None
        volume = float(candle["quote_volume_hour"])
        participation = float(participation)
        if not np.isfinite(volume) or volume <= 0 or not np.isfinite(participation) or participation <= 0:
            return 0.0
        return participation * volume

    def _portfolio(self, current_time: datetime) -> tuple[float, float, int, float]:
        live_cash = self.config.get("runmode") == "live" and "available_capital" not in self.config
        if live_cash:
            # Exchange cash has already paid entry fees. Add liquidation value directly.
            equity = float(self.wallets.get_total(self.config["stake_currency"]))
        else:
            # Simulated cash books open entry fees through Freqtrade's net trade PnL.
            equity = float(self.wallets.get_total_stake_amount())
            if "available_capital" not in self.config:
                equity /= float(self.config.get("tradable_balance_ratio", 1.0))
        exposure = 0.0
        risk = 0.0
        trades = Trade.get_open_trades()
        for trade in trades:
            candles = self._closed(trade.pair, current_time)
            mark = float(candles["close"].iat[-1]) if not candles.empty else trade.open_rate
            if trade.amount > 0:
                equity += float(trade.calc_close_trade_value(mark) if live_cash else
                                trade.calculate_profit(mark).profit_abs)
                exposure += float(trade.amount * mark)
            else:
                # Reserve pending orders without anchoring their stop before the actual fill.
                exposure += float(trade.stake_amount)
            state = self._read_state(trade)
            risk += trade.stake_amount * (
                state["risk_fraction"] if state else abs(self.stoploss) + self._costs()
            )
        return max(equity, 0.0), exposure, len(trades), risk

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float, proposed_stake: float,
        min_stake: float | None, max_stake: float, leverage: float, entry_tag: str | None,
        side: str, **kwargs,
    ) -> float:
        if self._quality_blocked(current_time):
            return 0.0
        candles = self._closed(pair, current_time)
        if candles.empty or side != "long" or leverage != 1:
            return 0.0
        candle = self._last_candle(pair, candles)
        if self._quality_bar_blocked(candle["date"]):
            return 0.0
        atr = float(candle["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return 0.0
        p = self._params()
        equity, exposure, count, risk = self._portfolio(current_time)
        risk_per_trade = float(p.get("risk_per_trade", 0.0025))
        guard = self._guard_scale(equity)
        scale = float(candle["downside_scale"]) if p.get(
            "enable_downside_scale", False
        ) else 1.0
        risk_budget = min(risk_per_trade * equity * scale * guard,
                          max(2 * risk_per_trade * equity * guard - risk, 0.0))
        stake = min(max_stake, float(p.get("notional_cap", 0.35)) * equity,
                    max(float(p.get("gross_cap", 0.70)) * equity - exposure, 0.0),
                    risk_budget / (self._distance(atr, current_rate) + self._costs()))
        liquidity_cap = self._liquidity_cap(candle)
        if liquidity_cap is not None:
            stake = min(stake, liquidity_cap)
        return 0.0 if count >= 2 or stake < (min_stake or 0) else float(stake)

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
        current_time: datetime, entry_tag: str | None, side: str, **kwargs,
    ) -> bool:
        if self._quality_blocked(current_time):
            return False
        candles = self._closed(pair, current_time)
        if candles.empty or side != "long":
            return False
        candle = self._last_candle(pair, candles)
        if not self._boundary(current_time, candle):
            return False
        if self._quality_bar_blocked(candle["date"]):
            return False
        if not np.isfinite(candle["atr"]) or candle["atr"] <= 0:
            return False
        liquidity_cap = self._liquidity_cap(candle)
        if liquidity_cap is not None and (liquidity_cap <= 0 or amount * rate > liquidity_cap + 1e-8):
            return False
        p = self._params()
        equity, exposure, count, risk = self._portfolio(current_time)
        guard = self._guard_scale(equity)
        if guard == 0:
            return False
        risk_limit = float(p.get("risk_per_trade", 0.0025)) * equity * guard
        scale = float(candle["downside_scale"]) if p.get("enable_downside_scale", False) else 1.0
        requested_risk = amount * rate * (self._distance(float(candle["atr"]), rate)
                                        + self._costs())
        return count < 2 and amount * rate <= float(p.get("notional_cap", 0.35)) * equity + 1e-8 and (
            exposure + amount * rate <= float(p.get("gross_cap", 0.70)) * equity + 1e-8
        ) and (
            requested_risk <= risk_limit * scale + 1e-8
            and requested_risk + risk <= 2 * risk_limit + 1e-8
        )

    def _state(self, pair: str, trade: Trade) -> dict | None:
        if trade.amount <= 0:
            return None
        state = self._read_state(trade)
        if state:
            return state
        candles = self._closed(pair, trade.open_date_utc)
        if candles.empty:
            return None
        candle = self._last_candle(pair, candles)
        atr = float(candle["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None
        distance = self._distance(atr, trade.open_rate)
        state = {"entry_atr": atr, "highest_close": float(trade.open_rate),
                 "stop": float(trade.open_rate * (1 - distance)),
                 "risk_fraction": float(distance + self._costs()),
                 "range_target": float(candle["range_mid"]),
                 "last_candle": candle["date"].isoformat()}
        self._save_state(trade, state, initial=True)
        return state

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs,
    ) -> None:
        if order.ft_order_side == trade.entry_side:
            self._state(pair, trade)

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, after_fill: bool, **kwargs,
    ) -> float | None:
        state = self._state(pair, trade)
        if state is None:
            return None
        candles = self._closed(pair, current_time)
        if candles.empty:
            return stoploss_from_absolute(state["stop"], current_rate, leverage=1)
        previous_time = pd.Timestamp(state["last_candle"])
        if candles["date"].iat[-1] > previous_time:
            new = candles.loc[candles["date"] > previous_time]
            p = self._params()
            state["highest_close"] = max(state["highest_close"], float(new["close"].max()))
            if state["highest_close"] >= trade.open_rate + state["entry_atr"] * float(
                p.get("trail_activate_atr", 1.0)
            ):
                state["stop"] = max(state["stop"], state["highest_close"]
                                    - state["entry_atr"] * float(p.get("trail_atr", 3.0)))
            state["last_candle"] = new["date"].iat[-1].isoformat()
            self._save_state(trade, state)
        return stoploss_from_absolute(state["stop"], current_rate, is_short=False, leverage=1)

    def custom_exit(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs,
    ) -> str | None:
        state = self._state(pair, trade)
        # A gap can leave a newly tightened stop above this detail candle's high.
        # Freqtrade cannot install that stop; exit at the available price instead.
        if state and current_rate <= state["stop"]:
            return "protective_stop_gap"
        candles = self._closed(pair, current_time)
        if candles.empty:
            return None
        candle = self._last_candle(pair, candles)
        if not self._boundary(current_time, candle):
            return None
        p = self._params()
        if state and candle["close"] <= state["stop"]:
            return "closed_stop_cross"
        if (pd.Timestamp(current_time) - pd.Timestamp(trade.open_date_utc)).total_seconds() >= (
            float(p.get("time_exit_hours", 8)) * 3600
        ):
            return "time_exit"
        if (pd.Timestamp(current_time) - pd.Timestamp(trade.open_date_utc)).total_seconds() < 1800:
            return None
        engine = trade.enter_tag or p.get("engine", self.default_engine)
        if engine == "range" and state and candle["close"] >= state["range_target"]:
            return "range_reclaimed"
        if engine == "pullback" and candle["close"] < candle["ema_slow"]:
            return "trend_failed"
        if engine == "breakout" and candle["close"] < candle["channel_low"]:
            return "channel_failed"
        return None

