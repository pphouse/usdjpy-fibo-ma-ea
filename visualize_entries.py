"""
エントリーポイント可視化スクリプト
時系列チャート上にエントリー/決済ポイント、フィボナッチレベル、MAを表示
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class EntryVisualizer:
    """エントリーポイント可視化クラス"""

    def __init__(self):
        self.params = {
            'fibo_lookback': 50,
            'ma_periods': [20, 50, 100],
            'overlap_tolerance_pips': 35,
            'take_profit_pips': 100,
            'stop_loss_pips': 100,
            'nanpin_interval_pips': 20,
            'lot_ratios': [1, 3, 3, 5],
            'max_nanpin_count': 3,
            'use_trend_filter': True,
        }
        self.trades = []
        self.df = None

    def generate_data(self, days=365):
        """リアルなUSDJPYデータを生成"""
        np.random.seed(42)
        n_bars = days * 6
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4h')

        base_price = 145.0
        trend = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 5
        noise = np.cumsum(np.random.normal(0, 0.003 * base_price, n_bars))
        prices = base_price + trend + noise

        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['high'] = df['close'] + np.abs(np.random.normal(0, 0.2, n_bars))
        df['low'] = df['close'] - np.abs(np.random.normal(0, 0.2, n_bars))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        return df

    def calculate_ma(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_fibonacci(self, df, idx, lookback):
        if idx < lookback:
            return None, None, 0, None, None

        window = df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.5:
            return None, None, 0, None, None

        if high_idx > low_idx:
            trend = 1  # 上昇トレンド（安値→高値の順）
            fibo_382 = high - range_val * 0.382
            fibo_618 = high - range_val * 0.618
        else:
            trend = -1  # 下降トレンド（高値→安値の順）
            fibo_382 = low + range_val * 0.382
            fibo_618 = low + range_val * 0.618

        return fibo_382, fibo_618, trend, high, low

    def check_entry(self, df, idx, fibo_382, fibo_618, trend):
        """エントリー条件チェック"""
        close = df['close'].iloc[idx]
        pip_value = 0.01
        tolerance = self.params['overlap_tolerance_pips'] * pip_value

        fibo_hit = None
        fibo_price = None
        if abs(close - fibo_382) < tolerance:
            fibo_hit = "38.2%"
            fibo_price = fibo_382
        elif abs(close - fibo_618) < tolerance:
            fibo_hit = "61.8%"
            fibo_price = fibo_618

        if fibo_hit is None:
            return None

        ma_hit = None
        ma_price = None
        for period in self.params['ma_periods']:
            ma_col = f'ma_{period}'
            if ma_col in df.columns:
                ma_val = df[ma_col].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = period
                    ma_price = ma_val
                    break

        if ma_hit is None:
            return None

        if self.params['use_trend_filter'] and 'ma_200' in df.columns:
            trend_ma = df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    return None
                if trend == -1 and close > trend_ma:
                    return None

        return {
            'direction': 'BUY' if trend == 1 else 'SELL',
            'fibo_level': fibo_hit,
            'fibo_price': fibo_price,
            'ma_period': ma_hit,
            'ma_price': ma_price,
            'entry_price': close
        }

    def simulate_trade(self, df, entry_idx, entry_info):
        """トレードシミュレーション"""
        direction = entry_info['direction']
        pip_value = 0.01
        lot_ratios = self.params['lot_ratios']

        entries = [{
            'idx': entry_idx,
            'price': entry_info['entry_price'],
            'lot_ratio': lot_ratios[0],
            'time': df.index[entry_idx],
            'type': 'ENTRY'
        }]

        nanpin_count = 0
        last_entry_price = entry_info['entry_price']

        for i in range(entry_idx + 1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]

            total_lot = sum(e['lot_ratio'] for e in entries)
            avg_price = sum(e['price'] * e['lot_ratio'] for e in entries) / total_lot

            tp_dist = self.params['take_profit_pips'] * pip_value
            sl_dist = self.params['stop_loss_pips'] * pip_value

            if direction == 'BUY':
                tp_price = avg_price + tp_dist
                sl_price = avg_price - sl_dist
            else:
                tp_price = avg_price - tp_dist
                sl_price = avg_price + sl_dist

            if direction == 'BUY':
                if current_high >= tp_price:
                    return self._create_result(entries, df, i, tp_price, 'WIN', entry_info, avg_price, tp_price, sl_price)
                elif current_low <= sl_price:
                    return self._create_result(entries, df, i, sl_price, 'LOSS', entry_info, avg_price, tp_price, sl_price)
            else:
                if current_low <= tp_price:
                    return self._create_result(entries, df, i, tp_price, 'WIN', entry_info, avg_price, tp_price, sl_price)
                elif current_high >= sl_price:
                    return self._create_result(entries, df, i, sl_price, 'LOSS', entry_info, avg_price, tp_price, sl_price)

            if nanpin_count < self.params['max_nanpin_count']:
                nanpin_dist = self.params['nanpin_interval_pips'] * pip_value
                if direction == 'BUY':
                    distance = last_entry_price - current_close
                else:
                    distance = current_close - last_entry_price

                if distance >= nanpin_dist:
                    nanpin_count += 1
                    lot_ratio = lot_ratios[min(nanpin_count, len(lot_ratios)-1)]
                    entries.append({
                        'idx': i,
                        'price': current_close,
                        'lot_ratio': lot_ratio,
                        'time': df.index[i],
                        'type': f'NANPIN_{nanpin_count}'
                    })
                    last_entry_price = current_close

        return None

    def _create_result(self, entries, df, exit_idx, exit_price, result, entry_info, avg_price, tp_price, sl_price):
        """トレード結果作成"""
        total_lot = sum(e['lot_ratio'] for e in entries)
        pip_value = 0.01

        if entry_info['direction'] == 'BUY':
            pips = (exit_price - avg_price) / pip_value
        else:
            pips = (avg_price - exit_price) / pip_value

        weighted_pips = pips * (total_lot / self.params['lot_ratios'][0])

        return {
            'entry_time': entries[0]['time'],
            'exit_time': df.index[exit_idx],
            'exit_idx': exit_idx,
            'direction': entry_info['direction'],
            'avg_entry_price': avg_price,
            'exit_price': exit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'pips': round(pips, 1),
            'weighted_pips': round(weighted_pips, 1),
            'result': result,
            'nanpin_count': len(entries) - 1,
            'total_lots': total_lot,
            'fibo_level': entry_info['fibo_level'],
            'fibo_price': entry_info['fibo_price'],
            'ma_period': entry_info['ma_period'],
            'ma_price': entry_info['ma_price'],
            'entries': entries
        }

    def run_backtest(self, df=None):
        """バックテスト実行"""
        if df is None:
            df = self.generate_data()

        for period in self.params['ma_periods']:
            df[f'ma_{period}'] = self.calculate_ma(df, period)
        df['ma_200'] = self.calculate_ma(df, 200)

        self.trades = []
        self.df = df
        lookback = self.params['fibo_lookback']

        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(df) - 100):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend, fibo_high, fibo_low = self.calculate_fibonacci(df, idx, lookback)
            if trend == 0:
                continue

            entry = self.check_entry(df, idx, fibo_382, fibo_618, trend)
            if entry is None:
                continue

            result = self.simulate_trade(df, idx, entry)
            if result:
                result['fibo_high'] = fibo_high
                result['fibo_low'] = fibo_low
                result['fibo_382'] = fibo_382
                result['fibo_618'] = fibo_618
                result['trend'] = trend
                self.trades.append(result)
                in_trade = True
                last_exit_idx = result['exit_idx']

        print(f"Total trades: {len(self.trades)}")
        return self.trades

    def visualize_overview(self, save_path=None):
        """全期間の概要チャート"""
        if not self.trades or self.df is None:
            print("No trades to visualize")
            return

        df = self.df
        fig, axes = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
        fig.suptitle('USDJPY 4H - Entry Points Overview\n'
                     f'TP={self.params["take_profit_pips"]}pips, '
                     f'SL={self.params["stop_loss_pips"]}pips, '
                     f'Nanpin Interval={self.params["nanpin_interval_pips"]}pips, '
                     f'Tolerance=±{self.params["overlap_tolerance_pips"]}pips',
                     fontsize=14, fontweight='bold')

        ax1 = axes[0]
        ax1.plot(df.index, df['close'], 'gray', linewidth=0.8, alpha=0.7, label='Price')
        ax1.plot(df.index, df['ma_20'], 'blue', linewidth=0.8, alpha=0.6, label='MA20')
        ax1.plot(df.index, df['ma_50'], 'orange', linewidth=0.8, alpha=0.6, label='MA50')
        ax1.plot(df.index, df['ma_100'], 'purple', linewidth=0.8, alpha=0.6, label='MA100')
        ax1.plot(df.index, df['ma_200'], 'red', linewidth=1.2, alpha=0.8, label='MA200 (Trend)')

        for trade in self.trades:
            entry_time = trade['entry_time']
            exit_time = trade['exit_time']
            entry_price = trade['entries'][0]['price']
            exit_price = trade['exit_price']
            is_win = trade['result'] == 'WIN'
            is_buy = trade['direction'] == 'BUY'

            marker = '^' if is_buy else 'v'
            entry_color = 'green' if is_buy else 'red'

            ax1.scatter(entry_time, entry_price, marker=marker, s=100, c=entry_color,
                       edgecolors='black', linewidth=0.5, zorder=5)

            for i, e in enumerate(trade['entries'][1:], 1):
                ax1.scatter(e['time'], e['price'], marker=marker, s=60, c='yellow',
                           edgecolors=entry_color, linewidth=1, zorder=5)

            exit_color = 'lime' if is_win else 'darkred'
            exit_marker = 'o'
            ax1.scatter(exit_time, exit_price, marker=exit_marker, s=80, c=exit_color,
                       edgecolors='black', linewidth=0.5, zorder=5)

        ax1.set_ylabel('Price (JPY)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax2 = axes[1]
        df_trades = pd.DataFrame(self.trades)
        cumulative = df_trades['weighted_pips'].cumsum()
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.4)
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.4)
        ax2.plot(df_trades['entry_time'], cumulative, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Cumulative Pips')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()

    def visualize_trade_detail(self, trade_idx, save_path=None):
        """個別トレードの詳細チャート"""
        if trade_idx >= len(self.trades):
            print(f"Trade index {trade_idx} out of range")
            return

        trade = self.trades[trade_idx]
        df = self.df

        entry_idx = df.index.get_loc(trade['entry_time'])
        exit_idx = trade['exit_idx']

        margin = 30
        start_idx = max(0, entry_idx - margin)
        end_idx = min(len(df), exit_idx + margin)
        chart_df = df.iloc[start_idx:end_idx].copy()

        fig, ax = plt.subplots(figsize=(16, 10))

        result_str = "WIN" if trade['result'] == 'WIN' else 'LOSS'
        result_color = 'green' if trade['result'] == 'WIN' else 'red'

        title = (f"Trade #{trade_idx + 1} - {trade['direction']} - {result_str} "
                f"({trade['weighted_pips']:+.1f} pips weighted)\n"
                f"Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
                f"Exit: {trade['exit_time'].strftime('%Y-%m-%d %H:%M')} | "
                f"Nanpin: {trade['nanpin_count']}x | "
                f"Fibo: {trade['fibo_level']} + MA{trade['ma_period']}")
        ax.set_title(title, fontsize=12, fontweight='bold', color=result_color)

        for i in range(len(chart_df)):
            row = chart_df.iloc[i]
            date = chart_df.index[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']

            color = 'green' if c >= o else 'red'
            ax.plot([date, date], [l, h], color=color, linewidth=0.8)
            ax.plot([date, date], [min(o, c), max(o, c)], color=color, linewidth=3)

        ax.plot(chart_df.index, chart_df['ma_20'], 'blue', linewidth=1, alpha=0.7, label='MA20')
        ax.plot(chart_df.index, chart_df['ma_50'], 'orange', linewidth=1, alpha=0.7, label='MA50')
        ax.plot(chart_df.index, chart_df['ma_100'], 'purple', linewidth=1, alpha=0.7, label='MA100')
        ax.plot(chart_df.index, chart_df['ma_200'], 'red', linewidth=1.5, alpha=0.8, label='MA200')

        fibo_high = trade['fibo_high']
        fibo_low = trade['fibo_low']
        fibo_382 = trade['fibo_382']
        fibo_618 = trade['fibo_618']

        ax.axhline(y=fibo_high, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=fibo_low, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=fibo_382, color='gold', linestyle='-', linewidth=1.5, alpha=0.8, label='Fibo 38.2%')
        ax.axhline(y=fibo_618, color='darkorange', linestyle='-', linewidth=1.5, alpha=0.8, label='Fibo 61.8%')

        ax.text(chart_df.index[-1], fibo_high, ' 100%', va='center', fontsize=9, color='gray')
        ax.text(chart_df.index[-1], fibo_low, ' 0%', va='center', fontsize=9, color='gray')
        ax.text(chart_df.index[-1], fibo_382, ' 38.2%', va='center', fontsize=9, color='gold')
        ax.text(chart_df.index[-1], fibo_618, ' 61.8%', va='center', fontsize=9, color='darkorange')

        is_buy = trade['direction'] == 'BUY'
        marker = '^' if is_buy else 'v'
        entry_color = 'lime' if is_buy else 'magenta'

        for i, e in enumerate(trade['entries']):
            if i == 0:
                ax.scatter(e['time'], e['price'], marker=marker, s=200, c=entry_color,
                          edgecolors='black', linewidth=1.5, zorder=10, label='Entry')
                ax.annotate(f"ENTRY\n{e['price']:.3f}\nLot: {e['lot_ratio']}",
                           (e['time'], e['price']),
                           xytext=(10, 20 if is_buy else -30),
                           textcoords='offset points',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.scatter(e['time'], e['price'], marker=marker, s=150, c='yellow',
                          edgecolors=entry_color, linewidth=2, zorder=10)
                ax.annotate(f"N{i}\n{e['price']:.3f}\nLot: {e['lot_ratio']}",
                           (e['time'], e['price']),
                           xytext=(10, 15 if is_buy else -25),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        avg_price = trade['avg_entry_price']
        ax.axhline(y=avg_price, color='blue', linestyle=':', linewidth=2, alpha=0.8)
        ax.text(chart_df.index[0], avg_price, f' Avg: {avg_price:.3f}', va='center',
               fontsize=9, color='blue', fontweight='bold')

        tp_price = trade['tp_price']
        sl_price = trade['sl_price']
        ax.axhline(y=tp_price, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=sl_price, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(chart_df.index[0], tp_price, f' TP: {tp_price:.3f}', va='center',
               fontsize=9, color='green', fontweight='bold')
        ax.text(chart_df.index[0], sl_price, f' SL: {sl_price:.3f}', va='center',
               fontsize=9, color='red', fontweight='bold')

        exit_color = 'lime' if trade['result'] == 'WIN' else 'red'
        ax.scatter(trade['exit_time'], trade['exit_price'], marker='X', s=200, c=exit_color,
                  edgecolors='black', linewidth=1.5, zorder=10, label='Exit')
        ax.annotate(f"EXIT ({result_str})\n{trade['exit_price']:.3f}\n{trade['pips']:+.1f}pips",
                   (trade['exit_time'], trade['exit_price']),
                   xytext=(10, -30 if trade['result'] == 'WIN' else 20),
                   textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=exit_color, alpha=0.7))

        ax.set_ylabel('Price (JPY)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)

        info_text = (f"Direction: {trade['direction']}\n"
                    f"Fibo Level: {trade['fibo_level']}\n"
                    f"MA Period: {trade['ma_period']}\n"
                    f"Nanpin Count: {trade['nanpin_count']}\n"
                    f"Total Lots: {trade['total_lots']}\n"
                    f"Pips: {trade['pips']:+.1f}\n"
                    f"Weighted Pips: {trade['weighted_pips']:+.1f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', bbox=props)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()

    def visualize_multiple_trades(self, num_trades=6, save_path=None):
        """複数トレードの詳細を一覧表示"""
        if not self.trades:
            print("No trades to visualize")
            return

        num_trades = min(num_trades, len(self.trades))

        win_trades = [t for t in self.trades if t['result'] == 'WIN']
        loss_trades = [t for t in self.trades if t['result'] == 'LOSS']

        selected = []
        if win_trades:
            selected.append(('WIN (Best)', max(win_trades, key=lambda x: x['weighted_pips'])))
        if loss_trades:
            selected.append(('LOSS (Worst)', min(loss_trades, key=lambda x: x['weighted_pips'])))

        nanpin_trades = [t for t in self.trades if t['nanpin_count'] > 0]
        if nanpin_trades:
            selected.append(('With Nanpin', nanpin_trades[0]))

        no_nanpin_trades = [t for t in self.trades if t['nanpin_count'] == 0]
        if no_nanpin_trades:
            selected.append(('No Nanpin', no_nanpin_trades[0]))

        for i, trade in enumerate(self.trades[:2]):
            if len(selected) < num_trades:
                selected.append((f'Trade #{i+1}', trade))

        n_cols = 2
        n_rows = (len(selected) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        fig.suptitle('Sample Trade Details', fontsize=16, fontweight='bold')

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (label, trade) in enumerate(selected):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            df = self.df
            entry_idx = df.index.get_loc(trade['entry_time'])
            exit_idx = trade['exit_idx']

            margin = 20
            start_idx = max(0, entry_idx - margin)
            end_idx = min(len(df), exit_idx + margin)
            chart_df = df.iloc[start_idx:end_idx].copy()

            ax.plot(chart_df.index, chart_df['close'], 'gray', linewidth=1, alpha=0.8)

            fibo_382 = trade['fibo_382']
            fibo_618 = trade['fibo_618']
            ax.axhline(y=fibo_382, color='gold', linestyle='-', linewidth=1, alpha=0.7)
            ax.axhline(y=fibo_618, color='darkorange', linestyle='-', linewidth=1, alpha=0.7)

            is_buy = trade['direction'] == 'BUY'
            marker = '^' if is_buy else 'v'

            for i, e in enumerate(trade['entries']):
                color = 'lime' if i == 0 else 'yellow'
                ax.scatter(e['time'], e['price'], marker=marker, s=80, c=color,
                          edgecolors='black', linewidth=1, zorder=5)

            exit_color = 'lime' if trade['result'] == 'WIN' else 'red'
            ax.scatter(trade['exit_time'], trade['exit_price'], marker='X', s=100,
                      c=exit_color, edgecolors='black', linewidth=1, zorder=5)

            result_color = 'green' if trade['result'] == 'WIN' else 'red'
            title = (f"{label}: {trade['direction']} | "
                    f"{trade['result']} ({trade['weighted_pips']:+.1f}pips) | "
                    f"N{trade['nanpin_count']}")
            ax.set_title(title, fontsize=10, fontweight='bold', color=result_color)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        for idx in range(len(selected), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()

    def visualize_algorithm_explanation(self, save_path=None):
        """アルゴリズムの視覚的説明"""
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Algorithm Visualization - Fibonacci + MA + Nanpin Strategy',
                    fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title('1. Fibonacci Level Calculation', fontsize=12, fontweight='bold')

        x = np.linspace(0, 50, 100)
        y = 100 + 10 * np.sin(x/5) + x/5

        ax1.plot(x, y, 'b-', linewidth=2)
        high_idx = np.argmax(y)
        low_idx = np.argmin(y)
        high_val = y[high_idx]
        low_val = y[low_idx]

        ax1.axhline(y=high_val, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=low_val, color='gray', linestyle='--', alpha=0.5)

        range_val = high_val - low_val
        fibo_382 = high_val - range_val * 0.382
        fibo_618 = high_val - range_val * 0.618

        ax1.axhline(y=fibo_382, color='gold', linestyle='-', linewidth=2)
        ax1.axhline(y=fibo_618, color='darkorange', linestyle='-', linewidth=2)

        ax1.scatter([x[high_idx]], [high_val], s=100, c='red', marker='v', zorder=5)
        ax1.scatter([x[low_idx]], [low_val], s=100, c='green', marker='^', zorder=5)

        ax1.text(52, high_val, '100% (High)', va='center', fontsize=10)
        ax1.text(52, low_val, '0% (Low)', va='center', fontsize=10)
        ax1.text(52, fibo_382, '38.2%', va='center', fontsize=10, color='goldenrod')
        ax1.text(52, fibo_618, '61.8%', va='center', fontsize=10, color='darkorange')
        ax1.set_xlim(0, 65)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title('2. Entry Condition (Fibo + MA Overlap)', fontsize=12, fontweight='bold')

        x = np.linspace(0, 50, 100)
        price = 145 + 2 * np.sin(x/8)
        ma20 = 145 + 1.5 * np.sin((x-5)/8)
        ma50 = 145 + 1 * np.sin((x-10)/8)

        ax2.plot(x, price, 'b-', linewidth=2, label='Price')
        ax2.plot(x, ma20, 'green', linewidth=1.5, alpha=0.8, label='MA20')
        ax2.plot(x, ma50, 'orange', linewidth=1.5, alpha=0.8, label='MA50')

        fibo_level = 144.5
        ax2.axhline(y=fibo_level, color='gold', linestyle='-', linewidth=2, label='Fibo Level')

        tolerance = 0.35
        ax2.fill_between([0, 50], fibo_level - tolerance, fibo_level + tolerance,
                        color='yellow', alpha=0.3, label=f'Tolerance (±{int(tolerance*100)}pips)')

        entry_x = 25
        entry_y = price[50]
        ax2.scatter([entry_x], [entry_y], s=150, c='lime', marker='^',
                   edgecolors='black', linewidth=2, zorder=10)
        ax2.annotate('ENTRY!\nPrice near Fibo\nAND near MA',
                    (entry_x, entry_y), xytext=(30, 145.5),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'),
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('3. Nanpin Logic (Lot Ratio 1:3:3:5)', fontsize=12, fontweight='bold')

        x = np.linspace(0, 40, 100)
        price = 145 - 0.08 * x + 0.5 * np.sin(x/3)

        ax3.plot(x, price, 'b-', linewidth=2, label='Price (falling)')

        entries = [
            (5, 144.8, 1, 'ENTRY (Lot: 1)'),
            (15, 144.2, 3, 'Nanpin 1 (Lot: 3)'),
            (25, 143.6, 3, 'Nanpin 2 (Lot: 3)'),
            (35, 143.0, 5, 'Nanpin 3 (Lot: 5)')
        ]

        colors = ['lime', 'yellow', 'yellow', 'yellow']
        for i, (ex, ey, lot, label) in enumerate(entries):
            ax3.scatter([ex], [ey], s=100 + lot*20, c=colors[i], marker='^',
                       edgecolors='black', linewidth=1.5, zorder=5)
            ax3.annotate(label, (ex, ey), xytext=(ex+2, ey+0.2),
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        total_lot = 1 + 3 + 3 + 5
        avg_price = (144.8*1 + 144.2*3 + 143.6*3 + 143.0*5) / total_lot
        ax3.axhline(y=avg_price, color='blue', linestyle=':', linewidth=2)
        ax3.text(0, avg_price + 0.1, f'Avg Entry: {avg_price:.2f}', fontsize=10, color='blue')

        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('4. Exit Logic (TP/SL from Avg Price)', fontsize=12, fontweight='bold')

        x = np.linspace(0, 50, 100)
        price = avg_price - 0.5 + 1.5 * np.sin(x/8) + 0.02 * x

        ax4.plot(x, price, 'b-', linewidth=2, label='Price')

        ax4.axhline(y=avg_price, color='blue', linestyle=':', linewidth=2, label='Avg Entry')
        tp_price = avg_price + 1.0
        sl_price = avg_price - 1.0
        ax4.axhline(y=tp_price, color='green', linestyle='--', linewidth=2, label='TP (+100pips)')
        ax4.axhline(y=sl_price, color='red', linestyle='--', linewidth=2, label='SL (-100pips)')

        ax4.fill_between([0, 50], avg_price, tp_price, color='green', alpha=0.1)
        ax4.fill_between([0, 50], sl_price, avg_price, color='red', alpha=0.1)

        tp_hit_x = 45
        ax4.scatter([tp_hit_x], [tp_price], s=150, c='lime', marker='X',
                   edgecolors='black', linewidth=2, zorder=10)
        ax4.annotate('TP HIT!\nWIN', (tp_hit_x, tp_price), xytext=(tp_hit_x-10, tp_price+0.3),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax4.legend(loc='lower right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close()


def main():
    print("="*65)
    print("    Entry Point Visualization")
    print("    USDJPY 4H - Fibonacci + MA + Nanpin Strategy")
    print("="*65)

    viz = EntryVisualizer()
    viz.run_backtest()

    print("\nGenerating visualizations...")

    viz.visualize_algorithm_explanation(
        save_path='/Users/naoto/ドル円/algorithm_visualization.png'
    )

    viz.visualize_overview(
        save_path='/Users/naoto/ドル円/entry_points_overview.png'
    )

    viz.visualize_multiple_trades(
        num_trades=6,
        save_path='/Users/naoto/ドル円/sample_trades.png'
    )

    print("\nGenerating individual trade charts...")
    for i in range(min(3, len(viz.trades))):
        viz.visualize_trade_detail(
            trade_idx=i,
            save_path=f'/Users/naoto/ドル円/trade_detail_{i+1}.png'
        )

    print("\n" + "="*65)
    print("Complete! Generated files:")
    print("  - algorithm_visualization.png  : Algorithm explanation")
    print("  - entry_points_overview.png    : All entry points on chart")
    print("  - sample_trades.png            : Sample trade details")
    print("  - trade_detail_N.png           : Individual trade charts")
    print("="*65)


if __name__ == "__main__":
    main()
