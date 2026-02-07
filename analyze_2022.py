"""
2022年の負け原因を分析
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class Analyzer2022:
    def __init__(self):
        self.df = None
        self.trades = None

    def generate_10year_data(self):
        """10年分のデータを生成（同じシード）"""
        print("データ生成中...")
        np.random.seed(42)

        days = 365 * 10
        n_bars = days * 6
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4h')

        base_price = 110.0
        long_trend = np.linspace(0, 40, n_bars)
        mid_cycle = np.sin(np.linspace(0, 20*np.pi, n_bars)) * 8
        noise = np.cumsum(np.random.normal(0, 0.002 * base_price, n_bars))
        noise = noise - np.linspace(noise[0], noise[-1], n_bars)

        prices = base_price + long_trend + mid_cycle + noise

        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['high'] = df['close'] + np.abs(np.random.normal(0, 0.15, n_bars))
        df['low'] = df['close'] - np.abs(np.random.normal(0, 0.15, n_bars))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        for period in [20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()

        self.df = df
        return df

    def calculate_fibonacci(self, idx, lookback=50):
        if idx < lookback:
            return None, None, 0

        window = self.df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.3:
            return None, None, 0

        if high_idx > low_idx:
            trend = 1
            fibo_382 = high - range_val * 0.382
            fibo_618 = high - range_val * 0.618
        else:
            trend = -1
            fibo_382 = low + range_val * 0.382
            fibo_618 = low + range_val * 0.618

        return fibo_382, fibo_618, trend

    def run_backtest_with_details(self, tp_pips=70, sl_pips=80, nanpin_interval=20, tolerance_pips=35):
        """詳細情報付きバックテスト"""
        if self.df is None:
            self.generate_10year_data()

        print(f"バックテスト実行中... TP={tp_pips}, SL={sl_pips}")

        pip_value = 0.01
        lot_ratios = [1, 3, 3, 5]
        max_nanpin = 3
        lookback = 50

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(self.df) - 500):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend = self.calculate_fibonacci(idx, lookback)
            if trend == 0:
                continue

            close = self.df['close'].iloc[idx]
            tolerance = tolerance_pips * pip_value

            fibo_hit = None
            if abs(close - fibo_382) < tolerance:
                fibo_hit = "38.2%"
            elif abs(close - fibo_618) < tolerance:
                fibo_hit = "61.8%"
            if fibo_hit is None:
                continue

            ma_hit = None
            for period in [20, 50, 100]:
                ma_val = self.df[f'ma_{period}'].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = period
                    break
            if ma_hit is None:
                continue

            trend_ma = self.df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    continue
                if trend == -1 and close > trend_ma:
                    continue

            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{'price': close, 'lot': lot_ratios[0], 'time': self.df.index[idx]}]
            last_entry_price = close
            nanpin_count = 0
            entry_price = close

            for i in range(idx + 1, min(idx + 1000, len(self.df))):
                current_high = self.df['high'].iloc[i]
                current_low = self.df['low'].iloc[i]
                current_close = self.df['close'].iloc[i]

                total_lot = sum(e['lot'] for e in entries)
                avg_price = sum(e['price'] * e['lot'] for e in entries) / total_lot

                tp_dist = tp_pips * pip_value
                sl_dist = sl_pips * pip_value

                if direction == 'BUY':
                    tp_price = avg_price + tp_dist
                    sl_price = avg_price - sl_dist
                else:
                    tp_price = avg_price - tp_dist
                    sl_price = avg_price + sl_dist

                result = None
                if direction == 'BUY':
                    if current_high >= tp_price:
                        pips = tp_pips
                        result = 'WIN'
                    elif current_low <= sl_price:
                        pips = -sl_pips
                        result = 'LOSS'
                else:
                    if current_low <= tp_price:
                        pips = tp_pips
                        result = 'WIN'
                    elif current_high >= sl_price:
                        pips = -sl_pips
                        result = 'LOSS'

                if result:
                    weighted_pips = pips * total_lot

                    # 価格の動きを計算
                    if direction == 'BUY':
                        max_favorable = (self.df.loc[entries[0]['time']:self.df.index[i], 'high'].max() - entry_price) / pip_value
                        max_adverse = (entry_price - self.df.loc[entries[0]['time']:self.df.index[i], 'low'].min()) / pip_value
                    else:
                        max_favorable = (entry_price - self.df.loc[entries[0]['time']:self.df.index[i], 'low'].min()) / pip_value
                        max_adverse = (self.df.loc[entries[0]['time']:self.df.index[i], 'high'].max() - entry_price) / pip_value

                    trades.append({
                        'entry_time': entries[0]['time'],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'avg_price': avg_price,
                        'exit_price': tp_price if result == 'WIN' else sl_price,
                        'pips': pips,
                        'weighted_pips': weighted_pips,
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot,
                        'fibo_level': fibo_hit,
                        'ma_period': ma_hit,
                        'max_favorable_excursion': max_favorable,
                        'max_adverse_excursion': max_adverse,
                        'holding_bars': i - idx
                    })
                    last_exit_idx = i
                    in_trade = True
                    break

                if nanpin_count < max_nanpin:
                    nanpin_dist = nanpin_interval * pip_value
                    if direction == 'BUY':
                        distance = last_entry_price - current_close
                    else:
                        distance = current_close - last_entry_price

                    if distance >= nanpin_dist:
                        nanpin_count += 1
                        lot = lot_ratios[min(nanpin_count, len(lot_ratios)-1)]
                        entries.append({'price': current_close, 'lot': lot, 'time': self.df.index[i]})
                        last_entry_price = current_close

        self.trades = pd.DataFrame(trades)
        self.trades['year'] = pd.to_datetime(self.trades['entry_time']).dt.year
        self.trades['month'] = pd.to_datetime(self.trades['entry_time']).dt.month
        print(f"トレード数: {len(self.trades)}")
        return self.trades

    def analyze_2022(self):
        """2022年を詳細分析"""
        trades_2022 = self.trades[self.trades['year'] == 2022].copy()
        trades_other = self.trades[self.trades['year'] != 2022].copy()

        print("\n" + "="*70)
        print("  2022年 vs 他の年 比較分析")
        print("="*70)

        # 基本統計
        print("\n【基本統計】")
        print(f"{'指標':<25} {'2022年':<20} {'他の年平均':<20}")
        print("-"*70)

        stats_2022 = {
            'trades': len(trades_2022),
            'win_rate': (trades_2022['result'] == 'WIN').mean() * 100,
            'total_pips': trades_2022['weighted_pips'].sum(),
            'avg_pips': trades_2022['weighted_pips'].mean(),
            'nanpin_avg': trades_2022['nanpin_count'].mean(),
        }

        other_years = trades_other['year'].unique()
        stats_other = {
            'trades': len(trades_other) / len(other_years),
            'win_rate': (trades_other['result'] == 'WIN').mean() * 100,
            'total_pips': trades_other.groupby('year')['weighted_pips'].sum().mean(),
            'avg_pips': trades_other['weighted_pips'].mean(),
            'nanpin_avg': trades_other['nanpin_count'].mean(),
        }

        print(f"{'トレード数':<25} {stats_2022['trades']:<20} {stats_other['trades']:<20.1f}")
        print(f"{'勝率 (%)':<25} {stats_2022['win_rate']:<20.2f} {stats_other['win_rate']:<20.2f}")
        print(f"{'合計損益 (pips)':<25} {stats_2022['total_pips']:<20.1f} {stats_other['total_pips']:<20.1f}")
        print(f"{'平均損益 (pips)':<25} {stats_2022['avg_pips']:<20.2f} {stats_other['avg_pips']:<20.2f}")
        print(f"{'平均ナンピン回数':<25} {stats_2022['nanpin_avg']:<20.2f} {stats_other['nanpin_avg']:<20.2f}")

        # 方向別分析
        print("\n【方向別分析 - 2022年】")
        for direction in ['BUY', 'SELL']:
            dir_trades = trades_2022[trades_2022['direction'] == direction]
            if len(dir_trades) > 0:
                win_rate = (dir_trades['result'] == 'WIN').mean() * 100
                total_pips = dir_trades['weighted_pips'].sum()
                print(f"  {direction}: {len(dir_trades)}トレード, 勝率{win_rate:.1f}%, 損益{total_pips:.0f}pips")

        print("\n【方向別分析 - 他の年】")
        for direction in ['BUY', 'SELL']:
            dir_trades = trades_other[trades_other['direction'] == direction]
            if len(dir_trades) > 0:
                win_rate = (dir_trades['result'] == 'WIN').mean() * 100
                total_pips = dir_trades['weighted_pips'].sum()
                print(f"  {direction}: {len(dir_trades)}トレード, 勝率{win_rate:.1f}%, 損益{total_pips:.0f}pips")

        # ナンピン回数別分析
        print("\n【ナンピン回数別 - 2022年】")
        for n in range(4):
            n_trades = trades_2022[trades_2022['nanpin_count'] == n]
            if len(n_trades) > 0:
                win_rate = (n_trades['result'] == 'WIN').mean() * 100
                total_pips = n_trades['weighted_pips'].sum()
                print(f"  N{n}: {len(n_trades)}トレード, 勝率{win_rate:.1f}%, 損益{total_pips:.0f}pips")

        print("\n【ナンピン回数別 - 他の年】")
        for n in range(4):
            n_trades = trades_other[trades_other['nanpin_count'] == n]
            if len(n_trades) > 0:
                win_rate = (n_trades['result'] == 'WIN').mean() * 100
                total_pips = n_trades['weighted_pips'].sum()
                print(f"  N{n}: {len(n_trades)}トレード, 勝率{win_rate:.1f}%, 損益{total_pips:.0f}pips")

        # 2022年の価格変動を分析
        print("\n【2022年の価格変動】")
        df_2022 = self.df[self.df.index.year == 2022]
        price_range = df_2022['high'].max() - df_2022['low'].min()
        volatility = df_2022['close'].pct_change().std() * 100
        trend = df_2022['close'].iloc[-1] - df_2022['close'].iloc[0]

        df_other = self.df[self.df.index.year != 2022]
        other_volatility = df_other['close'].pct_change().std() * 100

        print(f"  価格レンジ: {df_2022['low'].min():.2f} ~ {df_2022['high'].max():.2f} ({price_range:.2f}円)")
        print(f"  年間トレンド: {trend:+.2f}円")
        print(f"  ボラティリティ: {volatility:.4f}% (他の年平均: {other_volatility:.4f}%)")

        # 月別分析
        print("\n【2022年 月別パフォーマンス】")
        monthly = trades_2022.groupby('month').agg({
            'weighted_pips': ['sum', 'count'],
            'result': lambda x: (x == 'WIN').mean() * 100,
            'nanpin_count': 'mean'
        }).round(2)
        monthly.columns = ['損益', 'トレード数', '勝率', '平均ナンピン']
        print(monthly.to_string())

        # 大負けトレード分析
        print("\n【2022年 大負けトレード TOP5】")
        big_losses = trades_2022.nsmallest(5, 'weighted_pips')
        for i, (_, trade) in enumerate(big_losses.iterrows(), 1):
            print(f"  {i}. {trade['entry_time'].strftime('%Y-%m-%d')} {trade['direction']} "
                  f"損益:{trade['weighted_pips']:.0f}pips ナンピン:{trade['nanpin_count']}回 "
                  f"Fibo:{trade['fibo_level']} MA:{trade['ma_period']}")

        return trades_2022, stats_2022

    def visualize_2022_analysis(self, save_path):
        """2022年分析のビジュアル化"""
        trades_2022 = self.trades[self.trades['year'] == 2022]
        df_2022 = self.df[self.df.index.year == 2022]

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('2022 Year Analysis - Why Did Rank1 Lose?', fontsize=16, fontweight='bold')

        # 1. 2022年の価格チャートとトレード
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(df_2022.index, df_2022['close'], 'b-', linewidth=0.5, alpha=0.7, label='Price')
        ax1.plot(df_2022.index, df_2022['ma_200'], 'orange', linewidth=1, alpha=0.7, label='MA200')

        # トレードをプロット
        wins = trades_2022[trades_2022['result'] == 'WIN']
        losses = trades_2022[trades_2022['result'] == 'LOSS']
        ax1.scatter(wins['entry_time'], wins['entry_price'], color='green', marker='^', s=30, label='WIN', alpha=0.7)
        ax1.scatter(losses['entry_time'], losses['entry_price'], color='red', marker='v', s=30, label='LOSS', alpha=0.7)

        ax1.set_title('2022 Price Chart with Trades', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # 2. 月別パフォーマンス
        ax2 = fig.add_subplot(2, 2, 2)
        monthly = trades_2022.groupby('month')['weighted_pips'].sum()
        colors = ['green' if v > 0 else 'red' for v in monthly.values]
        ax2.bar(monthly.index, monthly.values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.set_title('2022 Monthly Performance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 方向別・ナンピン別の負け分析
        ax3 = fig.add_subplot(2, 2, 3)

        # ナンピン回数 x 方向 のヒートマップ
        pivot = trades_2022.pivot_table(
            values='weighted_pips',
            index='nanpin_count',
            columns='direction',
            aggfunc='sum',
            fill_value=0
        )
        im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_xticklabels(pivot.columns)
        ax3.set_yticklabels([f'N{i}' for i in pivot.index])
        ax3.set_xlabel('Direction')
        ax3.set_ylabel('Nanpin Count')
        ax3.set_title('2022 Pips by Direction & Nanpin', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Pips')

        # 値を表示
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = 'white' if abs(val) > 1000 else 'black'
                ax3.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=10)

        # 4. 年別比較（2022 vs 他の年）
        ax4 = fig.add_subplot(2, 2, 4)

        yearly = self.trades.groupby('year').agg({
            'weighted_pips': 'sum',
            'result': lambda x: (x == 'WIN').mean() * 100,
            'nanpin_count': 'mean'
        })

        x = range(len(yearly))
        colors = ['red' if y == 2022 else 'steelblue' for y in yearly.index]
        bars = ax4.bar(x, yearly['weighted_pips'], color=colors, alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(yearly.index, rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--')
        ax4.set_title('Yearly Performance (2022 in Red)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Pips')
        ax4.grid(True, alpha=0.3, axis='y')

        # 勝率を表示
        for bar, (year, row) in zip(bars, yearly.iterrows()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{row["result"]:.0f}%', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n分析グラフを保存しました: {save_path}")


def main():
    print("="*70)
    print("  2022年 負け原因分析")
    print("  Rank1: TP70 / SL80 / Nanpin20 / Tolerance35")
    print("="*70)

    analyzer = Analyzer2022()
    analyzer.generate_10year_data()
    analyzer.run_backtest_with_details(tp_pips=70, sl_pips=80, nanpin_interval=20, tolerance_pips=35)
    analyzer.analyze_2022()
    analyzer.visualize_2022_analysis('/Users/naoto/ドル円/analysis_2022.png')

    print("\n完了!")


if __name__ == "__main__":
    main()
