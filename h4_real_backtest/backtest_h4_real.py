"""
実データH4バックテスト
Axiory USDJPY H4 2015-2025
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


class H4RealBacktester:
    """H4実データバックテスター"""

    def __init__(self, spread_pips=0.3):
        self.spread_pips = spread_pips
        self.df = None
        self.results = {}

    def load_data(self):
        """H4データ読み込み"""
        print("H4データを読み込み中...")
        self.df = pd.read_csv(
            '/Users/naoto/ドル円/h4_real_backtest/USDJPY_H4_2015_2025.csv',
            index_col=0, parse_dates=True
        )

        # MA計算
        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

        print(f"データ: {len(self.df):,}本")
        print(f"期間: {self.df.index[0].date()} ~ {self.df.index[-1].date()}")
        print(f"価格: {self.df['low'].min():.3f} ~ {self.df['high'].max():.3f}")
        return self.df

    def calculate_fibonacci(self, idx, lookback=50):
        if idx < lookback:
            return None, None, 0, None, None

        window = self.df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.3:
            return None, None, 0, None, None

        if high_idx > low_idx:
            trend = 1
            fibo_382 = high - range_val * 0.382
            fibo_618 = high - range_val * 0.618
        else:
            trend = -1
            fibo_382 = low + range_val * 0.382
            fibo_618 = low + range_val * 0.618

        return fibo_382, fibo_618, trend, high, low

    def run_backtest(self, tp_pips, sl_pips, nanpin_interval, tolerance_pips,
                     max_nanpin=2, lot_ratios=None, name=""):
        """バックテスト実行"""
        if lot_ratios is None:
            lot_ratios = [1, 3, 3]

        if self.df is None:
            self.load_data()

        print(f"\n[{name}] TP={tp_pips}, SL={sl_pips}, Nanpin={nanpin_interval}, Tol={tolerance_pips}, MaxN={max_nanpin}")

        pip_value = 0.01
        lookback = 50
        spread = self.spread_pips

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(self.df) - 100):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend, fibo_high, fibo_low = self.calculate_fibonacci(idx, lookback)
            if trend == 0:
                continue

            close = self.df['close'].iloc[idx]
            tolerance = tolerance_pips * pip_value

            fibo_hit = None
            fibo_price = None
            if abs(close - fibo_382) < tolerance:
                fibo_hit = "38.2%"
                fibo_price = fibo_382
            elif abs(close - fibo_618) < tolerance:
                fibo_hit = "61.8%"
                fibo_price = fibo_618
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

            # トレードシミュレーション
            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{
                'idx': idx,
                'price': close,
                'lot': lot_ratios[0],
                'time': self.df.index[idx]
            }]
            last_entry_price = close
            nanpin_count = 0

            for i in range(idx + 1, min(idx + 500, len(self.df))):
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
                exit_price = None
                if direction == 'BUY':
                    if current_high >= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                    elif current_low <= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price
                else:
                    if current_low <= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                    elif current_high >= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price

                if result:
                    pips = tp_pips if result == 'WIN' else -sl_pips
                    spread_cost = sum(e['lot'] * spread for e in entries)
                    weighted_pips = pips * total_lot - spread_cost

                    trades.append({
                        'entry_time': entries[0]['time'],
                        'entry_idx': entries[0]['idx'],
                        'exit_time': self.df.index[i],
                        'exit_idx': i,
                        'direction': direction,
                        'entries': entries.copy(),
                        'avg_price': avg_price,
                        'exit_price': exit_price,
                        'pips': pips,
                        'weighted_pips': weighted_pips,
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot,
                        'fibo_hit': fibo_hit,
                        'ma_hit': ma_hit
                    })
                    last_exit_idx = i
                    in_trade = True
                    break

                # ナンピン判定
                if nanpin_count < max_nanpin:
                    nanpin_dist = nanpin_interval * pip_value
                    if direction == 'BUY':
                        distance = last_entry_price - current_close
                    else:
                        distance = current_close - last_entry_price

                    if distance >= nanpin_dist:
                        nanpin_count += 1
                        lot = lot_ratios[min(nanpin_count, len(lot_ratios)-1)]
                        entries.append({
                            'idx': i,
                            'price': current_close,
                            'lot': lot,
                            'time': self.df.index[i]
                        })
                        last_entry_price = current_close

        print(f"  トレード数: {len(trades)}")
        return trades

    def calculate_stats(self, trades, name):
        """統計計算"""
        if not trades:
            return None

        df = pd.DataFrame(trades)
        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        win_rate = wins / total * 100 if total > 0 else 0

        total_pips = df['weighted_pips'].sum()
        avg_pips = df['weighted_pips'].mean()

        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        yearly = df.groupby('year')['weighted_pips'].sum()

        cumulative = df['weighted_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        win_pips = df[df['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df[df['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else 999

        nanpin_stats = df.groupby('nanpin_count').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'weighted_pips': 'sum',
            'pips': 'count'
        }).round(2)
        nanpin_stats.columns = ['win_rate', 'total_pips', 'count']

        return {
            'name': name,
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': round(win_rate, 2),
            'total_pips': round(total_pips, 1),
            'avg_pips': round(avg_pips, 2),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'yearly': yearly,
            'nanpin_stats': nanpin_stats,
            'df_trades': df,
            'cumulative': cumulative
        }

    def run_comparison(self):
        """N2 vs N3 比較"""
        configs = [
            # H4用パラメータ（日足の約1/6なので調整）
            {'name': 'N3_TP100_SL100', 'tp': 100, 'sl': 100, 'interval': 20, 'tol': 35,
             'max_nanpin': 3, 'lots': [1, 3, 3, 5]},
            {'name': 'N2_TP100_SL100', 'tp': 100, 'sl': 100, 'interval': 20, 'tol': 35,
             'max_nanpin': 2, 'lots': [1, 3, 3]},
            {'name': 'N2_TP70_SL80', 'tp': 70, 'sl': 80, 'interval': 20, 'tol': 35,
             'max_nanpin': 2, 'lots': [1, 3, 3]},
            {'name': 'N0_TP100_SL100', 'tp': 100, 'sl': 100, 'interval': 20, 'tol': 35,
             'max_nanpin': 0, 'lots': [1]},
        ]

        for cfg in configs:
            trades = self.run_backtest(
                cfg['tp'], cfg['sl'], cfg['interval'], cfg['tol'],
                max_nanpin=cfg['max_nanpin'],
                lot_ratios=cfg['lots'],
                name=cfg['name']
            )
            stats = self.calculate_stats(trades, cfg['name'])
            if stats:
                stats['max_lot'] = sum(cfg['lots'][:cfg['max_nanpin']+1])
                self.results[cfg['name']] = stats

        return self.results

    def print_results(self):
        """結果表示"""
        print("\n" + "="*100)
        print("  H4実データバックテスト結果 (USDJPY 2015-2025, Axiory)")
        print("="*100)
        print(f"{'設定':<20}{'MaxLot':<8}{'Trades':<8}{'WR%':<8}{'Net Pips':<12}{'MaxDD':<10}{'PF':<8}{'AvgPips':<10}")
        print("-"*100)

        for name, stats in self.results.items():
            print(f"{name:<20}{stats.get('max_lot', '-'):<8}{stats['trades']:<8}"
                  f"{stats['win_rate']:<8}{stats['total_pips']:<12.0f}"
                  f"{stats['max_dd']:<10.0f}{stats['pf']:<8.2f}{stats['avg_pips']:<10.2f}")

        print("="*100)

        # ナンピン統計
        print("\n【ナンピン回数別パフォーマンス】")
        for name, stats in self.results.items():
            print(f"\n  {name}:")
            for idx, row in stats['nanpin_stats'].iterrows():
                print(f"    N{idx}: {int(row['count'])}回, 勝率{row['win_rate']:.1f}%, 損益{row['total_pips']:+,.0f}pips")

        # 年別
        print("\n【年別パフォーマンス (Net Pips)】")
        years = sorted(set().union(*[set(s['yearly'].index) for s in self.results.values()]))

        header = f"{'Year':<6}" + "".join([f"{n:<15}" for n in self.results.keys()])
        print(header)
        print("-" * len(header))

        for year in years:
            row = f"{year:<6}"
            for name, stats in self.results.items():
                val = stats['yearly'].get(year, 0)
                row += f"{val:>+12,.0f}   "
            print(row)

    def visualize(self, save_path):
        """グラフ作成"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('USDJPY H4 Real Data Backtest (Axiory 2015-2025)\nSpread: 0.3 pips',
                    fontsize=16, fontweight='bold')

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

        # 1. 価格チャート
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(self.df.index, self.df['close'], 'gray', linewidth=0.3, alpha=0.7)
        ax1.plot(self.df.index, self.df['ma_200'], 'red', linewidth=0.5, alpha=0.5)
        ax1.set_title('USDJPY H4 Price', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        # 2. 累積損益比較
        ax2 = fig.add_subplot(3, 2, 2)
        for i, (name, stats) in enumerate(self.results.items()):
            df_trades = stats['df_trades']
            cumulative = df_trades['weighted_pips'].cumsum()
            ax2.plot(df_trades['entry_time'], cumulative, color=colors[i],
                    linewidth=1.5, label=name, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Cumulative Net Pips', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 年別比較
        ax3 = fig.add_subplot(3, 2, 3)
        years = sorted(set().union(*[set(s['yearly'].index) for s in self.results.values()]))
        x = np.arange(len(years))
        width = 0.2

        for i, (name, stats) in enumerate(self.results.items()):
            values = [stats['yearly'].get(y, 0) for y in years]
            ax3.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.8)

        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(years, rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Yearly Performance', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Pips')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 勝率比較
        ax4 = fig.add_subplot(3, 2, 4)
        names = list(self.results.keys())
        win_rates = [s['win_rate'] for s in self.results.values()]
        bars = ax4.bar(names, win_rates, color=colors[:len(names)], alpha=0.8)
        ax4.axhline(y=50, color='red', linestyle='--', label='50%')
        ax4.set_title('Win Rate Comparison', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim(0, 100)
        for bar, val in zip(bars, win_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                    ha='center', fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Net Pips vs MaxDD
        ax5 = fig.add_subplot(3, 2, 5)
        net = [s['total_pips'] for s in self.results.values()]
        dd = [abs(s['max_dd']) for s in self.results.values()]
        x = np.arange(len(names))
        width = 0.35
        ax5.bar(x - width/2, net, width, label='Net Pips', color='green', alpha=0.7)
        ax5.bar(x + width/2, dd, width, label='Max DD (abs)', color='red', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(names, rotation=45, ha='right')
        ax5.set_title('Net Pips vs Max Drawdown', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Pips')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. サマリーテーブル
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')

        table_data = []
        for name, stats in self.results.items():
            profit_01lot = stats['total_pips'] * 100  # 0.1lot = 100円/pip
            row = [
                name,
                f"{stats['trades']}",
                f"{stats['win_rate']}%",
                f"{stats['total_pips']:+,.0f}",
                f"{stats['max_dd']:.0f}",
                f"{stats['pf']:.2f}",
                f"{profit_01lot:+,.0f}円"
            ]
            table_data.append(row)

        table = ax6.table(
            cellText=table_data,
            colLabels=['Config', 'Trades', 'WR', 'Net', 'MaxDD', 'PF', 'Profit(0.1lot)'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(7):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # ベストをハイライト
        best_idx = max(range(len(table_data)),
                      key=lambda i: float(table_data[i][3].replace('+', '').replace(',', '')))
        for j in range(7):
            table[(best_idx + 1, j)].set_facecolor('#C6EFCE')

        ax6.set_title('Summary (Best in green)', fontsize=11, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n保存: {save_path}")


def main():
    print("="*70)
    print("  H4実データバックテスト")
    print("  USDJPY 2015-2025 (Axiory)")
    print("  スプレッド: 0.3 pips")
    print("="*70)

    bt = H4RealBacktester(spread_pips=0.3)
    bt.load_data()
    bt.run_comparison()
    bt.print_results()
    bt.visualize('/Users/naoto/ドル円/h4_real_backtest/h4_backtest_results.png')

    # 利益計算
    print("\n" + "="*70)
    print("  【10年間利益】")
    print("="*70)
    for name, stats in bt.results.items():
        profit_001 = stats['total_pips'] * 10
        profit_01 = stats['total_pips'] * 100
        profit_1 = stats['total_pips'] * 1000
        print(f"  {name}:")
        print(f"    0.01lot: {profit_001:+,.0f}円 / 0.1lot: {profit_01:+,.0f}円 / 1.0lot: {profit_1:+,.0f}円")

    print("\n完了!")


if __name__ == "__main__":
    main()
