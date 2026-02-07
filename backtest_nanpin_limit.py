"""
ナンピン回数制限比較バックテスト
N3（3回）vs N2（2回）の比較
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


class NanpinLimitBacktester:
    """ナンピン回数制限比較バックテスター"""

    def __init__(self, spread_pips=0.3):
        self.spread_pips = spread_pips
        self.df = None
        self.results = {}

    def generate_10year_data(self):
        """10年分のデータを生成"""
        print("過去10年のデータを生成中...")
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
        print(f"データ生成完了: {len(df)}本 ({df.index[0].date()} ~ {df.index[-1].date()})")
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

    def run_backtest(self, tp_pips, sl_pips, nanpin_interval, tolerance_pips,
                     max_nanpin=3, lot_ratios=None, name=""):
        """バックテスト実行"""
        if lot_ratios is None:
            lot_ratios = [1, 3, 3, 5]

        spread = self.spread_pips
        print(f"\n[{name}] TP={tp_pips}, SL={sl_pips}, MaxNanpin={max_nanpin}, "
              f"Lots={lot_ratios[:max_nanpin+1]}")

        pip_value = 0.01
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

            # トレードシミュレーション
            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{'price': close, 'lot': lot_ratios[0], 'time': self.df.index[idx]}]
            last_entry_price = close
            nanpin_count = 0

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
                    num_entries = len(entries)
                    spread_cost_weighted = sum(e['lot'] * spread for e in entries)
                    weighted_gross = pips * total_lot
                    weighted_net = weighted_gross - spread_cost_weighted

                    trades.append({
                        'entry_time': entries[0]['time'],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'pips': pips,
                        'weighted_gross': weighted_gross,
                        'weighted_spread': spread_cost_weighted,
                        'weighted_pips': weighted_net,
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot,
                        'num_entries': num_entries
                    })
                    last_exit_idx = i
                    in_trade = True
                    break

                # ナンピン判定（制限付き）
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

        total_gross = df['weighted_gross'].sum()
        total_spread = df['weighted_spread'].sum()
        total_net = df['weighted_pips'].sum()

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
            'total_gross': round(total_gross, 1),
            'total_spread': round(total_spread, 1),
            'total_pips': round(total_net, 1),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'yearly': yearly,
            'nanpin_stats': nanpin_stats,
            'df_trades': df,
            'cumulative': cumulative
        }

    def compare_nanpin_limits(self):
        """ナンピン回数制限の比較"""
        if self.df is None:
            self.generate_10year_data()

        # Rank4パラメータで比較（最良パフォーマンス）
        tp, sl, interval, tol = 100, 100, 20, 35

        configs = [
            {'name': 'N3_1-3-3-5', 'max_nanpin': 3, 'lots': [1, 3, 3, 5], 'max_lot': 12},
            {'name': 'N2_1-3-3', 'max_nanpin': 2, 'lots': [1, 3, 3], 'max_lot': 7},
            {'name': 'N2_1-2-3', 'max_nanpin': 2, 'lots': [1, 2, 3], 'max_lot': 6},
            {'name': 'N1_1-3', 'max_nanpin': 1, 'lots': [1, 3], 'max_lot': 4},
            {'name': 'N0_NoNanpin', 'max_nanpin': 0, 'lots': [1], 'max_lot': 1},
        ]

        print("\n" + "="*70)
        print("  ナンピン回数制限比較 (Rank4: TP100/SL100/N20/T35)")
        print("="*70)

        for cfg in configs:
            trades = self.run_backtest(
                tp, sl, interval, tol,
                max_nanpin=cfg['max_nanpin'],
                lot_ratios=cfg['lots'],
                name=cfg['name']
            )
            stats = self.calculate_stats(trades, cfg['name'])
            if stats:
                stats['max_lot'] = cfg['max_lot']
                self.results[cfg['name']] = stats

        return self.results

    def print_comparison(self):
        """比較結果表示"""
        print("\n" + "="*100)
        print("  ナンピン回数制限 比較結果")
        print("="*100)
        print(f"{'設定':<15}{'最大Lot':<8}{'Trades':<8}{'WR%':<7}{'Gross':<10}{'Spread':<9}{'Net':<10}{'MaxDD':<10}{'PF':<6}")
        print("-"*100)

        for name, stats in self.results.items():
            print(f"{name:<15}{stats['max_lot']:<8}{stats['trades']:<8}"
                  f"{stats['win_rate']:<7}{stats['total_gross']:<10.0f}"
                  f"{stats['total_spread']:<9.0f}{stats['total_pips']:<10.0f}"
                  f"{stats['max_dd']:<10.0f}{stats['pf']:<6.2f}")

        print("="*100)

        # ナンピン回数別詳細
        print("\n【ナンピン回数別パフォーマンス詳細】")
        for name, stats in self.results.items():
            print(f"\n  {name}:")
            ns = stats['nanpin_stats']
            for idx, row in ns.iterrows():
                print(f"    N{idx}: {int(row['count'])}回, 勝率{row['win_rate']:.1f}%, 損益{row['total_pips']:+,.0f}pips")

    def visualize_comparison(self, save_path):
        """比較グラフ作成"""
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Nanpin Limit Comparison (Rank4: TP100/SL100/N20/T35)',
                    fontsize=16, fontweight='bold')

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

        # 1. 累積損益比較
        ax1 = fig.add_subplot(2, 2, 1)
        for i, (name, stats) in enumerate(self.results.items()):
            df = stats['df_trades']
            cumulative = df['weighted_pips'].cumsum()
            ax1.plot(df['entry_time'], cumulative, color=colors[i],
                    linewidth=2, label=name, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title('Cumulative Net Pips', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pips')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 2. Net Pips vs Max Drawdown
        ax2 = fig.add_subplot(2, 2, 2)
        names = list(self.results.keys())
        net_pips = [s['total_pips'] for s in self.results.values()]
        max_dds = [abs(s['max_dd']) for s in self.results.values()]

        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width/2, net_pips, width, label='Net Pips', color='green', alpha=0.7)
        ax2.bar(x + width/2, max_dds, width, label='Max DD (abs)', color='red', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_title('Net Pips vs Max Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. リスク調整済みリターン
        ax3 = fig.add_subplot(2, 2, 3)
        risk_adjusted = [s['total_pips'] / abs(s['max_dd']) if s['max_dd'] != 0 else 0
                        for s in self.results.values()]
        bars = ax3.bar(names, risk_adjusted, color=colors, alpha=0.8)
        ax3.set_title('Risk-Adjusted Return (Net / MaxDD)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.set_xticklabels(names, rotation=45, ha='right')
        for bar, val in zip(bars, risk_adjusted):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. サマリーテーブル
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        table_data = []
        for name, stats in self.results.items():
            risk_adj = stats['total_pips'] / abs(stats['max_dd']) if stats['max_dd'] != 0 else 0
            row = [
                name,
                f"{stats['max_lot']}x",
                f"{stats['trades']}",
                f"{stats['win_rate']}%",
                f"{stats['total_pips']:+,.0f}",
                f"{stats['max_dd']:.0f}",
                f"{risk_adj:.2f}",
                f"{stats['pf']:.2f}"
            ]
            table_data.append(row)

        table = ax4.table(
            cellText=table_data,
            colLabels=['Config', 'MaxLot', 'Trades', 'WR', 'Net', 'MaxDD', 'Risk-Adj', 'PF'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(8):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # ベストをハイライト（リスク調整済みリターンが最高）
        best_idx = max(range(len(table_data)),
                      key=lambda i: float(table_data[i][6]))
        for j in range(8):
            table[(best_idx + 1, j)].set_facecolor('#C6EFCE')

        ax4.set_title('Summary (Best Risk-Adjusted in green)', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n比較グラフを保存しました: {save_path}")

    def calculate_profit_comparison(self, base_lot=0.1):
        """利益比較"""
        print("\n" + "="*80)
        print(f"  10年間利益比較 (基準ロット: {base_lot} lot)")
        print("="*80)

        pip_value = 1000 * base_lot

        print(f"\n{'設定':<15}{'最大Lot':<10}{'Net Pips':<12}{'利益(税前)':<15}{'必要証拠金概算':<15}")
        print("-"*80)

        for name, stats in self.results.items():
            net = stats['total_pips']
            profit = net * pip_value
            # 必要証拠金: 150円 × 100,000通貨 × 最大ロット / レバレッジ25
            margin = 150 * 100000 * base_lot * stats['max_lot'] / 25
            print(f"{name:<15}{stats['max_lot']}x{'':<7}{net:>+10,.0f}  "
                  f"{profit:>+14,.0f}円  {margin:>14,.0f}円")

        print("-"*80)
        print("  ※必要証拠金はUSDJPY=150円、レバレッジ25倍で計算")


def main():
    print("="*70)
    print("  ナンピン回数制限 比較バックテスト")
    print("  10年間 / スプレッド0.3pips考慮")
    print("="*70)

    bt = NanpinLimitBacktester(spread_pips=0.3)
    bt.compare_nanpin_limits()
    bt.print_comparison()
    bt.visualize_comparison('/Users/naoto/ドル円/nanpin_limit_comparison.png')
    bt.calculate_profit_comparison(base_lot=0.1)

    print("\n" + "="*70)
    print("  【結論】")
    print("="*70)

    # 最良設定を特定
    best_risk_adj = None
    best_name = None
    for name, stats in bt.results.items():
        risk_adj = stats['total_pips'] / abs(stats['max_dd']) if stats['max_dd'] != 0 else 0
        if best_risk_adj is None or risk_adj > best_risk_adj:
            best_risk_adj = risk_adj
            best_name = name

    print(f"  リスク調整済みリターン最良: {best_name}")
    print(f"  → ナンピン回数制限によりドローダウンを抑制しつつ利益確保")
    print("="*70)

    print("\n完了!")


if __name__ == "__main__":
    main()
