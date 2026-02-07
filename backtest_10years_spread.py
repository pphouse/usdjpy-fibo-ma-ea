"""
Top 5 パラメータで過去10年バックテスト（スプレッド考慮版）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class LongTermBacktesterWithSpread:
    """長期バックテスター（スプレッド考慮版）"""

    def __init__(self, spread_pips=0.3):
        """
        Args:
            spread_pips: スプレッド (pips)
                - 0.2-0.3: 低スプレッド業者 (SBI FX, DMM FX等)
                - 0.5-1.0: 一般的な業者
                - 1.0-2.0: 高ボラティリティ時
        """
        self.spread_pips = spread_pips
        self.df = None
        self.results = {}
        self.results_no_spread = {}

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
        print(f"価格レンジ: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
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
                     name="", use_spread=True):
        """バックテスト実行"""
        spread = self.spread_pips if use_spread else 0
        spread_label = f"(Spread: {spread}pips)" if use_spread else "(No Spread)"
        print(f"\n[{name}] TP={tp_pips}, SL={sl_pips}, Nanpin={nanpin_interval}, Tol={tolerance_pips} {spread_label}")

        pip_value = 0.01
        lot_ratios = [1, 3, 3, 5]
        max_nanpin = 3
        lookback = 50

        trades = []
        in_trade = False
        last_exit_idx = 0
        total_spread_cost = 0

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
                    # スプレッドコスト計算
                    # エントリー回数 × スプレッド（往復分）
                    num_entries = len(entries)
                    spread_cost_pips = spread * num_entries

                    # 各エントリーのロット加重スプレッドコスト
                    spread_cost_weighted = sum(e['lot'] * spread for e in entries)

                    # 純利益（スプレッド控除後）
                    gross_pips = pips
                    net_pips = pips - spread_cost_pips  # 単純損益からスプレッド控除

                    weighted_gross = gross_pips * total_lot
                    weighted_net = weighted_gross - spread_cost_weighted

                    total_spread_cost += spread_cost_weighted

                    trades.append({
                        'entry_time': entries[0]['time'],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'gross_pips': gross_pips,
                        'spread_cost': spread_cost_pips,
                        'net_pips': net_pips,
                        'weighted_gross': weighted_gross,
                        'weighted_spread': spread_cost_weighted,
                        'weighted_pips': weighted_net,  # スプレッド控除後
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot,
                        'num_entries': num_entries
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
                        entries.append({'price': current_close, 'lot': lot, 'time': self.df.index[i]})
                        last_entry_price = current_close

        print(f"  トレード数: {len(trades)}, 総スプレッドコスト: {total_spread_cost:.1f} pips")
        return trades

    def calculate_stats(self, trades, name):
        """統計計算"""
        if not trades:
            return None

        df = pd.DataFrame(trades)
        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        win_rate = wins / total * 100 if total > 0 else 0

        # グロス（スプレッド前）
        total_gross = df['weighted_gross'].sum()

        # スプレッドコスト
        total_spread = df['weighted_spread'].sum()

        # ネット（スプレッド後）
        total_net = df['weighted_pips'].sum()
        avg_net = df['weighted_pips'].mean()

        # 年別統計
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        yearly = df.groupby('year')['weighted_pips'].sum()
        yearly_gross = df.groupby('year')['weighted_gross'].sum()

        # ドローダウン
        cumulative = df['weighted_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        # PF（ネットベース）
        win_pips = df[df['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df[df['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else 999

        # ナンピン統計
        nanpin_stats = df.groupby('nanpin_count').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'weighted_pips': 'sum',
            'weighted_spread': 'sum'
        }).round(2)

        return {
            'name': name,
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': round(win_rate, 2),
            'total_gross': round(total_gross, 1),
            'total_spread': round(total_spread, 1),
            'total_pips': round(total_net, 1),
            'avg_pips': round(avg_net, 2),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'yearly': yearly,
            'yearly_gross': yearly_gross,
            'nanpin_stats': nanpin_stats,
            'df_trades': df,
            'cumulative': cumulative
        }

    def run_all_backtests(self, params_list):
        """全パラメータでバックテスト実行（スプレッドあり/なし両方）"""
        if self.df is None:
            self.generate_10year_data()

        self.results = {}
        self.results_no_spread = {}

        print("\n" + "="*60)
        print(f"  スプレッド: {self.spread_pips} pips")
        print("="*60)

        for params in params_list:
            name = params['name']

            # スプレッドあり
            trades = self.run_backtest(
                params['tp'], params['sl'],
                params['nanpin'], params['tolerance'],
                name=name, use_spread=True
            )
            stats = self.calculate_stats(trades, name)
            if stats:
                self.results[name] = stats

            # スプレッドなし（比較用）
            trades_no = self.run_backtest(
                params['tp'], params['sl'],
                params['nanpin'], params['tolerance'],
                name=name, use_spread=False
            )
            stats_no = self.calculate_stats(trades_no, name)
            if stats_no:
                self.results_no_spread[name] = stats_no

        return self.results

    def print_comparison(self):
        """結果比較表示"""
        print("\n" + "="*120)
        print(f"  過去10年バックテスト結果比較 (Spread: {self.spread_pips} pips)")
        print("="*120)
        print(f"{'Rank':<6}{'TP':<5}{'SL':<5}{'N':<4}{'T':<4}{'Trades':<7}"
              f"{'WR%':<7}{'Gross':<10}{'Spread':<10}{'Net':<10}{'Impact':<8}{'MaxDD':<9}{'PF':<6}")
        print("-"*120)

        for name in self.results.keys():
            stats = self.results[name]
            stats_no = self.results_no_spread.get(name, {})

            parts = name.split('_')
            tp = parts[1].replace('TP', '')
            sl = parts[2].replace('SL', '')
            nanpin = parts[3].replace('N', '')
            tol = parts[4].replace('T', '')

            # スプレッドインパクト（%）
            gross = stats['total_gross']
            spread_cost = stats['total_spread']
            impact = (spread_cost / gross * 100) if gross > 0 else 0

            print(f"{parts[0]:<6}{tp:<5}{sl:<5}{nanpin:<4}{tol:<4}"
                  f"{stats['trades']:<7}{stats['win_rate']:<7}"
                  f"{stats['total_gross']:<10.0f}{stats['total_spread']:<10.0f}"
                  f"{stats['total_pips']:<10.0f}{impact:<8.1f}%"
                  f"{stats['max_dd']:<9.0f}{stats['pf']:<6.2f}")

        print("="*120)

        # サマリー
        print("\n【スプレッド影響サマリー】")
        for name in self.results.keys():
            stats = self.results[name]
            parts = name.split('_')
            gross = stats['total_gross']
            net = stats['total_pips']
            spread = stats['total_spread']
            impact = (spread / gross * 100) if gross > 0 else 0
            print(f"  {parts[0]}: グロス {gross:+,.0f} → ネット {net:+,.0f} pips "
                  f"(スプレッドコスト: {spread:,.0f} pips = {impact:.1f}%)")

    def visualize_comparison(self, save_path):
        """比較グラフ作成"""
        n_params = len(self.results)
        fig = plt.figure(figsize=(20, 18))
        fig.suptitle(f'10-Year Backtest Comparison (Spread: {self.spread_pips} pips)',
                    fontsize=16, fontweight='bold')

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

        # 1. 累積損益比較（グロス vs ネット）
        ax1 = fig.add_subplot(2, 2, 1)
        for i, (name, stats) in enumerate(self.results.items()):
            df = stats['df_trades']
            cumulative_net = df['weighted_pips'].cumsum()
            cumulative_gross = df['weighted_gross'].cumsum()
            label = name.split('_')[0]
            ax1.plot(df['entry_time'], cumulative_gross, color=colors[i],
                    linewidth=1, linestyle='--', alpha=0.5)
            ax1.plot(df['entry_time'], cumulative_net, color=colors[i],
                    linewidth=2, label=f'{label} (Net)', alpha=0.9)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title('Cumulative Pips: Gross (dashed) vs Net (solid)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pips')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 2. スプレッドコスト分析
        ax2 = fig.add_subplot(2, 2, 2)
        names = [n.split('_')[0] for n in self.results.keys()]
        gross_vals = [s['total_gross'] for s in self.results.values()]
        spread_vals = [s['total_spread'] for s in self.results.values()]
        net_vals = [s['total_pips'] for s in self.results.values()]

        x = np.arange(len(names))
        width = 0.25

        ax2.bar(x - width, gross_vals, width, label='Gross', color='lightblue', edgecolor='blue')
        ax2.bar(x, spread_vals, width, label='Spread Cost', color='salmon', edgecolor='red')
        ax2.bar(x + width, net_vals, width, label='Net', color='lightgreen', edgecolor='green')

        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_title('Gross vs Spread Cost vs Net', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 年別ネット損益
        ax3 = fig.add_subplot(2, 2, 3)
        x = np.arange(10)
        width = 0.15
        for i, (name, stats) in enumerate(self.results.items()):
            yearly = stats['yearly']
            years = sorted(yearly.index)[-10:]
            values = [yearly.get(y, 0) for y in years]
            offset = (i - n_params/2 + 0.5) * width
            label = name.split('_')[0]
            ax3.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(y) for y in years], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Yearly Net Performance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Pips')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. サマリーテーブル
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        table_data = []
        for name, stats in self.results.items():
            parts = name.split('_')
            impact = (stats['total_spread'] / stats['total_gross'] * 100) if stats['total_gross'] > 0 else 0
            row = [
                parts[0],
                f"{stats['trades']}",
                f"{stats['win_rate']}%",
                f"{stats['total_gross']:+,.0f}",
                f"{stats['total_spread']:,.0f}",
                f"{stats['total_pips']:+,.0f}",
                f"{impact:.1f}%",
                f"{stats['pf']:.2f}"
            ]
            table_data.append(row)

        table = ax4.table(
            cellText=table_data,
            colLabels=['Rank', 'Trades', 'WR', 'Gross', 'Spread', 'Net', 'Impact', 'PF'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(8):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        best_idx = max(range(len(table_data)),
                      key=lambda i: float(table_data[i][5].replace('+', '').replace(',', '')))
        for j in range(8):
            table[(best_idx + 1, j)].set_facecolor('#C6EFCE')

        ax4.set_title(f'Summary (Spread: {self.spread_pips} pips, Best in green)',
                     fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n比較グラフを保存しました: {save_path}")

    def calculate_profit_in_yen(self, base_lot=0.1):
        """実際の利益を円で計算"""
        print("\n" + "="*80)
        print(f"  実際の利益計算 (基準ロット: {base_lot} lot, スプレッド: {self.spread_pips} pips)")
        print("="*80)

        pip_value_per_lot = 1000  # 1 lot = 1000円/pip
        pip_value = pip_value_per_lot * base_lot  # base_lot あたりの1pip価値

        print(f"\n  1 pip = {pip_value:,.0f}円 (基準ロット {base_lot} lot の場合)")
        print(f"  ※ナンピン時は最大 12倍 (1+3+3+5) のロットになります\n")

        print(f"{'Rank':<8}{'Net Pips':<12}{'利益(税前)':<15}{'利益(税後20%)':<15}")
        print("-"*60)

        for name, stats in self.results.items():
            parts = name.split('_')
            net_pips = stats['total_pips']
            profit_gross = net_pips * pip_value
            profit_after_tax = profit_gross * 0.8  # 20%税金

            print(f"{parts[0]:<8}{net_pips:>+10,.0f}  {profit_gross:>+14,.0f}円  {profit_after_tax:>+14,.0f}円")

        print("-"*60)
        print("\n  ※上記は10年間の累計利益です")
        print("  ※必要証拠金: 約50-60万円 (最大ナンピン時、レバレッジ25倍)")


def main():
    print("="*60)
    print("  過去10年バックテスト（スプレッド考慮版）")
    print("  Top 5 パラメータ比較")
    print("="*60)

    #===========================================
    # スプレッド設定
    #===========================================
    SPREAD_PIPS = 0.3  # USDJPYの一般的なスプレッド
    #===========================================

    params_list = [
        {'name': 'Rank1_TP70_SL80_N20_T35', 'tp': 70, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank2_TP80_SL80_N20_T35', 'tp': 80, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank3_TP100_SL80_N20_T35', 'tp': 100, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank4_TP100_SL100_N20_T35', 'tp': 100, 'sl': 100, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank5_TP80_SL100_N30_T30', 'tp': 80, 'sl': 100, 'nanpin': 30, 'tolerance': 30},
    ]

    bt = LongTermBacktesterWithSpread(spread_pips=SPREAD_PIPS)
    bt.run_all_backtests(params_list)
    bt.print_comparison()
    bt.visualize_comparison('/Users/naoto/ドル円/backtest_10y_spread_comparison.png')

    # 実際の利益を円で計算
    print("\n" + "="*80)
    print("  【ロットサイズ別 利益計算】")
    print("="*80)
    for lot in [0.01, 0.1, 1.0]:
        bt.calculate_profit_in_yen(base_lot=lot)

    print("\n完了!")


if __name__ == "__main__":
    main()
