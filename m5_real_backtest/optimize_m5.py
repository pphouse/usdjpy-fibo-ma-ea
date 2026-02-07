"""
M5実データ パラメータ最適化
M1データから変換して最適化
ゴゴジャン人気No.1の時間枠
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


def load_m1_data():
    """M1データを読み込み"""
    base_path = Path("/Users/naoto/ドル円/usd_jpy_M1/extracted")

    all_files = []

    usdjpy_folder = base_path / "USDJPY"
    if usdjpy_folder.exists():
        for f in usdjpy_folder.glob("USDJPY_20*_*.csv"):
            if "_all" not in f.name:
                all_files.append(f)

    for f in base_path.glob("USDJPY_20*_*.csv"):
        if "_all" not in f.name:
            all_files.append(f)

    print(f"M1ファイル: {len(all_files)}件")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, header=None,
                           names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            dfs.append(df)
        except Exception as e:
            print(f"  {f.name}: ERROR - {e}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'],
                                        format='%Y.%m.%d %H:%M')
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')
    df_all = df_all[~df_all.index.duplicated(keep='first')]

    return df_all[['open', 'high', 'low', 'close', 'volume']]


def resample_to_m5(df_m1):
    """M1をM5に変換"""
    df_m5 = df_m1.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df_m5


class M5Optimizer:
    """M5パラメータ最適化"""

    def __init__(self, spread_pips=0.3):
        self.spread_pips = spread_pips
        self.df = None
        self.results = []

    def load_data(self):
        """データ読み込み"""
        print("M1データを読み込み中...")
        df_m1 = load_m1_data()

        print("M5に変換中...")
        self.df = resample_to_m5(df_m1)

        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

        print(f"M5データ: {len(self.df):,}本 ({self.df.index[0].date()} ~ {self.df.index[-1].date()})")

        output_path = '/Users/naoto/ドル円/m5_real_backtest/USDJPY_M5_2015_2025.csv'
        self.df.to_csv(output_path)
        print(f"保存: {output_path}")

        return self.df

    def calculate_fibonacci(self, idx, lookback=300):
        """フィボナッチレベル計算"""
        if idx < lookback:
            return None, None, 0

        window = self.df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.2:  # M5用に閾値を下げる
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

    def run_single_backtest(self, tp_pips, sl_pips, nanpin_interval, tolerance_pips=10):
        """単一パラメータでバックテスト"""
        lot_ratios = [1, 3, 3, 5]
        max_nanpin = 3
        pip_value = 0.01
        lookback = 300  # M5用に調整（約25時間分）
        spread = self.spread_pips

        trades = []
        in_trade = False
        last_exit_idx = 0

        # データが多いので間隔を空けてサンプリング
        step = 12  # 12バー（1時間）ごとにチェック

        for idx in range(lookback + 200, len(self.df) - 1000, step):
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
                fibo_hit = True
            elif abs(close - fibo_618) < tolerance:
                fibo_hit = True
            if not fibo_hit:
                continue

            ma_hit = None
            for period in [20, 50, 100]:
                ma_val = self.df[f'ma_{period}'].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = True
                    break
            if not ma_hit:
                continue

            trend_ma = self.df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    continue
                if trend == -1 and close > trend_ma:
                    continue

            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{'price': close, 'lot': lot_ratios[0]}]
            last_entry_price = close
            nanpin_count = 0

            for i in range(idx + 1, min(idx + 6000, len(self.df))):
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
                        result = 'WIN'
                    elif current_low <= sl_price:
                        result = 'LOSS'
                else:
                    if current_low <= tp_price:
                        result = 'WIN'
                    elif current_high >= sl_price:
                        result = 'LOSS'

                if result:
                    pips = tp_pips if result == 'WIN' else -sl_pips
                    spread_cost = sum(e['lot'] * spread for e in entries)
                    weighted_pips = pips * total_lot - spread_cost
                    trades.append({
                        'weighted_pips': weighted_pips,
                        'result': result,
                        'nanpin_count': nanpin_count
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
                        entries.append({'price': current_close, 'lot': lot})
                        last_entry_price = current_close

        if not trades:
            return None

        df_trades = pd.DataFrame(trades)
        total = len(df_trades)
        wins = (df_trades['result'] == 'WIN').sum()
        total_pips = df_trades['weighted_pips'].sum()

        cumulative = df_trades['weighted_pips'].cumsum()
        max_dd = (cumulative - cumulative.expanding().max()).min()

        win_pips = df_trades[df_trades['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df_trades[df_trades['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else 999

        return {
            'tp': tp_pips,
            'sl': sl_pips,
            'nanpin': nanpin_interval,
            'trades': total,
            'win_rate': round(wins / total * 100, 2),
            'total_pips': round(total_pips, 1),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'avg_pips': round(total_pips / total, 2)
        }

    def run_optimization(self):
        """グリッドサーチ最適化"""
        if self.df is None:
            self.load_data()

        # パラメータ範囲（M5用：スキャルピング向けに小さい値）
        tp_range = [10, 15, 20, 25, 30, 40]
        sl_range = [10, 15, 20, 25, 30, 40]
        nanpin_range = [3, 5, 7, 10, 12, 15]

        total_combinations = len(tp_range) * len(sl_range) * len(nanpin_range)
        print(f"\n最適化開始: {total_combinations}通り")
        print(f"  TP: {tp_range}")
        print(f"  SL: {sl_range}")
        print(f"  Nanpin: {nanpin_range}")

        self.results = []
        count = 0

        for tp, sl, nanpin in product(tp_range, sl_range, nanpin_range):
            count += 1
            if count % 20 == 0:
                print(f"  進捗: {count}/{total_combinations}")

            result = self.run_single_backtest(tp, sl, nanpin)
            if result:
                self.results.append(result)

        print(f"\n完了: {len(self.results)}件の有効な結果")
        return self.results

    def print_results(self):
        """結果表示"""
        df = pd.DataFrame(self.results)

        print("\n" + "="*100)
        print("  パラメータ最適化結果 (M5実データ, N3固定)")
        print("="*100)

        print("\n【Top 20 (Net Pips順)】")
        top20 = df.nlargest(20, 'total_pips')
        print(f"{'Rank':<5}{'TP':<6}{'SL':<6}{'Nanpin':<8}{'Trades':<8}{'WR%':<8}{'Net':<12}{'MaxDD':<10}{'PF':<8}")
        print("-"*80)
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            print(f"{i:<5}{row['tp']:<6}{row['sl']:<6}{row['nanpin']:<8}"
                  f"{row['trades']:<8}{row['win_rate']:<8}"
                  f"{row['total_pips']:<12.0f}{row['max_dd']:<10.0f}{row['pf']:<8.2f}")

        print("\n【全体統計】")
        print(f"  テスト数: {len(df)}")
        print(f"  Net Pips: {df['total_pips'].min():.0f} ~ {df['total_pips'].max():.0f}")
        print(f"  勝率: {df['win_rate'].min():.1f}% ~ {df['win_rate'].max():.1f}%")
        print(f"  PF: {df['pf'].min():.2f} ~ {df['pf'].max():.2f}")

        return top20

    def visualize_optimization(self, save_path):
        """最適化結果の可視化"""
        df = pd.DataFrame(self.results)

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('M5 Real Data Parameter Optimization (N3 Fixed, Axiory 2015-2025)\nGogoJungle Most Popular Timeframe',
                    fontsize=16, fontweight='bold')

        ax1 = fig.add_subplot(2, 3, 1)
        pivot1 = df.groupby(['tp', 'sl'])['total_pips'].mean().unstack()
        im1 = ax1.imshow(pivot1.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(pivot1.columns)))
        ax1.set_xticklabels(pivot1.columns)
        ax1.set_yticks(range(len(pivot1.index)))
        ax1.set_yticklabels(pivot1.index)
        ax1.set_xlabel('SL (pips)')
        ax1.set_ylabel('TP (pips)')
        ax1.set_title('Avg Net Pips (TP vs SL)', fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(2, 3, 2)
        nanpin_perf = df.groupby('nanpin')['total_pips'].mean()
        bars = ax2.bar(nanpin_perf.index.astype(str), nanpin_perf.values, color='steelblue', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Nanpin Interval (pips)')
        ax2.set_ylabel('Avg Net Pips')
        ax2.set_title('Avg Net Pips by Nanpin Interval', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, nanpin_perf.values):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = fig.add_subplot(2, 3, 3)
        top20 = df.nlargest(20, 'total_pips')
        ax3.scatter(top20['max_dd'].abs(), top20['total_pips'],
                   c=top20['pf'], cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Max Drawdown (abs)')
        ax3.set_ylabel('Net Pips')
        ax3.set_title('Top 20: Net vs MaxDD (color=PF)', fontsize=11, fontweight='bold')
        plt.colorbar(ax3.collections[0], ax=ax3, label='PF')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(2, 3, 4)
        tp_perf = df.groupby('tp').agg({'total_pips': 'mean', 'win_rate': 'mean'})
        ax4.bar(tp_perf.index.astype(str), tp_perf['total_pips'], color='green', alpha=0.7, label='Avg Net Pips')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(tp_perf.index.astype(str), tp_perf['win_rate'], 'ro-', label='Win Rate')
        ax4.set_xlabel('TP (pips)')
        ax4.set_ylabel('Avg Net Pips')
        ax4_twin.set_ylabel('Win Rate (%)')
        ax4.set_title('Performance by TP', fontsize=11, fontweight='bold')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')

        ax5 = fig.add_subplot(2, 3, 5)
        sl_perf = df.groupby('sl').agg({'total_pips': 'mean', 'max_dd': 'mean'})
        ax5.bar(sl_perf.index.astype(str), sl_perf['total_pips'], color='blue', alpha=0.7, label='Avg Net Pips')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(sl_perf.index.astype(str), sl_perf['max_dd'].abs(), 'ro-', label='Avg MaxDD')
        ax5.set_xlabel('SL (pips)')
        ax5.set_ylabel('Avg Net Pips')
        ax5_twin.set_ylabel('Avg Max DD (abs)')
        ax5.set_title('Performance by SL', fontsize=11, fontweight='bold')
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        ax5.grid(True, alpha=0.3, axis='y')

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')

        top10 = df.nlargest(10, 'total_pips')
        table_data = []
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            profit = row['total_pips'] * 100
            table_data.append([
                f"#{i}",
                f"{int(row['tp'])}",
                f"{int(row['sl'])}",
                f"{int(row['nanpin'])}",
                f"{row['win_rate']:.1f}%",
                f"{row['total_pips']:+,.0f}",
                f"{row['max_dd']:.0f}",
                f"{profit:+,.0f}円"
            ])

        table = ax6.table(
            cellText=table_data,
            colLabels=['Rank', 'TP', 'SL', 'Nanpin', 'WR', 'Net', 'MaxDD', 'Profit(0.1lot)'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for j in range(8):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        for j in range(8):
            table[(1, j)].set_facecolor('#C6EFCE')

        ax6.set_title('Top 10 Parameters', fontsize=11, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n保存: {save_path}")

    def visualize_heatmaps(self, save_path):
        """詳細ヒートマップ"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Heatmaps (M5 Real Data, N3 Fixed)', fontsize=14, fontweight='bold')

        nanpin_vals = sorted(df['nanpin'].unique())

        for i, nanpin in enumerate(nanpin_vals[:6]):
            ax = axes[i // 3, i % 3]
            subset = df[df['nanpin'] == nanpin]
            pivot = subset.groupby(['tp', 'sl'])['total_pips'].mean().unstack()

            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('SL')
            ax.set_ylabel('TP')
            ax.set_title(f'Nanpin={nanpin}pips', fontsize=11, fontweight='bold')

            for y in range(len(pivot.index)):
                for x in range(len(pivot.columns)):
                    val = pivot.values[y, x]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > 2000 else 'black'
                        ax.text(x, y, f'{val:.0f}', ha='center', va='center',
                               fontsize=8, color=color)

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")

    def save_results(self, save_path):
        """結果をCSV保存"""
        df = pd.DataFrame(self.results)
        df = df.sort_values('total_pips', ascending=False)
        df.to_csv(save_path, index=False)
        print(f"保存: {save_path}")


def main():
    print("="*70)
    print("  M5実データ パラメータ最適化")
    print("  USDJPY 2015-2025 (Axiory M1 → M5)")
    print("  N3固定 (1:3:3:5)")
    print("  ※ゴゴジャン人気No.1の時間枠")
    print("="*70)

    opt = M5Optimizer(spread_pips=0.3)
    opt.run_optimization()
    top20 = opt.print_results()

    opt.visualize_optimization('/Users/naoto/ドル円/m5_real_backtest/optimization_results.png')
    opt.visualize_heatmaps('/Users/naoto/ドル円/m5_real_backtest/optimization_heatmaps.png')
    opt.save_results('/Users/naoto/ドル円/m5_real_backtest/optimization_results.csv')

    best = top20.iloc[0]
    print("\n" + "="*70)
    print("  【ベストパラメータ】")
    print("="*70)
    print(f"  TP: {int(best['tp'])} pips")
    print(f"  SL: {int(best['sl'])} pips")
    print(f"  Nanpin: {int(best['nanpin'])} pips")
    print(f"  ---")
    print(f"  トレード数: {int(best['trades'])}")
    print(f"  勝率: {best['win_rate']}%")
    print(f"  Net Pips: {best['total_pips']:+,.0f}")
    print(f"  Max DD: {best['max_dd']:.0f}")
    print(f"  PF: {best['pf']:.2f}")
    print(f"  ---")
    print(f"  10年利益 (0.1lot): {best['total_pips'] * 100:+,.0f}円")
    print("="*70)

    print("\n完了!")


if __name__ == "__main__":
    main()
