"""
フィボナッチ + MA EA パラメータ最適化スクリプト
異なるTP/SL設定でバックテストを実行し、最適なパラメータを見つけます
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class ParameterOptimizer:
    """パラメータ最適化クラス"""

    def __init__(self):
        self.results = []

    def generate_sample_data(self, days=365):
        """サンプルデータを生成"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days*6, freq='4h')

        n = len(dates)
        returns = np.random.normal(0.0001, 0.002, n)
        price = 150.0
        prices = [price]

        for r in returns[1:]:
            price = price * (1 + r)
            prices.append(price)

        prices = np.array(prices)

        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
            'close': prices * (1 + np.random.normal(0, 0.001, n)),
        }, index=dates)

        return df

    def calculate_ma(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_fibonacci(self, df, idx, lookback=50):
        if idx < lookback:
            return None, None, 0

        window = df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.1:
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

    def run_single_backtest(self, df, tp_pips, sl_pips, ma_periods=[20, 50, 100]):
        """単一パラメータセットでバックテスト"""
        for period in ma_periods:
            df[f'ma_{period}'] = self.calculate_ma(df, period)
        df['ma_200'] = self.calculate_ma(df, 200)

        trades = []
        lookback = 50
        pip_value = 0.01

        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback, len(df) - 50):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend = self.calculate_fibonacci(df, idx, lookback)
            if trend == 0:
                continue

            close = df['close'].iloc[idx]
            fibo_tol = close * 0.002

            fibo_hit = (abs(close - fibo_382) < fibo_tol) or (abs(close - fibo_618) < fibo_tol)
            if not fibo_hit:
                continue

            ma_tol = close * 0.001
            ma_hit = False
            for period in ma_periods:
                ma_val = df[f'ma_{period}'].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < ma_tol:
                    ma_hit = True
                    break

            if not ma_hit:
                continue

            # トレードシミュレーション
            entry_price = close
            tp_distance = tp_pips * pip_value
            sl_distance = sl_pips * pip_value

            if trend == 1:
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
            else:
                tp_price = entry_price - tp_distance
                sl_price = entry_price + sl_distance

            for i in range(idx + 1, min(idx + 100, len(df))):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]

                if trend == 1:  # BUY
                    if high >= tp_price:
                        trades.append(tp_pips)
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif low <= sl_price:
                        trades.append(-sl_pips)
                        last_exit_idx = i
                        in_trade = True
                        break
                else:  # SELL
                    if low <= tp_price:
                        trades.append(tp_pips)
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif high >= sl_price:
                        trades.append(-sl_pips)
                        last_exit_idx = i
                        in_trade = True
                        break

        return trades

    def optimize(self, tp_range, sl_range):
        """パラメータ最適化を実行"""
        print("サンプルデータ生成中...")
        df = self.generate_sample_data()

        print("パラメータ最適化中...")
        self.results = []

        total = len(tp_range) * len(sl_range)
        count = 0

        for tp, sl in product(tp_range, sl_range):
            count += 1
            if count % 10 == 0:
                print(f"  進捗: {count}/{total}")

            trades = self.run_single_backtest(df.copy(), tp, sl)

            if len(trades) > 0:
                total_pips = sum(trades)
                wins = len([t for t in trades if t > 0])
                losses = len([t for t in trades if t < 0])
                win_rate = wins / len(trades) * 100 if len(trades) > 0 else 0

                cumulative = np.cumsum(trades)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                max_dd = min(drawdown) if len(drawdown) > 0 else 0

                pf = abs(sum([t for t in trades if t > 0]) /
                        sum([t for t in trades if t < 0])) if losses > 0 else 999

                self.results.append({
                    'TP': tp,
                    'SL': sl,
                    'トレード数': len(trades),
                    '勝率': round(win_rate, 2),
                    '合計Pips': round(total_pips, 1),
                    '最大DD': round(max_dd, 1),
                    'PF': round(pf, 2)
                })

        return pd.DataFrame(self.results)

    def visualize_optimization(self, save_path=None):
        """最適化結果をビジュアル化"""
        if not self.results:
            print("結果がありません")
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. 合計Pipsヒートマップ
        ax1 = axes[0, 0]
        pivot1 = df.pivot(index='SL', columns='TP', values='合計Pips')
        im1 = ax1.imshow(pivot1.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(pivot1.columns)))
        ax1.set_yticks(range(len(pivot1.index)))
        ax1.set_xticklabels(pivot1.columns)
        ax1.set_yticklabels(pivot1.index)
        ax1.set_xlabel('TP (pips)')
        ax1.set_ylabel('SL (pips)')
        ax1.set_title('合計Pips ヒートマップ', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Pips')

        # 値をセルに表示
        for i in range(len(pivot1.index)):
            for j in range(len(pivot1.columns)):
                val = pivot1.values[i, j]
                color = 'white' if abs(val) > pivot1.values.max() * 0.5 else 'black'
                ax1.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

        # 2. 勝率ヒートマップ
        ax2 = axes[0, 1]
        pivot2 = df.pivot(index='SL', columns='TP', values='勝率')
        im2 = ax2.imshow(pivot2.values, cmap='RdYlGn', aspect='auto', vmin=30, vmax=70)
        ax2.set_xticks(range(len(pivot2.columns)))
        ax2.set_yticks(range(len(pivot2.index)))
        ax2.set_xticklabels(pivot2.columns)
        ax2.set_yticklabels(pivot2.index)
        ax2.set_xlabel('TP (pips)')
        ax2.set_ylabel('SL (pips)')
        ax2.set_title('勝率 (%) ヒートマップ', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='%')

        for i in range(len(pivot2.index)):
            for j in range(len(pivot2.columns)):
                val = pivot2.values[i, j]
                ax2.text(j, i, f'{val:.1f}', ha='center', va='center', color='black', fontsize=8)

        # 3. 最適パラメータランキング
        ax3 = axes[1, 0]
        top10 = df.nlargest(10, '合計Pips')
        y_pos = range(len(top10))
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(top10)))
        bars = ax3.barh(y_pos, top10['合計Pips'], color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"TP{row['TP']}/SL{row['SL']}" for _, row in top10.iterrows()])
        ax3.set_xlabel('合計Pips')
        ax3.set_title('Top 10 パラメータ設定', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()

        # 4. プロフィットファクター比較
        ax4 = axes[1, 1]
        pivot4 = df.pivot(index='SL', columns='TP', values='PF')
        im4 = ax4.imshow(pivot4.values, cmap='Blues', aspect='auto', vmin=0, vmax=3)
        ax4.set_xticks(range(len(pivot4.columns)))
        ax4.set_yticks(range(len(pivot4.index)))
        ax4.set_xticklabels(pivot4.columns)
        ax4.set_yticklabels(pivot4.index)
        ax4.set_xlabel('TP (pips)')
        ax4.set_ylabel('SL (pips)')
        ax4.set_title('プロフィットファクター ヒートマップ', fontsize=14, fontweight='bold')
        plt.colorbar(im4, ax=ax4, label='PF')

        for i in range(len(pivot4.index)):
            for j in range(len(pivot4.columns)):
                val = pivot4.values[i, j]
                ax4.text(j, i, f'{val:.1f}', ha='center', va='center', color='black', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"最適化結果を保存しました: {save_path}")

        plt.show()


def main():
    """メイン関数"""
    print("="*60)
    print("  パラメータ最適化")
    print("="*60)

    optimizer = ParameterOptimizer()

    # 最適化するパラメータ範囲
    tp_range = [20, 30, 40, 50, 60, 70, 80, 100]
    sl_range = [20, 30, 40, 50, 60, 70, 80, 100]

    print(f"TP範囲: {tp_range}")
    print(f"SL範囲: {sl_range}")

    # 最適化実行
    results_df = optimizer.optimize(tp_range, sl_range)

    # 結果表示
    print("\n" + "="*60)
    print("  最適化結果 Top 10")
    print("="*60)
    top10 = results_df.nlargest(10, '合計Pips')
    print(top10.to_string(index=False))

    # 結果保存
    results_df.to_csv('/Users/naoto/ドル円/optimization_results.csv', index=False, encoding='utf-8-sig')
    print("\n結果をCSVに保存しました: optimization_results.csv")

    # ビジュアル化
    optimizer.visualize_optimization(save_path='/Users/naoto/ドル円/optimization_heatmap.png')


if __name__ == "__main__":
    main()
