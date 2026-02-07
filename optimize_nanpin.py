"""
フィボナッチ + MA + ナンピン EA パラメータ最適化
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class NanpinOptimizer:
    """ナンピンEAパラメータ最適化"""

    def __init__(self):
        self.results = []
        self.df = None
        self.best_result = None
        self.iteration = 0

    def generate_data(self, days=365):
        """データ生成"""
        print("データ生成中...")
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

        # MA計算
        for period in [20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()

        self.df = df
        print(f"データ生成完了: {len(df)}本")
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
        if range_val < 0.5:
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

    def run_single_backtest(self, tp_pips, sl_pips, nanpin_interval, tolerance_pips):
        """単一パラメータセットでバックテスト"""
        pip_value = 0.01
        lot_ratios = [1, 3, 3, 5]
        max_nanpin = 3
        lookback = 50

        trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(self.df) - 100):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend = self.calculate_fibonacci(idx, lookback)
            if trend == 0:
                continue

            close = self.df['close'].iloc[idx]
            tolerance = tolerance_pips * pip_value

            # フィボ条件
            fibo_hit = None
            if abs(close - fibo_382) < tolerance:
                fibo_hit = "38.2%"
            elif abs(close - fibo_618) < tolerance:
                fibo_hit = "61.8%"
            if fibo_hit is None:
                continue

            # MA条件
            ma_hit = None
            for period in [20, 50, 100]:
                ma_val = self.df[f'ma_{period}'].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = period
                    break
            if ma_hit is None:
                continue

            # トレンドフィルター
            trend_ma = self.df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    continue
                if trend == -1 and close > trend_ma:
                    continue

            # トレードシミュレーション
            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{'price': close, 'lot': lot_ratios[0]}]
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

                # TP/SL判定
                if direction == 'BUY':
                    if current_high >= tp_price:
                        pips = (tp_price - avg_price) / pip_value
                        weighted_pips = pips * total_lot
                        trades.append({'pips': pips, 'weighted_pips': weighted_pips,
                                      'result': 'WIN', 'nanpin': nanpin_count})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_low <= sl_price:
                        pips = (sl_price - avg_price) / pip_value
                        weighted_pips = pips * total_lot
                        trades.append({'pips': pips, 'weighted_pips': weighted_pips,
                                      'result': 'LOSS', 'nanpin': nanpin_count})
                        last_exit_idx = i
                        in_trade = True
                        break
                else:
                    if current_low <= tp_price:
                        pips = (avg_price - tp_price) / pip_value
                        weighted_pips = pips * total_lot
                        trades.append({'pips': pips, 'weighted_pips': weighted_pips,
                                      'result': 'WIN', 'nanpin': nanpin_count})
                        last_exit_idx = i
                        in_trade = True
                        break
                    elif current_high >= sl_price:
                        pips = (avg_price - sl_price) / pip_value
                        weighted_pips = pips * total_lot
                        trades.append({'pips': pips, 'weighted_pips': weighted_pips,
                                      'result': 'LOSS', 'nanpin': nanpin_count})
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
                        entries.append({'price': current_close, 'lot': lot})
                        last_entry_price = current_close

        return trades

    def calculate_stats(self, trades):
        """統計計算"""
        if not trades:
            return None

        df = pd.DataFrame(trades)
        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        win_rate = wins / total * 100 if total > 0 else 0

        total_weighted_pips = df['weighted_pips'].sum()

        # ドローダウン
        cumulative = df['weighted_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        # PF
        win_pips = df[df['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df[df['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else 999

        # リカバリーファクター
        rf = total_weighted_pips / abs(max_dd) if max_dd != 0 else 999

        return {
            'trades': total,
            'win_rate': round(win_rate, 2),
            'total_pips': round(total_weighted_pips, 1),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'rf': round(rf, 2)
        }

    def optimize(self, tp_range, sl_range, nanpin_range, tolerance_range):
        """パラメータ最適化"""
        if self.df is None:
            self.generate_data()

        total_combinations = len(tp_range) * len(sl_range) * len(nanpin_range) * len(tolerance_range)
        print(f"\n最適化開始: {total_combinations}通りをテスト")
        print("="*60)

        self.results = []
        self.iteration = 0
        best_pips = -999999

        for tp, sl, nanpin, tol in product(tp_range, sl_range, nanpin_range, tolerance_range):
            self.iteration += 1

            trades = self.run_single_backtest(tp, sl, nanpin, tol)
            stats = self.calculate_stats(trades)

            if stats:
                result = {
                    'TP': tp, 'SL': sl, 'Nanpin': nanpin, 'Tolerance': tol,
                    **stats
                }
                self.results.append(result)

                # ベスト更新チェック
                if stats['total_pips'] > best_pips:
                    best_pips = stats['total_pips']
                    self.best_result = result

            # 進捗表示（10%ごと）
            if self.iteration % max(1, total_combinations // 10) == 0:
                progress = self.iteration / total_combinations * 100
                print(f"進捗: {progress:.0f}% ({self.iteration}/{total_combinations})")
                if self.best_result:
                    print(f"  現在のベスト: TP={self.best_result['TP']}, SL={self.best_result['SL']}, "
                          f"Nanpin={self.best_result['Nanpin']}, Tol={self.best_result['Tolerance']} "
                          f"-> {self.best_result['total_pips']} pips")

                # 途中経過を保存
                self.save_progress_chart()

        print("="*60)
        print("最適化完了!")
        return pd.DataFrame(self.results)

    def save_progress_chart(self):
        """途中経過チャートを保存"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Optimization Progress ({self.iteration} iterations)',
                     fontsize=14, fontweight='bold')

        # 1. 損益分布
        ax1 = axes[0, 0]
        ax1.hist(df['total_pips'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--')
        if self.best_result:
            ax1.axvline(x=self.best_result['total_pips'], color='green', linewidth=2,
                       label=f"Best: {self.best_result['total_pips']}")
        ax1.set_xlabel('Total Pips')
        ax1.set_ylabel('Count')
        ax1.set_title('Profit Distribution')
        ax1.legend()

        # 2. TP vs 損益
        ax2 = axes[0, 1]
        tp_profit = df.groupby('TP')['total_pips'].mean()
        ax2.bar(tp_profit.index, tp_profit.values, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Take Profit (pips)')
        ax2.set_ylabel('Avg Total Pips')
        ax2.set_title('TP vs Avg Profit')
        ax2.axhline(y=0, color='red', linestyle='--')

        # 3. SL vs 損益
        ax3 = axes[1, 0]
        sl_profit = df.groupby('SL')['total_pips'].mean()
        ax3.bar(sl_profit.index, sl_profit.values, color='orange', alpha=0.7)
        ax3.set_xlabel('Stop Loss (pips)')
        ax3.set_ylabel('Avg Total Pips')
        ax3.set_title('SL vs Avg Profit')
        ax3.axhline(y=0, color='red', linestyle='--')

        # 4. Top 10 設定
        ax4 = axes[1, 1]
        top10 = df.nlargest(10, 'total_pips')
        y_pos = range(len(top10))
        labels = [f"TP{r['TP']}/SL{r['SL']}/N{r['Nanpin']}/T{r['Tolerance']}"
                  for _, r in top10.iterrows()]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(top10)))
        ax4.barh(y_pos, top10['total_pips'], color=colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=8)
        ax4.set_xlabel('Total Pips')
        ax4.set_title('Top 10 Parameters')
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig('/Users/naoto/ドル円/optimization_progress.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_final_results(self):
        """最終結果を保存"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Parameter Optimization Final Results\nFibonacci + MA + Nanpin EA',
                     fontsize=16, fontweight='bold')

        # 1. TP×SL ヒートマップ（損益）
        ax1 = fig.add_subplot(2, 3, 1)
        pivot1 = df.groupby(['TP', 'SL'])['total_pips'].mean().unstack()
        im1 = ax1.imshow(pivot1.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(pivot1.columns)))
        ax1.set_yticks(range(len(pivot1.index)))
        ax1.set_xticklabels(pivot1.columns)
        ax1.set_yticklabels(pivot1.index)
        ax1.set_xlabel('SL (pips)')
        ax1.set_ylabel('TP (pips)')
        ax1.set_title('TP x SL Heatmap (Avg Pips)')
        plt.colorbar(im1, ax=ax1)

        # 2. ナンピン間隔×許容誤差 ヒートマップ
        ax2 = fig.add_subplot(2, 3, 2)
        pivot2 = df.groupby(['Nanpin', 'Tolerance'])['total_pips'].mean().unstack()
        im2 = ax2.imshow(pivot2.values, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(len(pivot2.columns)))
        ax2.set_yticks(range(len(pivot2.index)))
        ax2.set_xticklabels(pivot2.columns)
        ax2.set_yticklabels(pivot2.index)
        ax2.set_xlabel('Tolerance (pips)')
        ax2.set_ylabel('Nanpin Interval (pips)')
        ax2.set_title('Nanpin x Tolerance Heatmap')
        plt.colorbar(im2, ax=ax2)

        # 3. 勝率 vs PF 散布図
        ax3 = fig.add_subplot(2, 3, 3)
        scatter = ax3.scatter(df['win_rate'], df['pf'], c=df['total_pips'],
                             cmap='RdYlGn', alpha=0.6, s=50)
        ax3.set_xlabel('Win Rate (%)')
        ax3.set_ylabel('Profit Factor')
        ax3.set_title('Win Rate vs PF (color=Pips)')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax3.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax3, label='Pips')

        # 4. Top 20 設定
        ax4 = fig.add_subplot(2, 3, 4)
        top20 = df.nlargest(20, 'total_pips')
        y_pos = range(len(top20))
        labels = [f"TP{r['TP']}/SL{r['SL']}/N{r['Nanpin']}/T{r['Tolerance']}"
                  for _, r in top20.iterrows()]
        colors = plt.cm.RdYlGn(np.linspace(0.9, 0.4, len(top20)))
        bars = ax4.barh(y_pos, top20['total_pips'], color=colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(labels, fontsize=7)
        ax4.set_xlabel('Total Pips')
        ax4.set_title('Top 20 Parameter Sets')
        ax4.invert_yaxis()
        for bar, pf in zip(bars, top20['pf']):
            ax4.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'PF:{pf}', va='center', fontsize=7)

        # 5. 損益分布
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(df['total_pips'], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax5.axvline(x=df['total_pips'].mean(), color='orange', linestyle='-',
                   linewidth=2, label=f"Mean: {df['total_pips'].mean():.1f}")
        ax5.set_xlabel('Total Pips')
        ax5.set_ylabel('Count')
        ax5.set_title('Profit Distribution')
        ax5.legend()

        # 6. ベストパラメータ詳細
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        if self.best_result:
            text = f"""
            === BEST PARAMETERS ===

            TP (Take Profit): {self.best_result['TP']} pips
            SL (Stop Loss): {self.best_result['SL']} pips
            Nanpin Interval: {self.best_result['Nanpin']} pips
            Tolerance: {self.best_result['Tolerance']} pips

            === PERFORMANCE ===

            Total Trades: {self.best_result['trades']}
            Win Rate: {self.best_result['win_rate']}%
            Total Pips: {self.best_result['total_pips']}
            Max Drawdown: {self.best_result['max_dd']}
            Profit Factor: {self.best_result['pf']}
            Recovery Factor: {self.best_result['rf']}
            """
            ax6.text(0.1, 0.5, text, transform=ax6.transAxes, fontsize=12,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig('/Users/naoto/ドル円/optimization_final.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("最終結果を保存しました: optimization_final.png")

    def print_top_results(self, n=10):
        """上位結果を表示"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        top = df.nlargest(n, 'total_pips')

        print(f"\n{'='*80}")
        print(f"  TOP {n} パラメータ設定")
        print(f"{'='*80}")
        print(f"{'Rank':<5}{'TP':<6}{'SL':<6}{'Nanpin':<8}{'Tol':<6}{'Trades':<8}{'WinRate':<9}{'Pips':<10}{'DD':<10}{'PF':<6}")
        print("-"*80)

        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"{i:<5}{row['TP']:<6}{row['SL']:<6}{row['Nanpin']:<8}{row['Tolerance']:<6}"
                  f"{row['trades']:<8}{row['win_rate']:<9}{row['total_pips']:<10}"
                  f"{row['max_dd']:<10}{row['pf']:<6}")

        print("="*80)


def main():
    print("="*60)
    print("  パラメータ最適化開始")
    print("  Fibonacci + MA + Nanpin EA")
    print("="*60)

    optimizer = NanpinOptimizer()

    # 最適化範囲
    tp_range = [30, 40, 50, 60, 70, 80, 100]
    sl_range = [30, 40, 50, 60, 70, 80, 100]
    nanpin_range = [20, 30, 40, 50, 60]
    tolerance_range = [15, 20, 25, 30, 35]

    print(f"\nTP範囲: {tp_range}")
    print(f"SL範囲: {sl_range}")
    print(f"ナンピン間隔範囲: {nanpin_range}")
    print(f"許容誤差範囲: {tolerance_range}")

    # 最適化実行
    results_df = optimizer.optimize(tp_range, sl_range, nanpin_range, tolerance_range)

    # 結果表示
    optimizer.print_top_results(20)

    # 最終結果保存
    optimizer.save_final_results()

    # CSV保存
    results_df.to_csv('/Users/naoto/ドル円/optimization_results.csv', index=False)
    print("\n結果をCSVに保存しました: optimization_results.csv")

    print("\n完了!")


if __name__ == "__main__":
    main()
