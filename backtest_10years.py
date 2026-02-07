"""
Top 5 パラメータで過去10年バックテスト
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


class LongTermBacktester:
    """長期バックテスター"""

    def __init__(self):
        self.df = None
        self.results = {}

    def generate_10year_data(self):
        """10年分のデータを生成"""
        print("過去10年のデータを生成中...")
        np.random.seed(42)

        days = 365 * 10  # 10年
        n_bars = days * 6  # 4時間足
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='4h')

        # より現実的な価格変動を生成
        base_price = 110.0  # 10年前の想定価格

        # 長期トレンド（10年で110円→150円程度の上昇を想定）
        long_trend = np.linspace(0, 40, n_bars)

        # 中期サイクル（年単位の波）
        mid_cycle = np.sin(np.linspace(0, 20*np.pi, n_bars)) * 8

        # 短期ノイズ
        noise = np.cumsum(np.random.normal(0, 0.002 * base_price, n_bars))
        # ノイズのドリフトを補正
        noise = noise - np.linspace(noise[0], noise[-1], n_bars)

        prices = base_price + long_trend + mid_cycle + noise

        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['high'] = df['close'] + np.abs(np.random.normal(0, 0.15, n_bars))
        df['low'] = df['close'] - np.abs(np.random.normal(0, 0.15, n_bars))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)

        # MA計算
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

    def run_backtest(self, tp_pips, sl_pips, nanpin_interval, tolerance_pips, name=""):
        """バックテスト実行"""
        print(f"\n[{name}] TP={tp_pips}, SL={sl_pips}, Nanpin={nanpin_interval}, Tol={tolerance_pips}")

        pip_value = 0.01
        lot_ratios = [1, 3, 3, 5]
        max_nanpin = 3
        lookback = 50

        trades = []
        in_trade = False
        last_exit_idx = 0
        trade_count = 0

        for idx in range(lookback + 200, len(self.df) - 500):
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

                # TP/SL判定
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
                    trades.append({
                        'entry_time': entries[0]['time'],
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'pips': pips,
                        'weighted_pips': weighted_pips,
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot
                    })
                    last_exit_idx = i
                    in_trade = True
                    trade_count += 1
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

        total_weighted_pips = df['weighted_pips'].sum()
        avg_weighted_pips = df['weighted_pips'].mean()

        # 年別統計
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        yearly = df.groupby('year')['weighted_pips'].sum()

        # ドローダウン
        cumulative = df['weighted_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        # PF
        win_pips = df[df['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df[df['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else 999

        # ナンピン統計
        nanpin_stats = df.groupby('nanpin_count').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'weighted_pips': 'sum'
        }).round(2)

        return {
            'name': name,
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': round(win_rate, 2),
            'total_pips': round(total_weighted_pips, 1),
            'avg_pips': round(avg_weighted_pips, 2),
            'max_dd': round(max_dd, 1),
            'pf': round(pf, 2),
            'yearly': yearly,
            'nanpin_stats': nanpin_stats,
            'df_trades': df,
            'cumulative': cumulative
        }

    def run_all_backtests(self, params_list):
        """全パラメータでバックテスト実行"""
        if self.df is None:
            self.generate_10year_data()

        self.results = {}
        for params in params_list:
            name = params['name']
            trades = self.run_backtest(
                params['tp'], params['sl'],
                params['nanpin'], params['tolerance'],
                name=name
            )
            stats = self.calculate_stats(trades, name)
            if stats:
                self.results[name] = stats

        return self.results

    def print_comparison(self):
        """結果比較表示"""
        print("\n" + "="*100)
        print("  過去10年バックテスト結果比較")
        print("="*100)
        print(f"{'Rank':<6}{'TP':<5}{'SL':<5}{'Nanpin':<8}{'Tol':<5}{'Trades':<8}{'WinRate':<9}{'TotalPips':<12}{'MaxDD':<10}{'PF':<6}{'AvgPips':<10}")
        print("-"*100)

        for name, stats in self.results.items():
            # パラメータを名前から抽出
            parts = name.split('_')
            tp = parts[1].replace('TP', '')
            sl = parts[2].replace('SL', '')
            nanpin = parts[3].replace('N', '')
            tol = parts[4].replace('T', '')

            print(f"{parts[0]:<6}{tp:<5}{sl:<5}{nanpin:<8}{tol:<5}"
                  f"{stats['trades']:<8}{stats['win_rate']:<9}"
                  f"{stats['total_pips']:<12}{stats['max_dd']:<10}"
                  f"{stats['pf']:<6}{stats['avg_pips']:<10}")

        print("="*100)

    def visualize_comparison(self, save_path):
        """比較グラフ作成"""
        n_params = len(self.results)
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle('10-Year Backtest Comparison (Top 5 Parameters)', fontsize=16, fontweight='bold')

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

        # 1. 累積損益比較
        ax1 = fig.add_subplot(2, 2, 1)
        for i, (name, stats) in enumerate(self.results.items()):
            df = stats['df_trades']
            cumulative = df['weighted_pips'].cumsum()
            label = name.replace('Rank', 'R').replace('_', ' ')
            ax1.plot(df['entry_time'], cumulative, color=colors[i], linewidth=1.5, label=label, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title('Cumulative Weighted Pips (10 Years)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pips')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 2. 年別パフォーマンス比較
        ax2 = fig.add_subplot(2, 2, 2)
        x = np.arange(10)  # 10年分
        width = 0.15
        for i, (name, stats) in enumerate(self.results.items()):
            yearly = stats['yearly']
            years = sorted(yearly.index)[-10:]  # 直近10年
            values = [yearly.get(y, 0) for y in years]
            offset = (i - n_params/2 + 0.5) * width
            label = name.split('_')[0]
            ax2.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(y) for y in years], rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Yearly Performance Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 主要指標比較
        ax3 = fig.add_subplot(2, 2, 3)
        metrics = ['Total Pips', 'Win Rate', 'PF', 'Max DD']
        x = np.arange(len(metrics))
        width = 0.15

        for i, (name, stats) in enumerate(self.results.items()):
            # 正規化して比較しやすく
            values = [
                stats['total_pips'] / 100,  # スケール調整
                stats['win_rate'],
                stats['pf'] * 20,  # スケール調整
                abs(stats['max_dd']) / 100  # 絶対値でスケール調整
            ]
            offset = (i - n_params/2 + 0.5) * width
            label = name.split('_')[0]
            ax3.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(['Pips/100', 'WinRate%', 'PF×20', '|DD|/100'])
        ax3.set_title('Key Metrics Comparison (Normalized)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. サマリーテーブル
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        table_data = []
        for name, stats in self.results.items():
            parts = name.split('_')
            row = [
                parts[0],
                f"{stats['trades']}",
                f"{stats['win_rate']}%",
                f"{stats['total_pips']:+.0f}",
                f"{stats['max_dd']:.0f}",
                f"{stats['pf']:.2f}"
            ]
            table_data.append(row)

        table = ax4.table(
            cellText=table_data,
            colLabels=['Rank', 'Trades', 'WinRate', 'TotalPips', 'MaxDD', 'PF'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # ヘッダー色
        for j in range(6):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # ベスト行をハイライト
        best_idx = max(range(len(table_data)),
                      key=lambda i: float(table_data[i][3].replace('+', '')))
        for j in range(6):
            table[(best_idx + 1, j)].set_facecolor('#C6EFCE')

        ax4.set_title('Summary Table (Best highlighted in green)', fontsize=12,
                     fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n比較グラフを保存しました: {save_path}")

    def save_individual_charts(self):
        """個別チャート保存"""
        for name, stats in self.results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{name} - 10 Year Backtest', fontsize=14, fontweight='bold')

            df = stats['df_trades']

            # 1. 累積損益
            ax1 = axes[0, 0]
            cumulative = df['weighted_pips'].cumsum()
            ax1.plot(df['entry_time'], cumulative, 'b-', linewidth=1.5)
            ax1.fill_between(df['entry_time'], cumulative, 0,
                            where=(cumulative >= 0), color='green', alpha=0.3)
            ax1.fill_between(df['entry_time'], cumulative, 0,
                            where=(cumulative < 0), color='red', alpha=0.3)
            ax1.set_title('Cumulative Pips')
            ax1.set_ylabel('Pips')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # 2. 年別
            ax2 = axes[0, 1]
            yearly = stats['yearly']
            colors_yearly = ['green' if v > 0 else 'red' for v in yearly.values]
            ax2.bar(range(len(yearly)), yearly.values, color=colors_yearly, alpha=0.7)
            ax2.set_xticks(range(len(yearly)))
            ax2.set_xticklabels([str(y) for y in yearly.index], rotation=45)
            ax2.axhline(y=0, color='black', linestyle='--')
            ax2.set_title('Yearly Performance')
            ax2.set_ylabel('Pips')
            ax2.grid(True, alpha=0.3, axis='y')

            # 3. ドローダウン
            ax3 = axes[1, 0]
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            ax3.fill_between(df['entry_time'], drawdown, 0, color='red', alpha=0.4)
            ax3.set_title(f'Drawdown (Max: {stats["max_dd"]:.0f} pips)')
            ax3.set_ylabel('Pips')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # 4. ナンピン統計
            ax4 = axes[1, 1]
            nanpin_stats = stats['nanpin_stats']
            if len(nanpin_stats) > 0:
                x = range(len(nanpin_stats))
                ax4.bar(x, nanpin_stats['weighted_pips'], color='steelblue', alpha=0.7)
                ax4.set_xticks(x)
                ax4.set_xticklabels([f'N{i}' for i in nanpin_stats.index])
                ax4.set_title('Performance by Nanpin Count')
                ax4.set_ylabel('Pips')
                ax4.axhline(y=0, color='black', linestyle='--')
                ax4.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            filename = f'/Users/naoto/ドル円/backtest_10y_{name}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"個別チャート保存: {filename}")


def main():
    print("="*60)
    print("  過去10年バックテスト")
    print("  Top 5 パラメータ比較")
    print("="*60)

    # Top 5 パラメータ
    params_list = [
        {'name': 'Rank1_TP70_SL80_N20_T35', 'tp': 70, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank2_TP80_SL80_N20_T35', 'tp': 80, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank3_TP100_SL80_N20_T35', 'tp': 100, 'sl': 80, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank4_TP100_SL100_N20_T35', 'tp': 100, 'sl': 100, 'nanpin': 20, 'tolerance': 35},
        {'name': 'Rank5_TP80_SL100_N30_T30', 'tp': 80, 'sl': 100, 'nanpin': 30, 'tolerance': 30},
    ]

    bt = LongTermBacktester()
    bt.run_all_backtests(params_list)
    bt.print_comparison()
    bt.visualize_comparison('/Users/naoto/ドル円/backtest_10y_comparison.png')
    bt.save_individual_charts()

    print("\n完了!")


if __name__ == "__main__":
    main()
