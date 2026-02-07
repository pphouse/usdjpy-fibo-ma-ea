"""
ベストパラメータの詳細可視化
TP150/SL150/Nanpin25 (N3固定)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


class BestParameterVisualizer:
    """ベストパラメータ可視化"""

    def __init__(self):
        self.df = None
        self.trades = []
        self.params = {
            'tp': 150,
            'sl': 150,
            'nanpin': 25,
            'tolerance': 35,
            'lot_ratios': [1, 3, 3, 5],
            'max_nanpin': 3,
            'spread': 0.3
        }

    def load_data(self):
        print("H4データを読み込み中...")
        self.df = pd.read_csv(
            '/Users/naoto/ドル円/h4_real_backtest/USDJPY_H4_2015_2025.csv',
            index_col=0, parse_dates=True
        )
        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()
        print(f"データ: {len(self.df):,}本")
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

    def run_backtest(self):
        """バックテスト実行（詳細情報付き）"""
        if self.df is None:
            self.load_data()

        p = self.params
        pip_value = 0.01
        lookback = 50

        self.trades = []
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
            tolerance = p['tolerance'] * pip_value

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

            direction = 'BUY' if trend == 1 else 'SELL'
            entries = [{
                'idx': idx,
                'price': close,
                'lot': p['lot_ratios'][0],
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

                tp_dist = p['tp'] * pip_value
                sl_dist = p['sl'] * pip_value

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
                    pips = p['tp'] if result == 'WIN' else -p['sl']
                    spread_cost = sum(e['lot'] * p['spread'] for e in entries)
                    weighted_pips = pips * total_lot - spread_cost

                    self.trades.append({
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

                if nanpin_count < p['max_nanpin']:
                    nanpin_dist = p['nanpin'] * pip_value
                    if direction == 'BUY':
                        distance = last_entry_price - current_close
                    else:
                        distance = current_close - last_entry_price

                    if distance >= nanpin_dist:
                        nanpin_count += 1
                        lot = p['lot_ratios'][min(nanpin_count, len(p['lot_ratios'])-1)]
                        entries.append({
                            'idx': i,
                            'price': current_close,
                            'lot': lot,
                            'time': self.df.index[i]
                        })
                        last_entry_price = current_close

        print(f"トレード数: {len(self.trades)}")
        return self.trades

    def visualize_full_performance(self, save_path):
        """フルパフォーマンス可視化"""
        df_trades = pd.DataFrame(self.trades)
        df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year

        fig = plt.figure(figsize=(24, 20))
        fig.suptitle(f'Best Parameters Performance: TP{self.params["tp"]}/SL{self.params["sl"]}/Nanpin{self.params["nanpin"]}\n'
                    f'USDJPY H4 Real Data (Axiory 2015-2025) | N3 Fixed (1:3:3:5)',
                    fontsize=16, fontweight='bold')

        # 1. 価格チャート + エントリーポイント
        ax1 = fig.add_subplot(4, 2, 1)
        ax1.plot(self.df.index, self.df['close'], 'gray', linewidth=0.3, alpha=0.7)
        ax1.plot(self.df.index, self.df['ma_200'], 'red', linewidth=0.5, alpha=0.5)

        for trade in self.trades:
            is_buy = trade['direction'] == 'BUY'
            is_win = trade['result'] == 'WIN'
            marker = '^' if is_buy else 'v'
            color = 'lime' if is_win else 'red'
            ax1.scatter(trade['entry_time'], trade['entries'][0]['price'],
                       marker=marker, s=15, c=color, alpha=0.6)

        ax1.set_title('Price Chart with Entry Points', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)

        # 2. 累積損益
        ax2 = fig.add_subplot(4, 2, 2)
        cumulative = df_trades['weighted_pips'].cumsum()
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.4)
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.4)
        ax2.plot(df_trades['entry_time'], cumulative, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=0.5)

        # 最高点・最低点をマーク
        max_idx = cumulative.idxmax()
        min_idx = cumulative.idxmin()
        ax2.scatter(df_trades.loc[max_idx, 'entry_time'], cumulative[max_idx],
                   s=100, c='green', marker='*', zorder=5, label=f'Max: {cumulative[max_idx]:+,.0f}')
        ax2.scatter(df_trades.loc[min_idx, 'entry_time'], cumulative[min_idx],
                   s=100, c='red', marker='*', zorder=5, label=f'Min: {cumulative[min_idx]:+,.0f}')

        ax2.set_title(f'Cumulative Performance: {cumulative.iloc[-1]:+,.0f} pips', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 年別パフォーマンス
        ax3 = fig.add_subplot(4, 2, 3)
        yearly = df_trades.groupby('year')['weighted_pips'].sum()
        colors = ['green' if v > 0 else 'red' for v in yearly.values]
        bars = ax3.bar(yearly.index.astype(str), yearly.values, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--')
        ax3.set_title('Yearly Performance', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Pips')
        for bar, val in zip(bars, yearly.values):
            ax3.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 年別勝率
        ax4 = fig.add_subplot(4, 2, 4)
        yearly_wr = df_trades.groupby('year').apply(lambda x: (x['result'] == 'WIN').sum() / len(x) * 100)
        yearly_count = df_trades.groupby('year').size()
        bars = ax4.bar(yearly_wr.index.astype(str), yearly_wr.values, color='steelblue', alpha=0.7)
        ax4.axhline(y=50, color='red', linestyle='--', label='50%')
        ax4.set_title('Yearly Win Rate', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_ylim(0, 100)
        for bar, val, cnt in zip(bars, yearly_wr.values, yearly_count.values):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 2,
                    f'{val:.0f}%\n({cnt})', ha='center', fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. ナンピン回数別パフォーマンス
        ax5 = fig.add_subplot(4, 2, 5)
        nanpin_stats = df_trades.groupby('nanpin_count').agg({
            'weighted_pips': 'sum',
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'pips': 'count'
        })
        nanpin_stats.columns = ['total_pips', 'win_rate', 'count']

        colors = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c'][:len(nanpin_stats)]
        bars = ax5.bar([f'N{i}' for i in nanpin_stats.index], nanpin_stats['total_pips'], color=colors, alpha=0.8)
        ax5.axhline(y=0, color='black', linestyle='--')
        ax5.set_title('Performance by Nanpin Count', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Total Pips')
        for bar, (idx, row) in zip(bars, nanpin_stats.iterrows()):
            val = row['total_pips']
            ax5.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:+,.0f}\n({int(row["count"])})\n{row["win_rate"]:.0f}%',
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. ドローダウン
        ax6 = fig.add_subplot(4, 2, 6)
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        ax6.fill_between(df_trades['entry_time'], drawdown, 0, color='red', alpha=0.4)
        ax6.plot(df_trades['entry_time'], drawdown, 'r-', linewidth=0.8)
        ax6.set_title(f'Drawdown (Max: {drawdown.min():,.0f} pips)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Pips')
        ax6.grid(True, alpha=0.3)

        # 7. 勝敗分布
        ax7 = fig.add_subplot(4, 2, 7)
        wins = (df_trades['result'] == 'WIN').sum()
        losses = (df_trades['result'] == 'LOSS').sum()
        ax7.pie([wins, losses], labels=[f'WIN ({wins})', f'LOSS ({losses})'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90,
               explode=(0.05, 0.05))
        ax7.set_title(f'Win/Loss Distribution (Total: {len(df_trades)})', fontsize=11, fontweight='bold')

        # 8. サマリーテーブル
        ax8 = fig.add_subplot(4, 2, 8)
        ax8.axis('off')

        total_pips = df_trades['weighted_pips'].sum()
        profit_001 = total_pips * 10
        profit_01 = total_pips * 100
        profit_1 = total_pips * 1000

        summary_text = f"""
【パラメータ】
  TP: {self.params['tp']} pips
  SL: {self.params['sl']} pips
  Nanpin Interval: {self.params['nanpin']} pips
  Max Nanpin: {self.params['max_nanpin']}回 (1:3:3:5)
  Spread: {self.params['spread']} pips

【パフォーマンス】
  期間: 2015-2025 (10年間)
  トレード数: {len(df_trades)}回
  勝率: {wins/len(df_trades)*100:.1f}%
  Net Pips: {total_pips:+,.0f}
  Max Drawdown: {drawdown.min():,.0f} pips
  Profit Factor: {df_trades[df_trades['weighted_pips']>0]['weighted_pips'].sum() / abs(df_trades[df_trades['weighted_pips']<0]['weighted_pips'].sum()):.2f}

【10年間利益】
  0.01 lot: {profit_001:+,.0f}円
  0.1 lot: {profit_01:+,.0f}円
  1.0 lot: {profit_1:+,.0f}円
"""
        ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes,
                fontsize=12, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax8.set_title('Summary', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")

    def visualize_monthly(self, save_path):
        """月別パフォーマンス"""
        df_trades = pd.DataFrame(self.trades)
        df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')

        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('Monthly Performance Analysis', fontsize=14, fontweight='bold')

        # 1. 月別損益
        ax1 = axes[0]
        monthly = df_trades.groupby('month')['weighted_pips'].sum()
        colors = ['green' if v > 0 else 'red' for v in monthly.values]
        ax1.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_title('Monthly Net Pips', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Pips')

        # 年ごとに縦線
        year_changes = []
        for i, m in enumerate(monthly.index):
            if i > 0 and m.year != monthly.index[i-1].year:
                ax1.axvline(x=i-0.5, color='blue', linestyle='--', alpha=0.5)
                ax1.text(i, ax1.get_ylim()[1], str(m.year), fontsize=9, ha='left')

        ax1.set_xticks(range(0, len(monthly), 12))
        ax1.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), 12)], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 累積（月別）
        ax2 = axes[1]
        cumulative_monthly = monthly.cumsum()
        ax2.fill_between(range(len(cumulative_monthly)), cumulative_monthly.values, 0,
                        where=(cumulative_monthly.values >= 0), color='green', alpha=0.4)
        ax2.fill_between(range(len(cumulative_monthly)), cumulative_monthly.values, 0,
                        where=(cumulative_monthly.values < 0), color='red', alpha=0.4)
        ax2.plot(range(len(cumulative_monthly)), cumulative_monthly.values, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_title('Cumulative Monthly Performance', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.set_xticks(range(0, len(monthly), 12))
        ax2.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), 12)], rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")

    def visualize_trade_samples(self, save_path):
        """トレードサンプル"""
        df_trades = pd.DataFrame(self.trades)

        # サンプル選択
        samples = []
        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']

        if wins:
            samples.append(('Best WIN', max(wins, key=lambda x: x['weighted_pips'])))
        if losses:
            samples.append(('Worst LOSS', min(losses, key=lambda x: x['weighted_pips'])))

        # ナンピン別
        for n in [0, 1, 2, 3]:
            n_trades = [t for t in self.trades if t['nanpin_count'] == n]
            if n_trades:
                samples.append((f'N{n} Sample', n_trades[0]))
                if len(samples) >= 6:
                    break

        n_samples = len(samples)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Trade Samples (Best Parameters)', fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for idx, (label, trade) in enumerate(samples):
            if idx >= 6:
                break
            ax = axes[idx]

            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            margin = 20
            start_idx = max(0, entry_idx - margin)
            end_idx = min(len(self.df), exit_idx + margin)
            chart_df = self.df.iloc[start_idx:end_idx]

            # ローソク足
            for i in range(len(chart_df)):
                row = chart_df.iloc[i]
                date = chart_df.index[i]
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                color = 'green' if c >= o else 'red'
                ax.plot([date, date], [l, h], color=color, linewidth=0.5)
                ax.plot([date, date], [min(o,c), max(o,c)], color=color, linewidth=2)

            # エントリー
            is_buy = trade['direction'] == 'BUY'
            marker = '^' if is_buy else 'v'
            for i, e in enumerate(trade['entries']):
                c = 'lime' if i == 0 else 'yellow'
                ax.scatter(e['time'], e['price'], marker=marker, s=80, c=c,
                          edgecolors='black', linewidth=1, zorder=5)

            # 決済
            exit_color = 'lime' if trade['result'] == 'WIN' else 'red'
            ax.scatter(trade['exit_time'], trade['exit_price'], marker='X', s=100,
                      c=exit_color, edgecolors='black', linewidth=1, zorder=5)

            title_color = 'green' if trade['result'] == 'WIN' else 'red'
            ax.set_title(f"{label}: {trade['direction']} | {trade['result']} "
                        f"({trade['weighted_pips']:+,.0f}pips) | N{trade['nanpin_count']}",
                        fontsize=10, fontweight='bold', color=title_color)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 残りを非表示
        for idx in range(len(samples), 6):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")


def main():
    print("="*70)
    print("  ベストパラメータ 詳細可視化")
    print("  TP150/SL150/Nanpin25 (N3固定)")
    print("="*70)

    viz = BestParameterVisualizer()
    viz.run_backtest()

    print("\n可視化を生成中...")
    viz.visualize_full_performance('/Users/naoto/ドル円/h4_real_backtest/best_performance.png')
    viz.visualize_monthly('/Users/naoto/ドル円/h4_real_backtest/best_monthly.png')
    viz.visualize_trade_samples('/Users/naoto/ドル円/h4_real_backtest/best_samples.png')

    print("\n完了!")


if __name__ == "__main__":
    main()
