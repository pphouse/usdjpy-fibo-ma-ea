"""
実データでのエントリーポイント可視化
USDJPY日足10年 + エントリー/決済ポイント表示
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False


def download_usdjpy_data():
    """yfinanceからUSDJPY日足データを取得"""
    import yfinance as yf
    print("USDJPYデータをダウンロード中...")
    ticker = yf.Ticker("USDJPY=X")
    df = ticker.history(period="10y", interval="1d")
    df = df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'
    })
    df.index = df.index.tz_localize(None)
    print(f"データ取得完了: {len(df)}本 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


class EntryVisualizer:
    """エントリーポイント可視化"""

    def __init__(self, spread_pips=0.3):
        self.spread_pips = spread_pips
        self.df = None
        self.all_trades = []

    def load_data(self):
        self.df = download_usdjpy_data()
        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()
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
        if range_val < 0.5:
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

    def run_backtest(self, tp_pips=150, sl_pips=150, nanpin_interval=30,
                     tolerance_pips=50, max_nanpin=2, lot_ratios=None):
        """バックテスト実行（トレード詳細を保存）"""
        if lot_ratios is None:
            lot_ratios = [1, 3, 3]

        if self.df is None:
            self.load_data()

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
            ma_price = None
            for period in [20, 50, 100]:
                ma_val = self.df[f'ma_{period}'].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = period
                    ma_price = ma_val
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

            for i in range(idx + 1, min(idx + 200, len(self.df))):
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
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'pips': pips,
                        'weighted_pips': weighted_pips,
                        'result': result,
                        'nanpin_count': nanpin_count,
                        'total_lots': total_lot,
                        'fibo_hit': fibo_hit,
                        'fibo_price': fibo_price,
                        'fibo_high': fibo_high,
                        'fibo_low': fibo_low,
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

        self.all_trades = trades
        print(f"トレード数: {len(trades)}")
        return trades

    def visualize_overview(self, save_path):
        """全体概要チャート"""
        if not self.all_trades:
            print("トレードがありません")
            return

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('USDJPY Real Data - Entry Points Overview (Daily, 10 Years)\n'
                    'Data Source: Yahoo Finance | N2 Limit (1:3:3)',
                    fontsize=14, fontweight='bold')

        # 1. 価格チャート + エントリーポイント
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(self.df.index, self.df['close'], 'gray', linewidth=0.8, alpha=0.7, label='Price')
        ax1.plot(self.df.index, self.df['ma_50'], 'blue', linewidth=0.5, alpha=0.5, label='MA50')
        ax1.plot(self.df.index, self.df['ma_200'], 'red', linewidth=0.8, alpha=0.7, label='MA200')

        # エントリーポイント
        for trade in self.all_trades:
            is_buy = trade['direction'] == 'BUY'
            is_win = trade['result'] == 'WIN'
            marker = '^' if is_buy else 'v'
            entry_color = 'lime' if is_buy else 'magenta'

            # 初回エントリー
            ax1.scatter(trade['entry_time'], trade['entries'][0]['price'],
                       marker=marker, s=60, c=entry_color,
                       edgecolors='black', linewidth=0.5, zorder=5)

            # ナンピン
            for e in trade['entries'][1:]:
                ax1.scatter(e['time'], e['price'], marker=marker, s=40,
                           c='yellow', edgecolors=entry_color, linewidth=0.5, zorder=5)

            # 決済
            exit_color = 'green' if is_win else 'red'
            ax1.scatter(trade['exit_time'], trade['exit_price'],
                       marker='x', s=50, c=exit_color, linewidth=1.5, zorder=5)

        ax1.set_ylabel('Price (JPY)')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Price Chart with Entry/Exit Points', fontsize=11, fontweight='bold')

        # 2. 累積損益
        ax2 = fig.add_subplot(3, 1, 2)
        df_trades = pd.DataFrame(self.all_trades)
        cumulative = df_trades['weighted_pips'].cumsum()
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.4)
        ax2.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.4)
        ax2.plot(df_trades['entry_time'], cumulative, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_ylabel('Cumulative Pips')
        ax2.set_title('Cumulative Performance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. トレード分布
        ax3 = fig.add_subplot(3, 2, 5)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        ax3.pie([wins, losses], labels=[f'WIN ({wins})', f'LOSS ({losses})'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
        ax3.set_title('Win/Loss Distribution', fontsize=11, fontweight='bold')

        # 4. ナンピン回数別
        ax4 = fig.add_subplot(3, 2, 6)
        nanpin_stats = df_trades.groupby('nanpin_count').agg({
            'weighted_pips': 'sum',
            'result': 'count'
        })
        colors = ['#3498db', '#9b59b6', '#e67e22'][:len(nanpin_stats)]
        bars = ax4.bar(nanpin_stats.index, nanpin_stats['weighted_pips'], color=colors)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Nanpin Count')
        ax4.set_ylabel('Total Pips')
        ax4.set_title('Performance by Nanpin Count', fontsize=11, fontweight='bold')
        for bar, (idx, row) in zip(bars, nanpin_stats.iterrows()):
            y = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, y,
                    f'{y:+,.0f}\n({int(row["result"])})',
                    ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存しました: {save_path}")

    def visualize_sample_trades(self, save_path, num_samples=6):
        """サンプルトレードの詳細"""
        if not self.all_trades:
            return

        # WIN/LOSSから均等にサンプル
        wins = [t for t in self.all_trades if t['result'] == 'WIN']
        losses = [t for t in self.all_trades if t['result'] == 'LOSS']

        samples = []
        if wins:
            samples.append(('Best WIN', max(wins, key=lambda x: x['weighted_pips'])))
        if losses:
            samples.append(('Worst LOSS', min(losses, key=lambda x: x['weighted_pips'])))

        # ナンピンあり
        nanpin_trades = [t for t in self.all_trades if t['nanpin_count'] > 0]
        if nanpin_trades:
            samples.append(('With Nanpin (WIN)', next((t for t in nanpin_trades if t['result'] == 'WIN'), nanpin_trades[0])))

        # 追加サンプル
        for i, t in enumerate(self.all_trades[:3]):
            if len(samples) < num_samples:
                samples.append((f'Trade #{i+1}', t))

        n_cols = 2
        n_rows = (len(samples) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        fig.suptitle('Sample Trade Details (USDJPY Daily, Real Data)', fontsize=14, fontweight='bold')

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (label, trade) in enumerate(samples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']

            margin = 15
            start_idx = max(0, entry_idx - margin)
            end_idx = min(len(self.df), exit_idx + margin)
            chart_df = self.df.iloc[start_idx:end_idx]

            # ローソク足風
            for i in range(len(chart_df)):
                row_data = chart_df.iloc[i]
                date = chart_df.index[i]
                o, h, l, c = row_data['open'], row_data['high'], row_data['low'], row_data['close']
                color = 'green' if c >= o else 'red'
                ax.plot([date, date], [l, h], color=color, linewidth=0.8)
                ax.plot([date, date], [min(o, c), max(o, c)], color=color, linewidth=3)

            # MA
            ax.plot(chart_df.index, chart_df['ma_50'], 'blue', linewidth=1, alpha=0.5)
            ax.plot(chart_df.index, chart_df['ma_200'], 'red', linewidth=1, alpha=0.7)

            # フィボナッチ
            ax.axhline(y=trade['fibo_price'], color='gold', linestyle='-', linewidth=1.5, alpha=0.7)

            # エントリー
            is_buy = trade['direction'] == 'BUY'
            marker = '^' if is_buy else 'v'
            for i, e in enumerate(trade['entries']):
                color = 'lime' if i == 0 else 'yellow'
                ax.scatter(e['time'], e['price'], marker=marker, s=100,
                          c=color, edgecolors='black', linewidth=1, zorder=10)

            # TP/SL
            ax.axhline(y=trade['tp_price'], color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=trade['sl_price'], color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=trade['avg_price'], color='blue', linestyle=':', linewidth=1.5, alpha=0.8)

            # 決済
            exit_color = 'lime' if trade['result'] == 'WIN' else 'red'
            ax.scatter(trade['exit_time'], trade['exit_price'], marker='X', s=120,
                      c=exit_color, edgecolors='black', linewidth=1, zorder=10)

            title_color = 'green' if trade['result'] == 'WIN' else 'red'
            title = (f"{label}: {trade['direction']} | {trade['result']} "
                    f"({trade['weighted_pips']:+.0f}pips) | N{trade['nanpin_count']}")
            ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # 余分なサブプロット非表示
        for idx in range(len(samples), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存しました: {save_path}")

    def visualize_yearly_analysis(self, save_path):
        """年別分析"""
        if not self.all_trades:
            return

        df_trades = pd.DataFrame(self.all_trades)
        df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Yearly Performance Analysis (USDJPY Daily, Real Data)', fontsize=14, fontweight='bold')

        # 1. 年別損益
        ax1 = axes[0, 0]
        yearly_pips = df_trades.groupby('year')['weighted_pips'].sum()
        colors = ['green' if x > 0 else 'red' for x in yearly_pips.values]
        bars = ax1.bar(yearly_pips.index.astype(str), yearly_pips.values, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_title('Yearly Net Pips', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Pips')
        for bar, val in zip(bars, yearly_pips.values):
            ax1.text(bar.get_x() + bar.get_width()/2, val,
                    f'{val:+,.0f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 年別勝率
        ax2 = axes[0, 1]
        yearly_wr = df_trades.groupby('year').apply(
            lambda x: (x['result'] == 'WIN').sum() / len(x) * 100
        )
        ax2.bar(yearly_wr.index.astype(str), yearly_wr.values, color='steelblue', alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', label='50%')
        ax2.set_title('Yearly Win Rate', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 年別トレード数
        ax3 = axes[1, 0]
        yearly_count = df_trades.groupby('year').size()
        ax3.bar(yearly_count.index.astype(str), yearly_count.values, color='purple', alpha=0.7)
        ax3.set_title('Yearly Trade Count', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Trades')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. 年別ナンピン使用率
        ax4 = axes[1, 1]
        yearly_nanpin = df_trades.groupby('year').apply(
            lambda x: (x['nanpin_count'] > 0).sum() / len(x) * 100
        )
        ax4.bar(yearly_nanpin.index.astype(str), yearly_nanpin.values, color='orange', alpha=0.7)
        ax4.set_title('Yearly Nanpin Usage Rate', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Usage (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存しました: {save_path}")


def main():
    print("="*70)
    print("  実データ エントリーポイント可視化")
    print("  USDJPY 日足 10年 / N2制限 (1:3:3)")
    print("="*70)

    viz = EntryVisualizer(spread_pips=0.3)
    viz.run_backtest(
        tp_pips=150, sl_pips=150,
        nanpin_interval=30, tolerance_pips=50,
        max_nanpin=2, lot_ratios=[1, 3, 3]
    )

    print("\n可視化を生成中...")
    viz.visualize_overview('/Users/naoto/ドル円/real_entries_overview.png')
    viz.visualize_sample_trades('/Users/naoto/ドル円/real_sample_trades.png')
    viz.visualize_yearly_analysis('/Users/naoto/ドル円/real_yearly_analysis.png')

    # 統計サマリー
    df = pd.DataFrame(viz.all_trades)
    total = len(df)
    wins = (df['result'] == 'WIN').sum()
    net = df['weighted_pips'].sum()

    print("\n" + "="*70)
    print("  サマリー")
    print("="*70)
    print(f"  トレード数: {total}")
    print(f"  勝率: {wins/total*100:.1f}%")
    print(f"  Net Pips: {net:+,.0f}")
    print(f"  利益 (0.1lot): {net * 100:+,.0f}円")
    print("="*70)

    print("\n完了!")


if __name__ == "__main__":
    main()
