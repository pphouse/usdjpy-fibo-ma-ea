"""
H1ベストパラメータ 詳細可視化
TP120/SL100/Nanpin25
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, header=None,
                           names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
            dfs.append(df)
        except:
            pass

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['datetime'] = pd.to_datetime(df_all['date'] + ' ' + df_all['time'],
                                        format='%Y.%m.%d %H:%M')
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    df_all = df_all.set_index('datetime')
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    return df_all[['open', 'high', 'low', 'close', 'volume']]


def resample_to_h1(df_m1):
    """M1をH1に変換"""
    return df_m1.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


class H1BestVisualizer:
    """ベストパラメータの詳細可視化"""

    def __init__(self):
        self.tp_pips = 120
        self.sl_pips = 100
        self.nanpin_interval = 25
        self.spread_pips = 0.3
        self.lot_ratios = [1, 3, 3, 5]
        self.max_nanpin = 3
        self.df = None
        self.trades = []

    def load_data(self):
        """データ読み込み"""
        print("M1データを読み込み中...")
        df_m1 = load_m1_data()
        print("H1に変換中...")
        self.df = resample_to_h1(df_m1)

        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

        print(f"データ: {len(self.df):,}本")
        return self.df

    def calculate_fibonacci(self, idx, lookback=100):
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

    def run_backtest(self):
        """バックテスト実行"""
        if self.df is None:
            self.load_data()

        pip_value = 0.01
        lookback = 100
        tolerance_pips = 25
        tolerance = tolerance_pips * pip_value

        self.trades = []
        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(self.df) - 200):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend = self.calculate_fibonacci(idx, lookback)
            if trend == 0:
                continue

            close = self.df['close'].iloc[idx]

            fibo_hit = abs(close - fibo_382) < tolerance or abs(close - fibo_618) < tolerance
            if not fibo_hit:
                continue

            ma_hit = False
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
            entry_time = self.df.index[idx]
            entries = [{'price': close, 'lot': self.lot_ratios[0], 'time': entry_time}]
            last_entry_price = close
            nanpin_count = 0

            for i in range(idx + 1, min(idx + 1000, len(self.df))):
                current_high = self.df['high'].iloc[i]
                current_low = self.df['low'].iloc[i]
                current_close = self.df['close'].iloc[i]

                total_lot = sum(e['lot'] for e in entries)
                avg_price = sum(e['price'] * e['lot'] for e in entries) / total_lot

                tp_dist = self.tp_pips * pip_value
                sl_dist = self.sl_pips * pip_value

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
                    pips = self.tp_pips if result == 'WIN' else -self.sl_pips
                    spread_cost = sum(e['lot'] * self.spread_pips for e in entries)
                    weighted_pips = pips * total_lot - spread_cost

                    self.trades.append({
                        'entry_time': entry_time,
                        'exit_time': self.df.index[i],
                        'direction': direction,
                        'entry_price': entries[0]['price'],
                        'avg_price': avg_price,
                        'exit_price': tp_price if result == 'WIN' else sl_price,
                        'result': result,
                        'pips': pips,
                        'weighted_pips': weighted_pips,
                        'total_lot': total_lot,
                        'nanpin_count': nanpin_count,
                        'entries': entries.copy()
                    })
                    last_exit_idx = i
                    in_trade = True
                    break

                if nanpin_count < self.max_nanpin:
                    nanpin_dist = self.nanpin_interval * pip_value
                    if direction == 'BUY':
                        distance = last_entry_price - current_close
                    else:
                        distance = current_close - last_entry_price

                    if distance >= nanpin_dist:
                        nanpin_count += 1
                        lot = self.lot_ratios[min(nanpin_count, len(self.lot_ratios)-1)]
                        entries.append({
                            'price': current_close,
                            'lot': lot,
                            'time': self.df.index[i]
                        })
                        last_entry_price = current_close

        print(f"トレード数: {len(self.trades)}")
        return self.trades

    def visualize_full_performance(self, save_path):
        """フルパフォーマンスダッシュボード"""
        df_trades = pd.DataFrame(self.trades)

        fig = plt.figure(figsize=(24, 18))
        fig.suptitle(f'H1 Best Parameters: TP{self.tp_pips}/SL{self.sl_pips}/Nanpin{self.nanpin_interval}\n'
                    f'Axiory 2015-2025 Real Data', fontsize=16, fontweight='bold')

        # 1. 累積損益
        ax1 = fig.add_subplot(3, 2, 1)
        df_trades['cumulative'] = df_trades['weighted_pips'].cumsum()
        ax1.fill_between(range(len(df_trades)), df_trades['cumulative'],
                        where=df_trades['cumulative'] >= 0, alpha=0.5, color='green')
        ax1.fill_between(range(len(df_trades)), df_trades['cumulative'],
                        where=df_trades['cumulative'] < 0, alpha=0.5, color='red')
        ax1.plot(df_trades['cumulative'], 'b-', linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Cumulative Pips')
        ax1.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 年別パフォーマンス
        ax2 = fig.add_subplot(3, 2, 2)
        df_trades['year'] = pd.to_datetime(df_trades['entry_time']).dt.year
        yearly = df_trades.groupby('year')['weighted_pips'].sum()
        colors = ['green' if v >= 0 else 'red' for v in yearly.values]
        bars = ax2.bar(yearly.index.astype(str), yearly.values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, yearly.values):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:+,.0f}',
                    ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Net Pips')
        ax2.set_title('Yearly Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. ナンピン分析
        ax3 = fig.add_subplot(3, 2, 3)
        nanpin_perf = df_trades.groupby('nanpin_count').agg({
            'weighted_pips': ['sum', 'mean', 'count']
        })
        nanpin_perf.columns = ['total_pips', 'avg_pips', 'count']

        x = np.arange(len(nanpin_perf))
        width = 0.35
        bars1 = ax3.bar(x - width/2, nanpin_perf['total_pips'], width, label='Total Pips', alpha=0.7)
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, nanpin_perf['count'], width, label='Count', color='orange', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'N{i}' for i in nanpin_perf.index])
        ax3.set_xlabel('Nanpin Level')
        ax3.set_ylabel('Total Pips')
        ax3_twin.set_ylabel('Trade Count')
        ax3.set_title('Performance by Nanpin Level', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. ドローダウン
        ax4 = fig.add_subplot(3, 2, 4)
        cummax = df_trades['cumulative'].expanding().max()
        drawdown = df_trades['cumulative'] - cummax
        ax4.fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.5)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        ax4.scatter([max_dd_idx], [max_dd], color='darkred', s=100, zorder=5)
        ax4.annotate(f'Max DD: {max_dd:,.0f}', xy=(max_dd_idx, max_dd),
                    xytext=(max_dd_idx+20, max_dd+500), fontsize=10)
        ax4.set_xlabel('Trade #')
        ax4.set_ylabel('Drawdown (pips)')
        ax4.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. 勝敗分布
        ax5 = fig.add_subplot(3, 2, 5)
        win_count = (df_trades['result'] == 'WIN').sum()
        loss_count = (df_trades['result'] == 'LOSS').sum()
        ax5.pie([win_count, loss_count], labels=['WIN', 'LOSS'],
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax5.set_title(f'Win/Loss Distribution (Win Rate: {win_count/(win_count+loss_count)*100:.1f}%)',
                     fontsize=12, fontweight='bold')

        # 6. サマリー
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')

        total_pips = df_trades['weighted_pips'].sum()
        win_rate = win_count / len(df_trades) * 100
        avg_win = df_trades[df_trades['result'] == 'WIN']['weighted_pips'].mean()
        avg_loss = abs(df_trades[df_trades['result'] == 'LOSS']['weighted_pips'].mean())
        pf = df_trades[df_trades['weighted_pips'] > 0]['weighted_pips'].sum() / abs(df_trades[df_trades['weighted_pips'] < 0]['weighted_pips'].sum())
        profit_01lot = total_pips * 100

        summary_text = f"""
═══════════════════════════════════════════
               PERFORMANCE SUMMARY
═══════════════════════════════════════════

  Parameters:
    TP: {self.tp_pips} pips
    SL: {self.sl_pips} pips
    Nanpin: {self.nanpin_interval} pips
    Max Nanpin: N{self.max_nanpin}

  Results:
    Total Trades: {len(df_trades):,}
    Win Rate: {win_rate:.1f}%
    Net Pips: {total_pips:+,.0f}
    Max Drawdown: {max_dd:,.0f}
    Profit Factor: {pf:.2f}

  Average:
    Avg Win: {avg_win:+,.1f} pips
    Avg Loss: {avg_loss:,.1f} pips

  Profit (10 years):
    0.1 lot: ¥{profit_01lot:+,.0f}
    1.0 lot: ¥{profit_01lot*10:+,.0f}

═══════════════════════════════════════════
"""
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")

    def visualize_monthly(self, save_path):
        """月次パフォーマンス"""
        df_trades = pd.DataFrame(self.trades)
        df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')

        monthly = df_trades.groupby('month')['weighted_pips'].sum()

        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle('Monthly Performance Analysis', fontsize=14, fontweight='bold')

        ax1 = axes[0]
        colors = ['green' if v >= 0 else 'red' for v in monthly.values]
        ax1.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        years = monthly.index.year.unique()
        for year in years:
            idx = np.where(monthly.index.year == year)[0]
            if len(idx) > 0:
                ax1.axvline(x=idx[0]-0.5, color='blue', linestyle='--', alpha=0.3)
                ax1.text(idx[0], ax1.get_ylim()[1]*0.95, str(year), fontsize=10)

        ax1.set_xlabel('Month')
        ax1.set_ylabel('Pips')
        ax1.set_title('Monthly Net Pips', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        cumulative = monthly.cumsum()
        ax2.fill_between(range(len(cumulative)), cumulative.values,
                        where=cumulative.values >= 0, alpha=0.5, color='green')
        ax2.fill_between(range(len(cumulative)), cumulative.values,
                        where=cumulative.values < 0, alpha=0.5, color='red')
        ax2.plot(cumulative.values, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Pips')
        ax2.set_title('Cumulative Monthly Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {save_path}")


def main():
    print("="*70)
    print("  ベストパラメータ 詳細可視化")
    print("  TP120/SL100/Nanpin25 (N3固定)")
    print("="*70)

    viz = H1BestVisualizer()
    viz.load_data()
    viz.run_backtest()

    print("\n可視化を生成中...")
    viz.visualize_full_performance('/Users/naoto/ドル円/h1_real_backtest/best_performance.png')
    viz.visualize_monthly('/Users/naoto/ドル円/h1_real_backtest/best_monthly.png')

    print("\n完了!")


if __name__ == "__main__":
    main()
