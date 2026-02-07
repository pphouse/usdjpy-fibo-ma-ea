"""
実際のUSDJPYデータを使用したバックテスト
yfinanceから日足10年分を取得
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


def download_usdjpy_data():
    """yfinanceからUSDJPY日足データを取得"""
    import yfinance as yf

    print("実際のUSDJPYデータをダウンロード中...")
    ticker = yf.Ticker("USDJPY=X")
    df = ticker.history(period="10y", interval="1d")

    # カラム名を統一
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # タイムゾーン情報を削除
    df.index = df.index.tz_localize(None)

    print(f"データ取得完了: {len(df)}本")
    print(f"期間: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"価格レンジ: {df['low'].min():.2f} ~ {df['high'].max():.2f}")

    return df


class RealDataBacktester:
    """実データバックテスター"""

    def __init__(self, spread_pips=0.3):
        self.spread_pips = spread_pips
        self.df = None
        self.results = {}

    def load_data(self):
        """データ読み込み"""
        self.df = download_usdjpy_data()

        # MA計算
        for period in [20, 50, 100, 200]:
            self.df[f'ma_{period}'] = self.df['close'].rolling(window=period).mean()

        return self.df

    def calculate_fibonacci(self, idx, lookback=50):
        if idx < lookback:
            return None, None, 0

        window = self.df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.5:  # 日足なので最小レンジを調整
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

        if self.df is None:
            self.load_data()

        spread = self.spread_pips
        print(f"\n[{name}] TP={tp_pips}, SL={sl_pips}, MaxNanpin={max_nanpin}")

        pip_value = 0.01
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

    def run_comparison(self):
        """パラメータ比較"""
        print("\n" + "="*70)
        print("  実データバックテスト (USDJPY 日足 10年)")
        print("  データソース: Yahoo Finance")
        print("="*70)

        # 日足用にパラメータ調整（4H比で約1.5倍）
        configs = [
            {'name': 'Rank4_N3', 'tp': 150, 'sl': 150, 'interval': 30, 'tol': 50,
             'max_nanpin': 3, 'lots': [1, 3, 3, 5]},
            {'name': 'Rank4_N2', 'tp': 150, 'sl': 150, 'interval': 30, 'tol': 50,
             'max_nanpin': 2, 'lots': [1, 3, 3]},
            {'name': 'Rank4_N0', 'tp': 150, 'sl': 150, 'interval': 30, 'tol': 50,
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
                self.results[cfg['name']] = stats

        return self.results

    def print_results(self):
        """結果表示"""
        print("\n" + "="*100)
        print("  実データバックテスト結果 (USDJPY 日足)")
        print("="*100)
        print(f"{'設定':<15}{'Trades':<8}{'WR%':<8}{'Gross':<12}{'Spread':<10}{'Net':<12}{'MaxDD':<10}{'PF':<6}")
        print("-"*100)

        for name, stats in self.results.items():
            print(f"{name:<15}{stats['trades']:<8}{stats['win_rate']:<8}"
                  f"{stats['total_gross']:<12.0f}{stats['total_spread']:<10.0f}"
                  f"{stats['total_pips']:<12.0f}{stats['max_dd']:<10.0f}{stats['pf']:<6.2f}")

        print("="*100)

        # 年別
        print("\n【年別パフォーマンス (Net Pips)】")
        for name, stats in self.results.items():
            print(f"\n  {name}:")
            for year, pips in stats['yearly'].items():
                print(f"    {year}: {pips:+,.0f}")

        # ナンピン統計
        print("\n【ナンピン回数別パフォーマンス】")
        for name, stats in self.results.items():
            print(f"\n  {name}:")
            for idx, row in stats['nanpin_stats'].iterrows():
                print(f"    N{idx}: {int(row['count'])}回, 勝率{row['win_rate']:.1f}%, 損益{row['total_pips']:+,.0f}pips")

    def visualize(self, save_path):
        """グラフ作成"""
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('USDJPY Real Data Backtest (Daily, 10 Years)\nData Source: Yahoo Finance',
                    fontsize=16, fontweight='bold')

        colors = ['#e74c3c', '#3498db', '#2ecc71']

        # 1. 価格チャート
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(self.df.index, self.df['close'], 'gray', linewidth=0.8, alpha=0.7)
        ax1.plot(self.df.index, self.df['ma_50'], 'blue', linewidth=0.5, alpha=0.5, label='MA50')
        ax1.plot(self.df.index, self.df['ma_200'], 'red', linewidth=0.5, alpha=0.5, label='MA200')
        ax1.set_title('USDJPY Price (Real Data)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 2. 累積損益
        ax2 = fig.add_subplot(2, 2, 2)
        for i, (name, stats) in enumerate(self.results.items()):
            df_trades = stats['df_trades']
            cumulative = df_trades['weighted_pips'].cumsum()
            ax2.plot(df_trades['entry_time'], cumulative, color=colors[i],
                    linewidth=2, label=name, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_title('Cumulative Net Pips', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pips')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 3. 年別比較
        ax3 = fig.add_subplot(2, 2, 3)
        all_years = set()
        for stats in self.results.values():
            all_years.update(stats['yearly'].index)
        years = sorted(all_years)

        x = np.arange(len(years))
        width = 0.25
        for i, (name, stats) in enumerate(self.results.items()):
            values = [stats['yearly'].get(y, 0) for y in years]
            ax3.bar(x + i*width, values, width, label=name, color=colors[i], alpha=0.8)

        ax3.set_xticks(x + width)
        ax3.set_xticklabels([str(y) for y in years], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.set_title('Yearly Performance', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Pips')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. サマリー
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        table_data = []
        for name, stats in self.results.items():
            row = [
                name,
                f"{stats['trades']}",
                f"{stats['win_rate']}%",
                f"{stats['total_pips']:+,.0f}",
                f"{stats['max_dd']:.0f}",
                f"{stats['pf']:.2f}"
            ]
            table_data.append(row)

        table = ax4.table(
            cellText=table_data,
            colLabels=['Config', 'Trades', 'WR', 'Net Pips', 'MaxDD', 'PF'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2.0)

        for j in range(6):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax4.set_title('Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nグラフを保存しました: {save_path}")


def main():
    print("="*70)
    print("  実データバックテスト")
    print("  USDJPY 日足 10年 (Yahoo Finance)")
    print("="*70)

    bt = RealDataBacktester(spread_pips=0.3)
    bt.run_comparison()
    bt.print_results()
    bt.visualize('/Users/naoto/ドル円/backtest_real_data.png')

    # 利益計算
    print("\n" + "="*70)
    print("  10年間利益 (基準ロット 0.1 lot = 100円/pip)")
    print("="*70)
    for name, stats in bt.results.items():
        profit = stats['total_pips'] * 100
        print(f"  {name}: {stats['total_pips']:+,.0f} pips = {profit:+,.0f}円")

    print("\n完了!")


if __name__ == "__main__":
    main()
