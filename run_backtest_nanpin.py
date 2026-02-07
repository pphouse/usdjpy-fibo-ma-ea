"""
フィボナッチ + MA + ナンピン EA バックテスト
ナンピン比率: 1:3:3:5
Fibo/MA許容誤差: ±10pips
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


class FiboMANanpinBacktester:
    """フィボナッチ + MA + ナンピン バックテスター"""

    def __init__(self, symbol="USDJPY"):
        self.symbol = symbol
        self.trades = []
        self.trade_groups = []  # ナンピングループ単位

        # パラメータ設定
        self.params = {
            'fibo_lookback': 50,
            'ma_periods': [20, 50, 100],
            'overlap_tolerance_pips': 10,      # Fibo/MA許容誤差 (±10pips)
            'take_profit_pips': 50,            # TP (平均建値から)
            'stop_loss_pips': 50,              # SL (平均建値から)
            'nanpin_interval_pips': 30,        # ナンピン間隔
            'lot_ratios': [1, 3, 3, 5],        # ロット比率 1:3:3:5
            'max_nanpin_count': 3,             # 最大ナンピン回数
            'use_trend_filter': True,
        }

    def set_params(self, tp_pips=None, sl_pips=None, nanpin_interval=None, tolerance_pips=None):
        """パラメータ設定"""
        if tp_pips is not None:
            self.params['take_profit_pips'] = tp_pips
        if sl_pips is not None:
            self.params['stop_loss_pips'] = sl_pips
        if nanpin_interval is not None:
            self.params['nanpin_interval_pips'] = nanpin_interval
        if tolerance_pips is not None:
            self.params['overlap_tolerance_pips'] = tolerance_pips

        print(f"設定: TP={self.params['take_profit_pips']}pips, "
              f"SL={self.params['stop_loss_pips']}pips, "
              f"ナンピン間隔={self.params['nanpin_interval_pips']}pips, "
              f"許容誤差=±{self.params['overlap_tolerance_pips']}pips")

    def generate_realistic_data(self, days=365):
        """リアルなUSDJPYデータを生成"""
        print("過去1年のデータを生成中...")
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

        print(f"データ生成完了: {len(df)}本 ({df.index[0].date()} ~ {df.index[-1].date()})")
        return df

    def calculate_ma(self, df, period):
        return df['close'].rolling(window=period).mean()

    def calculate_fibonacci(self, df, idx, lookback):
        if idx < lookback:
            return None, None, 0

        window = df.iloc[idx-lookback:idx]
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

    def check_entry(self, df, idx, fibo_382, fibo_618, trend):
        """エントリー条件チェック（±10pips許容）"""
        close = df['close'].iloc[idx]
        pip_value = 0.01

        # 許容誤差（pips → 価格）
        tolerance = self.params['overlap_tolerance_pips'] * pip_value

        # フィボナッチ条件
        fibo_hit = None
        if abs(close - fibo_382) < tolerance:
            fibo_hit = "38.2%"
        elif abs(close - fibo_618) < tolerance:
            fibo_hit = "61.8%"

        if fibo_hit is None:
            return None

        # MA条件
        ma_hit = None
        for period in self.params['ma_periods']:
            ma_col = f'ma_{period}'
            if ma_col in df.columns:
                ma_val = df[ma_col].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < tolerance:
                    ma_hit = period
                    break

        if ma_hit is None:
            return None

        # トレンドフィルター
        if self.params['use_trend_filter'] and 'ma_200' in df.columns:
            trend_ma = df['ma_200'].iloc[idx]
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    return None
                if trend == -1 and close > trend_ma:
                    return None

        return {
            'direction': 'BUY' if trend == 1 else 'SELL',
            'fibo_level': fibo_hit,
            'ma_period': ma_hit,
            'entry_price': close
        }

    def simulate_nanpin_trade(self, df, entry_idx, entry_info):
        """ナンピントレードをシミュレート"""
        direction = entry_info['direction']
        pip_value = 0.01
        lot_ratios = self.params['lot_ratios']

        # エントリーリスト
        entries = [{
            'idx': entry_idx,
            'price': entry_info['entry_price'],
            'lot_ratio': lot_ratios[0],
            'time': df.index[entry_idx]
        }]

        nanpin_count = 0
        last_entry_price = entry_info['entry_price']

        for i in range(entry_idx + 1, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            current_close = df['close'].iloc[i]

            # 平均建値計算
            total_lot = sum(e['lot_ratio'] for e in entries)
            avg_price = sum(e['price'] * e['lot_ratio'] for e in entries) / total_lot

            # TP/SL価格
            tp_dist = self.params['take_profit_pips'] * pip_value
            sl_dist = self.params['stop_loss_pips'] * pip_value

            if direction == 'BUY':
                tp_price = avg_price + tp_dist
                sl_price = avg_price - sl_dist
            else:
                tp_price = avg_price - tp_dist
                sl_price = avg_price + sl_dist

            # TP/SLチェック
            if direction == 'BUY':
                if current_high >= tp_price:
                    # 利確
                    return self._create_trade_result(entries, df, i, tp_price, 'WIN', entry_info)
                elif current_low <= sl_price:
                    # 損切
                    return self._create_trade_result(entries, df, i, sl_price, 'LOSS', entry_info)
            else:
                if current_low <= tp_price:
                    return self._create_trade_result(entries, df, i, tp_price, 'WIN', entry_info)
                elif current_high >= sl_price:
                    return self._create_trade_result(entries, df, i, sl_price, 'LOSS', entry_info)

            # ナンピンチェック
            if nanpin_count < self.params['max_nanpin_count']:
                nanpin_dist = self.params['nanpin_interval_pips'] * pip_value

                if direction == 'BUY':
                    distance = last_entry_price - current_close
                else:
                    distance = current_close - last_entry_price

                if distance >= nanpin_dist:
                    nanpin_count += 1
                    lot_ratio = lot_ratios[min(nanpin_count, len(lot_ratios)-1)]
                    entries.append({
                        'idx': i,
                        'price': current_close,
                        'lot_ratio': lot_ratio,
                        'time': df.index[i]
                    })
                    last_entry_price = current_close

        return None

    def _create_trade_result(self, entries, df, exit_idx, exit_price, result, entry_info):
        """トレード結果を作成"""
        total_lot = sum(e['lot_ratio'] for e in entries)
        avg_price = sum(e['price'] * e['lot_ratio'] for e in entries) / total_lot
        pip_value = 0.01

        if entry_info['direction'] == 'BUY':
            pips = (exit_price - avg_price) / pip_value
        else:
            pips = (avg_price - exit_price) / pip_value

        # ロット加重の損益
        weighted_pips = pips * (total_lot / self.params['lot_ratios'][0])

        return {
            'entry_time': entries[0]['time'],
            'exit_time': df.index[exit_idx],
            'direction': entry_info['direction'],
            'avg_entry_price': avg_price,
            'exit_price': exit_price,
            'pips': round(pips, 1),
            'weighted_pips': round(weighted_pips, 1),  # ロット加重損益
            'result': result,
            'nanpin_count': len(entries) - 1,
            'total_lots': total_lot,
            'fibo_level': entry_info['fibo_level'],
            'ma_period': entry_info['ma_period'],
            'entries': entries
        }

    def run_backtest(self, df=None):
        """バックテスト実行"""
        if df is None:
            df = self.generate_realistic_data()

        print("\nバックテスト実行中（ナンピン版）...")

        # MA計算
        for period in self.params['ma_periods']:
            df[f'ma_{period}'] = self.calculate_ma(df, period)
        df['ma_200'] = self.calculate_ma(df, 200)

        self.trades = []
        self.df = df
        lookback = self.params['fibo_lookback']

        in_trade = False
        last_exit_idx = 0

        for idx in range(lookback + 200, len(df) - 100):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            fibo_382, fibo_618, trend = self.calculate_fibonacci(df, idx, lookback)
            if trend == 0:
                continue

            entry = self.check_entry(df, idx, fibo_382, fibo_618, trend)
            if entry is None:
                continue

            result = self.simulate_nanpin_trade(df, idx, entry)
            if result:
                self.trades.append(result)
                in_trade = True
                last_exit_idx = df.index.get_loc(result['exit_time'])

        print(f"バックテスト完了: {len(self.trades)}トレードグループ")
        return self.trades

    def get_statistics(self):
        """統計を計算"""
        if not self.trades:
            return None

        df = pd.DataFrame(self.trades)

        total = len(df)
        wins = len(df[df['result'] == 'WIN'])
        losses = len(df[df['result'] == 'LOSS'])
        win_rate = wins / total * 100 if total > 0 else 0

        # 通常損益
        total_pips = df['pips'].sum()
        avg_pips = df['pips'].mean()

        # ロット加重損益
        total_weighted_pips = df['weighted_pips'].sum()
        avg_weighted_pips = df['weighted_pips'].mean()

        # ナンピン統計
        total_nanpins = df['nanpin_count'].sum()
        avg_nanpins = df['nanpin_count'].mean()
        max_nanpins = df['nanpin_count'].max()

        # ナンピン回数別勝率
        nanpin_stats = df.groupby('nanpin_count').agg({
            'result': lambda x: (x == 'WIN').sum() / len(x) * 100,
            'weighted_pips': 'sum'
        }).round(2)

        # ドローダウン
        cumulative = df['weighted_pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        # PF
        win_pips = df[df['weighted_pips'] > 0]['weighted_pips'].sum()
        loss_pips = abs(df[df['weighted_pips'] < 0]['weighted_pips'].sum())
        pf = win_pips / loss_pips if loss_pips > 0 else float('inf')

        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'total_pips': round(total_pips, 1),
            'avg_pips': round(avg_pips, 2),
            'total_weighted_pips': round(total_weighted_pips, 1),
            'avg_weighted_pips': round(avg_weighted_pips, 2),
            'max_drawdown': round(max_dd, 1),
            'profit_factor': round(pf, 2),
            'total_nanpins': int(total_nanpins),
            'avg_nanpins': round(avg_nanpins, 2),
            'max_nanpins': int(max_nanpins),
            'nanpin_stats': nanpin_stats
        }

    def print_results(self):
        """結果を表示"""
        stats = self.get_statistics()
        if stats is None:
            print("トレードがありません")
            return

        print("\n" + "="*65)
        print("        バックテスト結果サマリー（ナンピン版）")
        print("="*65)
        print(f"  通貨ペア: {self.symbol}")
        print(f"  時間足: 4時間足 (H4)")
        print(f"  TP: {self.params['take_profit_pips']} pips (平均建値から)")
        print(f"  SL: {self.params['stop_loss_pips']} pips (平均建値から)")
        print(f"  ナンピン間隔: {self.params['nanpin_interval_pips']} pips")
        print(f"  ロット比率: 1:3:3:5")
        print(f"  Fibo/MA許容誤差: ±{self.params['overlap_tolerance_pips']} pips")
        print("-"*65)
        print(f"  総トレード数: {stats['total_trades']}")
        print(f"  勝ちトレード: {stats['wins']}")
        print(f"  負けトレード: {stats['losses']}")
        print(f"  勝率: {stats['win_rate']}%")
        print("-"*65)
        print(f"  【単純損益】")
        print(f"    合計: {stats['total_pips']} pips")
        print(f"    平均: {stats['avg_pips']} pips/トレード")
        print(f"  【ロット加重損益】")
        print(f"    合計: {stats['total_weighted_pips']} pips相当")
        print(f"    平均: {stats['avg_weighted_pips']} pips相当/トレード")
        print("-"*65)
        print(f"  最大ドローダウン: {stats['max_drawdown']} pips相当")
        print(f"  プロフィットファクター: {stats['profit_factor']}")
        print("-"*65)
        print(f"  【ナンピン統計】")
        print(f"    総ナンピン回数: {stats['total_nanpins']}")
        print(f"    平均ナンピン回数: {stats['avg_nanpins']}")
        print(f"    最大ナンピン回数: {stats['max_nanpins']}")
        print("-"*65)
        print("  【ナンピン回数別パフォーマンス】")
        if 'nanpin_stats' in stats and stats['nanpin_stats'] is not None:
            for idx, row in stats['nanpin_stats'].iterrows():
                print(f"    ナンピン{idx}回: 勝率{row['result']:.1f}%, 損益{row['weighted_pips']:.1f}pips相当")
        print("="*65)

    def visualize(self, save_path=None):
        """結果をビジュアル化"""
        if not self.trades:
            print("トレードがありません")
            return

        df_trades = pd.DataFrame(self.trades)

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle(f'Fibonacci + MA + Nanpin EA Backtest Results\n'
                     f'TP={self.params["take_profit_pips"]}pips, '
                     f'SL={self.params["stop_loss_pips"]}pips, '
                     f'Nanpin Interval={self.params["nanpin_interval_pips"]}pips, '
                     f'Tolerance=±{self.params["overlap_tolerance_pips"]}pips',
                     fontsize=14, fontweight='bold')

        # 1. 累積損益（ロット加重）
        ax1 = fig.add_subplot(3, 2, 1)
        cumulative = df_trades['weighted_pips'].cumsum()
        ax1.plot(df_trades['entry_time'], cumulative, 'b-', linewidth=2)
        ax1.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative >= 0), color='green', alpha=0.3)
        ax1.fill_between(df_trades['entry_time'], cumulative, 0,
                        where=(cumulative < 0), color='red', alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.set_title('Cumulative Weighted Pips (Lot-adjusted)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Pips Equivalent')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. 勝率
        ax2 = fig.add_subplot(3, 2, 2)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        colors = ['#2ecc71', '#e74c3c']
        ax2.pie([wins, losses], labels=['Win', 'Loss'], colors=colors,
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
        ax2.set_title(f'Win Rate ({wins}W / {losses}L)', fontsize=12, fontweight='bold')

        # 3. ナンピン回数分布
        ax3 = fig.add_subplot(3, 2, 3)
        nanpin_counts = df_trades['nanpin_count'].value_counts().sort_index()
        colors_nanpin = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c'][:len(nanpin_counts)]
        bars = ax3.bar(nanpin_counts.index, nanpin_counts.values, color=colors_nanpin)
        ax3.set_xlabel('Nanpin Count')
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Trade Distribution by Nanpin Count', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, nanpin_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. ナンピン回数別勝率
        ax4 = fig.add_subplot(3, 2, 4)
        nanpin_winrate = df_trades.groupby('nanpin_count').apply(
            lambda x: (x['result'] == 'WIN').sum() / len(x) * 100
        )
        bars = ax4.bar(nanpin_winrate.index, nanpin_winrate.values, color='steelblue')
        ax4.axhline(y=50, color='red', linestyle='--', label='50%')
        ax4.set_xlabel('Nanpin Count')
        ax4.set_ylabel('Win Rate (%)')
        ax4.set_title('Win Rate by Nanpin Count', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, nanpin_winrate.values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()

        # 5. 月別パフォーマンス
        ax5 = fig.add_subplot(3, 2, 5)
        df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')
        monthly = df_trades.groupby('month')['weighted_pips'].sum()
        colors = ['green' if x > 0 else 'red' for x in monthly.values]
        ax5.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(monthly)))
        ax5.set_xticklabels([str(m) for m in monthly.index], rotation=45)
        ax5.axhline(y=0, color='black', linewidth=0.5)
        ax5.set_title('Monthly Performance (Weighted Pips)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Pips Equivalent')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. ドローダウン
        ax6 = fig.add_subplot(3, 2, 6)
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        ax6.fill_between(df_trades['entry_time'], drawdown, 0, color='red', alpha=0.4)
        ax6.plot(df_trades['entry_time'], drawdown, 'r-', linewidth=1)
        ax6.set_title('Drawdown (Weighted Pips)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Pips Equivalent')
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nグラフを保存しました: {save_path}")

        plt.close()

    def export_trades(self, filepath):
        """トレード履歴をエクスポート"""
        if not self.trades:
            return

        # entriesを除外してエクスポート
        export_data = []
        for t in self.trades:
            row = {k: v for k, v in t.items() if k != 'entries'}
            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"トレード履歴を保存しました: {filepath}")


def main():
    """メイン関数"""
    print("="*65)
    print("    Fibonacci + MA + Nanpin EA Backtest")
    print("    USDJPY 4H - Past 1 Year")
    print("    Lot Ratio: 1:3:3:5")
    print("="*65)

    #===========================================
    # パラメータ設定（ここで変更可能）
    #===========================================
    TP_PIPS = 50              # 利確 pips（平均建値から）
    SL_PIPS = 50              # 損切 pips（平均建値から）
    NANPIN_INTERVAL = 30      # ナンピン間隔 pips
    TOLERANCE_PIPS = 30       # Fibo/MA許容誤差 ±pips
    #===========================================

    bt = FiboMANanpinBacktester("USDJPY")
    bt.set_params(
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        nanpin_interval=NANPIN_INTERVAL,
        tolerance_pips=TOLERANCE_PIPS
    )

    # バックテスト実行
    bt.run_backtest()

    # 結果表示
    bt.print_results()

    # ビジュアル化
    bt.visualize(save_path='/Users/naoto/ドル円/backtest_nanpin_results.png')

    # トレード履歴エクスポート
    bt.export_trades('/Users/naoto/ドル円/trade_history_nanpin.csv')

    print("\n完了!")


if __name__ == "__main__":
    main()
