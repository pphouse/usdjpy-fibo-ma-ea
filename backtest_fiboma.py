"""
フィボナッチ + 移動平均線 EA バックテストスクリプト
過去1年のデータでバックテストを実行し、結果をビジュアル化します
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class FiboMABacktester:
    """フィボナッチ + MA バックテスター"""

    def __init__(self, symbol="USDJPY", timeframe=mt5.TIMEFRAME_H4):
        self.symbol = symbol
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = []

        # パラメータ（手動設定可能）
        self.params = {
            'fibo_lookback': 50,        # フィボナッチ計算用バー数
            'fibo_level1': 0.382,       # フィボナッチ38.2%
            'fibo_level2': 0.618,       # フィボナッチ61.8%
            'fibo_tolerance': 0.002,    # フィボナッチ許容誤差
            'ma_periods': [20, 50, 100], # MA期間
            'ma_tolerance': 0.001,      # MA許容誤差
            'take_profit_pips': 50,     # 利確pips（手動設定可）
            'stop_loss_pips': 50,       # 損切pips（手動設定可）
            'lot_size': 0.1,            # ロットサイズ
            'use_trend_filter': True,   # トレンドフィルター使用
            'trend_ma_period': 200      # トレンド判定MA
        }

    def set_tp_sl(self, tp_pips, sl_pips):
        """TP/SLをpipsで設定"""
        self.params['take_profit_pips'] = tp_pips
        self.params['stop_loss_pips'] = sl_pips
        print(f"TP: {tp_pips} pips, SL: {sl_pips} pips に設定しました")

    def connect_mt5(self):
        """MT5に接続"""
        if not mt5.initialize():
            print("MT5初期化エラー")
            return False
        print(f"MT5接続成功: {mt5.terminal_info()}")
        return True

    def get_historical_data(self, days=365):
        """過去データを取得"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        rates = mt5.copy_rates_range(
            self.symbol,
            self.timeframe,
            start_date,
            end_date
        )

        if rates is None or len(rates) == 0:
            print("データ取得エラー")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        print(f"取得データ: {len(df)}本 ({start_date.date()} ~ {end_date.date()})")
        return df

    def calculate_ma(self, df, period):
        """移動平均線を計算"""
        return df['close'].rolling(window=period).mean()

    def calculate_fibonacci(self, df, idx, lookback):
        """フィボナッチリトレースメントを計算"""
        if idx < lookback:
            return None, None, 0

        window = df.iloc[idx-lookback:idx]
        high = window['high'].max()
        low = window['low'].min()
        high_idx = window['high'].idxmax()
        low_idx = window['low'].idxmin()

        range_val = high - low
        if range_val < 0.01:  # レンジが狭すぎる
            return None, None, 0

        # トレンド方向判定
        if high_idx > low_idx:  # 上昇トレンド
            trend = 1
            fibo_382 = high - range_val * 0.382
            fibo_618 = high - range_val * 0.618
        else:  # 下降トレンド
            trend = -1
            fibo_382 = low + range_val * 0.382
            fibo_618 = low + range_val * 0.618

        return fibo_382, fibo_618, trend

    def check_entry_conditions(self, df, idx, fibo_382, fibo_618, trend):
        """エントリー条件をチェック"""
        close = df['close'].iloc[idx]

        # フィボナッチ条件
        fibo_tol = close * self.params['fibo_tolerance']
        fibo_hit = False
        fibo_level = ""

        if abs(close - fibo_382) < fibo_tol:
            fibo_hit = True
            fibo_level = "38.2%"
        elif abs(close - fibo_618) < fibo_tol:
            fibo_hit = True
            fibo_level = "61.8%"

        if not fibo_hit:
            return None

        # MA条件
        ma_tol = close * self.params['ma_tolerance']
        ma_hit = False
        ma_period = 0

        for period in self.params['ma_periods']:
            ma_col = f'ma_{period}'
            if ma_col in df.columns:
                ma_val = df[ma_col].iloc[idx]
                if pd.notna(ma_val) and abs(close - ma_val) < ma_tol:
                    ma_hit = True
                    ma_period = period
                    break

        if not ma_hit:
            return None

        # トレンドフィルター
        if self.params['use_trend_filter']:
            trend_ma = df['ma_200'].iloc[idx] if 'ma_200' in df.columns else None
            if pd.notna(trend_ma):
                if trend == 1 and close < trend_ma:
                    return None
                if trend == -1 and close > trend_ma:
                    return None

        return {
            'direction': 'BUY' if trend == 1 else 'SELL',
            'fibo_level': fibo_level,
            'ma_period': ma_period,
            'entry_price': close
        }

    def simulate_trade(self, df, entry_idx, entry_info):
        """トレードをシミュレート"""
        entry_price = entry_info['entry_price']
        direction = entry_info['direction']

        # pip値（USDJPYの場合）
        pip_value = 0.01 if 'JPY' in self.symbol else 0.0001

        tp_distance = self.params['take_profit_pips'] * pip_value
        sl_distance = self.params['stop_loss_pips'] * pip_value

        if direction == 'BUY':
            tp_price = entry_price + tp_distance
            sl_price = entry_price - sl_distance
        else:
            tp_price = entry_price - tp_distance
            sl_price = entry_price + sl_distance

        # トレード結果をシミュレート
        for i in range(entry_idx + 1, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            if direction == 'BUY':
                if high >= tp_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pips': self.params['take_profit_pips'],
                        'result': 'WIN',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
                elif low <= sl_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pips': -self.params['stop_loss_pips'],
                        'result': 'LOSS',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
            else:  # SELL
                if low <= tp_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pips': self.params['take_profit_pips'],
                        'result': 'WIN',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }
                elif high >= sl_price:
                    return {
                        'entry_time': df.index[entry_idx],
                        'exit_time': df.index[i],
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pips': -self.params['stop_loss_pips'],
                        'result': 'LOSS',
                        'fibo_level': entry_info['fibo_level'],
                        'ma_period': entry_info['ma_period']
                    }

        return None  # データ終了

    def run_backtest(self, df=None, days=365):
        """バックテストを実行"""
        if df is None:
            df = self.get_historical_data(days)
            if df is None:
                return None

        # MAを計算
        for period in self.params['ma_periods']:
            df[f'ma_{period}'] = self.calculate_ma(df, period)
        df['ma_200'] = self.calculate_ma(df, 200)

        self.trades = []
        in_trade = False
        last_exit_idx = 0

        lookback = self.params['fibo_lookback']

        print("バックテスト実行中...")

        for idx in range(lookback, len(df) - 1):
            if in_trade and idx < last_exit_idx:
                continue
            in_trade = False

            # フィボナッチ計算
            fibo_382, fibo_618, trend = self.calculate_fibonacci(df, idx, lookback)
            if trend == 0:
                continue

            # エントリー条件チェック
            entry_info = self.check_entry_conditions(df, idx, fibo_382, fibo_618, trend)
            if entry_info is None:
                continue

            # トレードシミュレート
            trade_result = self.simulate_trade(df, idx, entry_info)
            if trade_result:
                self.trades.append(trade_result)
                in_trade = True
                # 次のエントリーまでスキップするインデックスを計算
                if trade_result['exit_time'] in df.index:
                    last_exit_idx = df.index.get_loc(trade_result['exit_time'])

        print(f"バックテスト完了: {len(self.trades)}トレード")
        return self.trades

    def calculate_statistics(self):
        """統計を計算"""
        if not self.trades:
            return None

        df_trades = pd.DataFrame(self.trades)

        total_trades = len(df_trades)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        total_pips = df_trades['pips'].sum()
        avg_pips = df_trades['pips'].mean()
        max_win = df_trades['pips'].max()
        max_loss = df_trades['pips'].min()

        # ドローダウン計算
        cumulative = df_trades['pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # 連勝・連敗
        results = df_trades['result'].values
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for r in results:
            if r == 'WIN':
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        stats = {
            '総トレード数': total_trades,
            '勝ちトレード': wins,
            '負けトレード': losses,
            '勝率 (%)': round(win_rate, 2),
            '合計Pips': round(total_pips, 1),
            '平均Pips': round(avg_pips, 2),
            '最大勝ち (pips)': round(max_win, 1),
            '最大負け (pips)': round(max_loss, 1),
            '最大ドローダウン (pips)': round(max_drawdown, 1),
            '最大連勝': max_consecutive_wins,
            '最大連敗': max_consecutive_losses,
            'プロフィットファクター': round(abs(df_trades[df_trades['pips'] > 0]['pips'].sum() /
                                              df_trades[df_trades['pips'] < 0]['pips'].sum()), 2) if losses > 0 else float('inf')
        }

        return stats

    def visualize_results(self, df=None, save_path=None):
        """結果をビジュアル化"""
        if not self.trades:
            print("トレードがありません")
            return

        df_trades = pd.DataFrame(self.trades)

        # フィギュア作成
        fig = plt.figure(figsize=(16, 14))

        # 1. 累積Pipsチャート
        ax1 = fig.add_subplot(3, 2, 1)
        cumulative_pips = df_trades['pips'].cumsum()
        ax1.plot(df_trades['entry_time'], cumulative_pips, 'b-', linewidth=2, label='累積Pips')
        ax1.fill_between(df_trades['entry_time'], cumulative_pips, 0,
                        where=(cumulative_pips >= 0), color='green', alpha=0.3)
        ax1.fill_between(df_trades['entry_time'], cumulative_pips, 0,
                        where=(cumulative_pips < 0), color='red', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_title('累積損益 (Pips)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('日付')
        ax1.set_ylabel('Pips')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. 勝率円グラフ
        ax2 = fig.add_subplot(3, 2, 2)
        wins = len(df_trades[df_trades['result'] == 'WIN'])
        losses = len(df_trades[df_trades['result'] == 'LOSS'])
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)
        ax2.pie([wins, losses], explode=explode, labels=['勝ち', '負け'],
                colors=colors, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12})
        ax2.set_title(f'勝率 ({wins}勝 / {losses}敗)', fontsize=14, fontweight='bold')

        # 3. 月別パフォーマンス
        ax3 = fig.add_subplot(3, 2, 3)
        df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M')
        monthly_pips = df_trades.groupby('month')['pips'].sum()
        colors = ['green' if x > 0 else 'red' for x in monthly_pips.values]
        bars = ax3.bar(range(len(monthly_pips)), monthly_pips.values, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(monthly_pips)))
        ax3.set_xticklabels([str(m) for m in monthly_pips.index], rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('月別損益 (Pips)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Pips')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. フィボナッチレベル別パフォーマンス
        ax4 = fig.add_subplot(3, 2, 4)
        fibo_stats = df_trades.groupby('fibo_level').agg({
            'pips': ['sum', 'count', 'mean']
        }).round(2)
        fibo_stats.columns = ['合計Pips', 'トレード数', '平均Pips']

        x = range(len(fibo_stats))
        width = 0.35
        bars1 = ax4.bar([i - width/2 for i in x], fibo_stats['合計Pips'],
                       width, label='合計Pips', color='steelblue')
        ax4.set_xticks(x)
        ax4.set_xticklabels(fibo_stats.index)
        ax4.set_title('フィボナッチレベル別パフォーマンス', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Pips')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # トレード数を表示
        for bar, count in zip(bars1, fibo_stats['トレード数']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={int(count)}', ha='center', fontsize=10)

        # 5. MA期間別パフォーマンス
        ax5 = fig.add_subplot(3, 2, 5)
        ma_stats = df_trades.groupby('ma_period').agg({
            'pips': ['sum', 'count', 'mean']
        }).round(2)
        ma_stats.columns = ['合計Pips', 'トレード数', '平均Pips']

        x = range(len(ma_stats))
        colors = ['#3498db', '#9b59b6', '#e67e22'][:len(ma_stats)]
        bars = ax5.bar(x, ma_stats['合計Pips'], color=colors, alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'MA{int(p)}' for p in ma_stats.index])
        ax5.set_title('移動平均線期間別パフォーマンス', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Pips')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, ma_stats['トレード数']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={int(count)}', ha='center', fontsize=10)

        # 6. ドローダウンチャート
        ax6 = fig.add_subplot(3, 2, 6)
        cumulative = df_trades['pips'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max

        ax6.fill_between(df_trades['entry_time'], drawdown, 0,
                        color='red', alpha=0.4, label='ドローダウン')
        ax6.plot(df_trades['entry_time'], drawdown, 'r-', linewidth=1)
        ax6.set_title('ドローダウン (Pips)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('日付')
        ax6.set_ylabel('Pips')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"グラフを保存しました: {save_path}")

        plt.show()

    def print_statistics(self):
        """統計を表示"""
        stats = self.calculate_statistics()
        if stats is None:
            print("統計を計算できません")
            return

        print("\n" + "="*50)
        print("       バックテスト結果サマリー")
        print("="*50)
        print(f"  設定: TP={self.params['take_profit_pips']}pips, SL={self.params['stop_loss_pips']}pips")
        print("-"*50)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*50)

    def export_trades(self, filepath):
        """トレード履歴をCSVでエクスポート"""
        if not self.trades:
            print("エクスポートするトレードがありません")
            return

        df = pd.DataFrame(self.trades)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"トレード履歴を保存しました: {filepath}")


def run_backtest_without_mt5(symbol="USDJPY", tp_pips=50, sl_pips=50):
    """
    MT5なしでサンプルデータでバックテストを実行
    （MT5がインストールされていない環境用）
    """
    print("サンプルデータでバックテストを実行します...")

    # サンプルデータ生成（実際のUSDJPYに近い動き）
    np.random.seed(42)
    dates = pd.date_range(start='2025-02-01', end='2026-02-01', freq='4h')

    # ランダムウォークで価格を生成
    n = len(dates)
    returns = np.random.normal(0, 0.001, n)
    price = 150.0  # 開始価格
    prices = [price]

    for r in returns[1:]:
        price = price * (1 + r)
        prices.append(price)

    prices = np.array(prices)

    # OHLCデータ作成
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n))),
        'close': prices * (1 + np.random.normal(0, 0.001, n)),
        'tick_volume': np.random.randint(100, 1000, n),
        'spread': np.random.randint(1, 5, n),
        'real_volume': np.random.randint(1000, 10000, n)
    }, index=dates)

    # バックテスター作成
    bt = FiboMABacktester(symbol=symbol)
    bt.set_tp_sl(tp_pips, sl_pips)

    # バックテスト実行
    bt.run_backtest(df=df)

    # 結果表示
    bt.print_statistics()
    bt.visualize_results(save_path='/Users/naoto/ドル円/backtest_results.png')
    bt.export_trades('/Users/naoto/ドル円/trade_history.csv')

    return bt


def main():
    """メイン関数"""
    # パラメータ設定（手動で変更可能）
    SYMBOL = "USDJPY"
    TIMEFRAME = mt5.TIMEFRAME_H4  # 4時間足
    TP_PIPS = 50  # 利確pips（ここを変更）
    SL_PIPS = 50  # 損切pips（ここを変更）
    BACKTEST_DAYS = 365  # 過去1年

    print("="*60)
    print("  フィボナッチ + 移動平均線 EA バックテスト")
    print("="*60)
    print(f"  通貨ペア: {SYMBOL}")
    print(f"  時間足: H4")
    print(f"  TP: {TP_PIPS} pips")
    print(f"  SL: {SL_PIPS} pips")
    print(f"  期間: 過去{BACKTEST_DAYS}日")
    print("="*60)

    # バックテスター作成
    bt = FiboMABacktester(symbol=SYMBOL, timeframe=TIMEFRAME)
    bt.set_tp_sl(TP_PIPS, SL_PIPS)

    # MT5接続
    if bt.connect_mt5():
        # バックテスト実行
        bt.run_backtest(days=BACKTEST_DAYS)

        # 結果表示
        bt.print_statistics()

        # ビジュアル化
        bt.visualize_results(save_path='/Users/naoto/ドル円/backtest_results.png')

        # トレード履歴エクスポート
        bt.export_trades('/Users/naoto/ドル円/trade_history.csv')

        # MT5終了
        mt5.shutdown()
    else:
        # MT5が利用できない場合はサンプルデータで実行
        print("\nMT5に接続できません。サンプルデータで実行します...")
        run_backtest_without_mt5(SYMBOL, TP_PIPS, SL_PIPS)


if __name__ == "__main__":
    main()
