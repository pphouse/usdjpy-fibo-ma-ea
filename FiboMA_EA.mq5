//+------------------------------------------------------------------+
//|                                                   FiboMA_EA.mq5 |
//|                        フィボナッチ + 移動平均線 エントリーEA      |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "FiboMA EA"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- 入力パラメータ
input group "=== エントリー設定 ==="
input ENUM_TIMEFRAMES TimeFrame = PERIOD_H4;           // 時間足（H4 or D1）
input int    FiboLookbackBars = 50;                    // フィボナッチ計算用の過去バー数
input double FiboLevel1 = 38.2;                        // フィボナッチレベル1 (%)
input double FiboLevel2 = 61.8;                        // フィボナッチレベル2 (%)
input double FiboTolerance = 0.002;                    // フィボナッチレベル許容誤差（価格比率）

input group "=== 移動平均線設定 ==="
input int    MA_Period1 = 20;                          // MA期間1
input int    MA_Period2 = 50;                          // MA期間2
input int    MA_Period3 = 100;                         // MA期間3
input ENUM_MA_METHOD MA_Method = MODE_SMA;             // MA計算方法
input double MA_Tolerance = 0.001;                     // MA許容誤差（価格比率）

input group "=== 決済設定 ==="
input double TakeProfit_Pips = 50.0;                   // 利確 (Pips)
input double StopLoss_Pips = 50.0;                     // 損切り (Pips)

input group "=== ロット設定 ==="
input double LotSize = 0.1;                            // ロットサイズ
input int    MagicNumber = 123456;                     // マジックナンバー

input group "=== フィルター設定 ==="
input bool   UseTrendFilter = true;                    // トレンドフィルター使用
input int    TrendMA_Period = 200;                     // トレンド判定用MA期間

//--- グローバル変数
CTrade trade;
int ma1_handle, ma2_handle, ma3_handle, trend_ma_handle;
double pip_value;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- PIP値の計算
    pip_value = _Point;
    if(_Digits == 3 || _Digits == 5)
        pip_value = _Point * 10;

    //--- MAハンドル作成
    ma1_handle = iMA(_Symbol, TimeFrame, MA_Period1, 0, MA_Method, PRICE_CLOSE);
    ma2_handle = iMA(_Symbol, TimeFrame, MA_Period2, 0, MA_Method, PRICE_CLOSE);
    ma3_handle = iMA(_Symbol, TimeFrame, MA_Period3, 0, MA_Method, PRICE_CLOSE);

    if(UseTrendFilter)
        trend_ma_handle = iMA(_Symbol, TimeFrame, TrendMA_Period, 0, MA_Method, PRICE_CLOSE);

    if(ma1_handle == INVALID_HANDLE || ma2_handle == INVALID_HANDLE || ma3_handle == INVALID_HANDLE)
    {
        Print("MAハンドル作成エラー");
        return(INIT_FAILED);
    }

    //--- トレード設定
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    Print("FiboMA EA 初期化完了");
    Print("時間足: ", EnumToString(TimeFrame));
    Print("TP: ", TakeProfit_Pips, " pips, SL: ", StopLoss_Pips, " pips");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(ma1_handle);
    IndicatorRelease(ma2_handle);
    IndicatorRelease(ma3_handle);
    if(UseTrendFilter)
        IndicatorRelease(trend_ma_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- 新しいバーでのみ処理
    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, TimeFrame, 0);

    if(last_bar_time == current_bar_time)
        return;
    last_bar_time = current_bar_time;

    //--- 既存ポジションチェック
    if(HasOpenPosition())
        return;

    //--- フィボナッチレベル計算
    double fibo_382, fibo_618;
    int trend_direction;

    if(!CalculateFibonacci(fibo_382, fibo_618, trend_direction))
        return;

    //--- MA値取得
    double ma1[], ma2[], ma3[];
    ArraySetAsSeries(ma1, true);
    ArraySetAsSeries(ma2, true);
    ArraySetAsSeries(ma3, true);

    if(CopyBuffer(ma1_handle, 0, 0, 3, ma1) < 3) return;
    if(CopyBuffer(ma2_handle, 0, 0, 3, ma2) < 3) return;
    if(CopyBuffer(ma3_handle, 0, 0, 3, ma3) < 3) return;

    //--- 現在価格
    double close = iClose(_Symbol, TimeFrame, 1);
    double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    //--- トレンドフィルター
    if(UseTrendFilter)
    {
        double trend_ma[];
        ArraySetAsSeries(trend_ma, true);
        if(CopyBuffer(trend_ma_handle, 0, 0, 2, trend_ma) < 2) return;

        // 上昇トレンドでロング、下降トレンドでショートのみ
        if(trend_direction == 1 && current_price < trend_ma[0]) return;  // 上昇中だがMA200下
        if(trend_direction == -1 && current_price > trend_ma[0]) return; // 下降中だがMA200上
    }

    //--- エントリー条件チェック
    bool fibo_condition = false;
    bool ma_condition = false;
    string entry_reason = "";

    //--- フィボナッチ条件（価格がフィボレベル付近）
    double tolerance = close * FiboTolerance;

    if(MathAbs(close - fibo_382) < tolerance)
    {
        fibo_condition = true;
        entry_reason += "Fibo38.2% ";
    }
    else if(MathAbs(close - fibo_618) < tolerance)
    {
        fibo_condition = true;
        entry_reason += "Fibo61.8% ";
    }

    if(!fibo_condition) return;

    //--- MA条件（価格がいずれかのMA付近）
    double ma_tol = close * MA_Tolerance;

    if(MathAbs(close - ma1[1]) < ma_tol)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period1) + " ";
    }
    if(MathAbs(close - ma2[1]) < ma_tol)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period2) + " ";
    }
    if(MathAbs(close - ma3[1]) < ma_tol)
    {
        ma_condition = true;
        entry_reason += "MA" + IntegerToString(MA_Period3) + " ";
    }

    if(!ma_condition) return;

    //--- エントリー実行
    double tp, sl;
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    if(trend_direction == 1)  // 上昇トレンド → 押し目買い
    {
        tp = ask + TakeProfit_Pips * pip_value;
        sl = ask - StopLoss_Pips * pip_value;

        if(trade.Buy(LotSize, _Symbol, ask, sl, tp, "FiboMA Buy: " + entry_reason))
        {
            Print("BUY エントリー: ", entry_reason);
            DrawEntryArrow(true, ask);
        }
    }
    else if(trend_direction == -1)  // 下降トレンド → 戻り売り
    {
        tp = bid - TakeProfit_Pips * pip_value;
        sl = bid + StopLoss_Pips * pip_value;

        if(trade.Sell(LotSize, _Symbol, bid, sl, tp, "FiboMA Sell: " + entry_reason))
        {
            Print("SELL エントリー: ", entry_reason);
            DrawEntryArrow(false, bid);
        }
    }
}

//+------------------------------------------------------------------+
//| フィボナッチレベル計算                                            |
//+------------------------------------------------------------------+
bool CalculateFibonacci(double &fibo_382, double &fibo_618, int &trend)
{
    //--- 直近N本のバーから高値・安値を取得
    double high_array[], low_array[];
    ArraySetAsSeries(high_array, true);
    ArraySetAsSeries(low_array, true);

    if(CopyHigh(_Symbol, TimeFrame, 1, FiboLookbackBars, high_array) < FiboLookbackBars) return false;
    if(CopyLow(_Symbol, TimeFrame, 1, FiboLookbackBars, low_array) < FiboLookbackBars) return false;

    //--- 最高値・最安値とその位置を特定
    int highest_idx = ArrayMaximum(high_array);
    int lowest_idx = ArrayMinimum(low_array);

    double highest = high_array[highest_idx];
    double lowest = low_array[lowest_idx];
    double range = highest - lowest;

    if(range < _Point * 100) return false;  // レンジが狭すぎる

    //--- トレンド方向判定（高値が先か安値が先か）
    if(highest_idx < lowest_idx)
    {
        // 高値が最近 → 上昇トレンド → 押し目を探す
        trend = 1;
        fibo_382 = highest - range * 0.382;
        fibo_618 = highest - range * 0.618;
    }
    else
    {
        // 安値が最近 → 下降トレンド → 戻りを探す
        trend = -1;
        fibo_382 = lowest + range * 0.382;
        fibo_618 = lowest + range * 0.618;
    }

    //--- フィボナッチレベルを描画
    DrawFibonacciLevels(highest, lowest, fibo_382, fibo_618, trend);

    return true;
}

//+------------------------------------------------------------------+
//| フィボナッチレベル描画                                            |
//+------------------------------------------------------------------+
void DrawFibonacciLevels(double high, double low, double fibo_382, double fibo_618, int trend)
{
    datetime time_start = iTime(_Symbol, TimeFrame, FiboLookbackBars);
    datetime time_end = iTime(_Symbol, TimeFrame, 0);

    //--- 既存のオブジェクトを削除
    ObjectDelete(0, "Fibo_High");
    ObjectDelete(0, "Fibo_Low");
    ObjectDelete(0, "Fibo_382");
    ObjectDelete(0, "Fibo_618");

    //--- 水平線を描画
    color line_color = (trend == 1) ? clrDodgerBlue : clrOrangeRed;

    ObjectCreate(0, "Fibo_High", OBJ_HLINE, 0, 0, high);
    ObjectSetInteger(0, "Fibo_High", OBJPROP_COLOR, clrGreen);
    ObjectSetInteger(0, "Fibo_High", OBJPROP_STYLE, STYLE_SOLID);
    ObjectSetString(0, "Fibo_High", OBJPROP_TEXT, "High: " + DoubleToString(high, _Digits));

    ObjectCreate(0, "Fibo_Low", OBJ_HLINE, 0, 0, low);
    ObjectSetInteger(0, "Fibo_Low", OBJPROP_COLOR, clrRed);
    ObjectSetInteger(0, "Fibo_Low", OBJPROP_STYLE, STYLE_SOLID);
    ObjectSetString(0, "Fibo_Low", OBJPROP_TEXT, "Low: " + DoubleToString(low, _Digits));

    ObjectCreate(0, "Fibo_382", OBJ_HLINE, 0, 0, fibo_382);
    ObjectSetInteger(0, "Fibo_382", OBJPROP_COLOR, clrGold);
    ObjectSetInteger(0, "Fibo_382", OBJPROP_STYLE, STYLE_DASH);
    ObjectSetString(0, "Fibo_382", OBJPROP_TEXT, "38.2%: " + DoubleToString(fibo_382, _Digits));

    ObjectCreate(0, "Fibo_618", OBJ_HLINE, 0, 0, fibo_618);
    ObjectSetInteger(0, "Fibo_618", OBJPROP_COLOR, clrOrange);
    ObjectSetInteger(0, "Fibo_618", OBJPROP_STYLE, STYLE_DASH);
    ObjectSetString(0, "Fibo_618", OBJPROP_TEXT, "61.8%: " + DoubleToString(fibo_618, _Digits));
}

//+------------------------------------------------------------------+
//| エントリー矢印描画                                                |
//+------------------------------------------------------------------+
void DrawEntryArrow(bool is_buy, double price)
{
    string name = "Entry_" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

    ObjectCreate(0, name, OBJ_ARROW, 0, TimeCurrent(), price);
    ObjectSetInteger(0, name, OBJPROP_ARROWCODE, is_buy ? 233 : 234);
    ObjectSetInteger(0, name, OBJPROP_COLOR, is_buy ? clrDodgerBlue : clrOrangeRed);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
}

//+------------------------------------------------------------------+
//| ポジション保有チェック                                            |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| チャートイベント処理                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    //--- チャートクリックでフィボナッチを再描画
    if(id == CHARTEVENT_CLICK)
    {
        double fibo_382, fibo_618;
        int trend;
        CalculateFibonacci(fibo_382, fibo_618, trend);
        ChartRedraw();
    }
}
//+------------------------------------------------------------------+
