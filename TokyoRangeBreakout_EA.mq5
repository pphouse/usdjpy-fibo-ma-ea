//+------------------------------------------------------------------+
//|                                     TokyoRangeBreakout_EA.mq5    |
//|        東京レンジブレイクアウト EA - ゴゴジャン向け                  |
//|        Tokyo Range Breakout Strategy                              |
//+------------------------------------------------------------------+
#property copyright "Tokyo Range Breakout EA"
#property version   "1.00"
#property description "東京時間のレンジをロンドン時間にブレイクアウトする戦略"
#property strict

#include <Trade\Trade.mqh>

//--- 入力パラメータ
input group "=== 時間設定 (サーバー時間 UTC) ==="
input int    TokyoStartHour = 0;              // 東京レンジ開始時 (UTC) = JST 9時
input int    TokyoEndHour   = 6;              // 東京レンジ終了時 (UTC) = JST 15時
input int    LondonStartHour = 6;             // ロンドン取引開始 (UTC) = JST 15時
input int    LondonEndHour   = 14;            // ロンドン取引終了 (UTC) = JST 23時

input group "=== レンジフィルター ==="
input double MinRangePips = 10.0;             // 最小レンジ幅 (Pips)
input double MaxRangePips = 50.0;             // 最大レンジ幅 (Pips)
input double BreakoutBuffer_Pips = 5.0;       // ブレイクアウトバッファ (Pips)

input group "=== 決済設定 ==="
input double TakeProfit_Pips = 10.0;          // 利確 (Pips)
input double StopLoss_Pips   = 50.0;          // 損切り (Pips)

input group "=== ロット設定 ==="
input double LotSize = 0.1;                   // ロットサイズ
input bool   UseAutoLot = false;              // 自動ロット計算
input double RiskPercent = 1.0;               // リスク率 (%)

input group "=== その他設定 ==="
input int    MagicNumber = 202501;            // マジックナンバー
input int    MaxSpreadPips = 5;               // 最大スプレッド (Pips)
input bool   TradeOnFriday = true;            // 金曜日に取引する
input int    ServerGMTOffset = 0;             // サーバーGMTオフセット

//--- グローバル変数
CTrade trade;
double pip_value;

// 日次レンジ管理
double tokyo_high = 0;
double tokyo_low = 0;
bool   range_formed = false;
bool   trade_done_today = false;
datetime last_trade_date = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- PIP値の計算
    pip_value = _Point;
    if(_Digits == 3 || _Digits == 5)
        pip_value = _Point * 10;

    //--- トレード設定
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    Print("=== Tokyo Range Breakout EA 初期化完了 ===");
    Print("東京時間: ", TokyoStartHour, ":00 - ", TokyoEndHour, ":00 UTC");
    Print("ロンドン時間: ", LondonStartHour, ":00 - ", LondonEndHour, ":00 UTC");
    Print("レンジフィルター: ", MinRangePips, " - ", MaxRangePips, " pips");
    Print("TP: ", TakeProfit_Pips, " pips, SL: ", StopLoss_Pips, " pips");
    Print("ブレイクアウトバッファ: ", BreakoutBuffer_Pips, " pips");

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    ObjectDelete(0, "TokyoHigh");
    ObjectDelete(0, "TokyoLow");
    ObjectDelete(0, "TokyoRange");
    Comment("");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    MqlDateTime dt;
    TimeCurrent(dt);
    int current_hour = dt.hour;
    datetime today = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));

    //--- 日付が変わったらリセット
    if(today != last_trade_date)
    {
        ResetDaily();
        last_trade_date = today;
    }

    //--- 金曜日フィルター
    if(!TradeOnFriday && dt.day_of_week == 5)
        return;

    //--- 東京時間: レンジを収集
    if(current_hour >= TokyoStartHour && current_hour < TokyoEndHour)
    {
        CollectTokyoRange();
        return;
    }

    //--- 東京時間終了時にレンジ確定
    if(current_hour == TokyoEndHour && !range_formed && tokyo_high > 0)
    {
        FinalizeRange();
    }

    //--- ロンドン時間: ブレイクアウトチェック
    if(current_hour >= LondonStartHour && current_hour < LondonEndHour)
    {
        if(range_formed && !trade_done_today)
        {
            CheckBreakout();
        }
    }

    //--- チャート情報表示
    DisplayInfo(current_hour);
}

//+------------------------------------------------------------------+
//| 日次リセット                                                      |
//+------------------------------------------------------------------+
void ResetDaily()
{
    tokyo_high = 0;
    tokyo_low = 999999;
    range_formed = false;
    trade_done_today = false;

    ObjectDelete(0, "TokyoHigh");
    ObjectDelete(0, "TokyoLow");
}

//+------------------------------------------------------------------+
//| 東京レンジ収集                                                    |
//+------------------------------------------------------------------+
void CollectTokyoRange()
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double high = (bid + ask) / 2.0;
    double low = high;

    // M1バーのHigh/Lowを使用
    double bar_high = iHigh(_Symbol, PERIOD_M1, 0);
    double bar_low = iLow(_Symbol, PERIOD_M1, 0);

    if(bar_high > tokyo_high)
        tokyo_high = bar_high;
    if(bar_low < tokyo_low)
        tokyo_low = bar_low;
}

//+------------------------------------------------------------------+
//| レンジ確定                                                        |
//+------------------------------------------------------------------+
void FinalizeRange()
{
    double range_pips = (tokyo_high - tokyo_low) / pip_value;

    if(range_pips >= MinRangePips && range_pips <= MaxRangePips)
    {
        range_formed = true;
        Print("東京レンジ確定: High=", DoubleToString(tokyo_high, _Digits),
              " Low=", DoubleToString(tokyo_low, _Digits),
              " Range=", DoubleToString(range_pips, 1), " pips");

        // レンジライン描画
        DrawRangeLines();
    }
    else
    {
        Print("レンジフィルター: ", DoubleToString(range_pips, 1),
              " pips (", MinRangePips, "-", MaxRangePips, " pips外)");
        range_formed = false;
    }
}

//+------------------------------------------------------------------+
//| ブレイクアウトチェック                                              |
//+------------------------------------------------------------------+
void CheckBreakout()
{
    //--- スプレッドチェック
    double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double spread_pips = spread / pip_value;

    if(spread_pips > MaxSpreadPips)
        return;

    double close = iClose(_Symbol, PERIOD_M5, 0);
    double buffer = BreakoutBuffer_Pips * pip_value;

    //--- ブレイクアウト判定
    if(close > tokyo_high + buffer)
    {
        // 上方ブレイクアウト → BUY
        ExecuteTrade(ORDER_TYPE_BUY);
    }
    else if(close < tokyo_low - buffer)
    {
        // 下方ブレイクアウト → SELL
        ExecuteTrade(ORDER_TYPE_SELL);
    }
}

//+------------------------------------------------------------------+
//| トレード実行                                                      |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE order_type)
{
    double lot = CalculateLot();
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    double entry_price, tp_price, sl_price;

    if(order_type == ORDER_TYPE_BUY)
    {
        entry_price = ask;
        tp_price = entry_price + TakeProfit_Pips * pip_value;
        sl_price = entry_price - StopLoss_Pips * pip_value;

        if(trade.Buy(lot, _Symbol, entry_price, sl_price, tp_price,
                     "TRB BUY"))
        {
            trade_done_today = true;
            Print("BUYエントリー: Price=", DoubleToString(entry_price, _Digits),
                  " TP=", DoubleToString(tp_price, _Digits),
                  " SL=", DoubleToString(sl_price, _Digits));
            DrawEntryArrow(true, entry_price);
        }
    }
    else
    {
        entry_price = bid;
        tp_price = entry_price - TakeProfit_Pips * pip_value;
        sl_price = entry_price + StopLoss_Pips * pip_value;

        if(trade.Sell(lot, _Symbol, entry_price, sl_price, tp_price,
                      "TRB SELL"))
        {
            trade_done_today = true;
            Print("SELLエントリー: Price=", DoubleToString(entry_price, _Digits),
                  " TP=", DoubleToString(tp_price, _Digits),
                  " SL=", DoubleToString(sl_price, _Digits));
            DrawEntryArrow(false, entry_price);
        }
    }
}

//+------------------------------------------------------------------+
//| ロット計算                                                        |
//+------------------------------------------------------------------+
double CalculateLot()
{
    if(!UseAutoLot)
        return LotSize;

    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100.0;
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tick_value == 0 || tick_size == 0)
        return LotSize;

    double sl_points = StopLoss_Pips * pip_value / tick_size;
    double lot = risk_amount / (sl_points * tick_value);

    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lot = MathMax(min_lot, MathMin(max_lot, lot));
    lot = MathFloor(lot / lot_step) * lot_step;

    return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| レンジライン描画                                                   |
//+------------------------------------------------------------------+
void DrawRangeLines()
{
    datetime time_start = iTime(_Symbol, PERIOD_H1, 0);

    // 東京High
    ObjectDelete(0, "TokyoHigh");
    ObjectCreate(0, "TokyoHigh", OBJ_HLINE, 0, 0, tokyo_high);
    ObjectSetInteger(0, "TokyoHigh", OBJPROP_COLOR, clrDodgerBlue);
    ObjectSetInteger(0, "TokyoHigh", OBJPROP_STYLE, STYLE_DASH);
    ObjectSetInteger(0, "TokyoHigh", OBJPROP_WIDTH, 1);
    ObjectSetString(0, "TokyoHigh", OBJPROP_TEXT,
                    "Tokyo High: " + DoubleToString(tokyo_high, _Digits));

    // 東京Low
    ObjectDelete(0, "TokyoLow");
    ObjectCreate(0, "TokyoLow", OBJ_HLINE, 0, 0, tokyo_low);
    ObjectSetInteger(0, "TokyoLow", OBJPROP_COLOR, clrOrangeRed);
    ObjectSetInteger(0, "TokyoLow", OBJPROP_STYLE, STYLE_DASH);
    ObjectSetInteger(0, "TokyoLow", OBJPROP_WIDTH, 1);
    ObjectSetString(0, "TokyoLow", OBJPROP_TEXT,
                    "Tokyo Low: " + DoubleToString(tokyo_low, _Digits));

    // ブレイクアウトライン (バッファ付き)
    double buffer = BreakoutBuffer_Pips * pip_value;

    ObjectDelete(0, "BreakHigh");
    ObjectCreate(0, "BreakHigh", OBJ_HLINE, 0, 0, tokyo_high + buffer);
    ObjectSetInteger(0, "BreakHigh", OBJPROP_COLOR, clrLime);
    ObjectSetInteger(0, "BreakHigh", OBJPROP_STYLE, STYLE_DOT);

    ObjectDelete(0, "BreakLow");
    ObjectCreate(0, "BreakLow", OBJ_HLINE, 0, 0, tokyo_low - buffer);
    ObjectSetInteger(0, "BreakLow", OBJPROP_COLOR, clrRed);
    ObjectSetInteger(0, "BreakLow", OBJPROP_STYLE, STYLE_DOT);
}

//+------------------------------------------------------------------+
//| エントリー矢印描画                                                |
//+------------------------------------------------------------------+
void DrawEntryArrow(bool is_buy, double price)
{
    string name = "TRB_Entry_" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);

    ObjectCreate(0, name, OBJ_ARROW, 0, TimeCurrent(), price);
    ObjectSetInteger(0, name, OBJPROP_ARROWCODE, is_buy ? 233 : 234);
    ObjectSetInteger(0, name, OBJPROP_COLOR, is_buy ? clrDodgerBlue : clrOrangeRed);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
}

//+------------------------------------------------------------------+
//| チャート情報表示                                                  |
//+------------------------------------------------------------------+
void DisplayInfo(int current_hour)
{
    string session = "";
    if(current_hour >= TokyoStartHour && current_hour < TokyoEndHour)
        session = "TOKYO (レンジ収集中)";
    else if(current_hour >= LondonStartHour && current_hour < LondonEndHour)
        session = "LONDON (ブレイクアウト待ち)";
    else
        session = "OFF SESSION";

    double range_pips = (tokyo_high > 0 && tokyo_low < 999999) ?
                        (tokyo_high - tokyo_low) / pip_value : 0;

    string info = "";
    info += "━━━ Tokyo Range Breakout EA ━━━\n";
    info += "Session: " + session + "\n";
    info += "Tokyo High: " + DoubleToString(tokyo_high, _Digits) + "\n";
    info += "Tokyo Low:  " + DoubleToString(tokyo_low < 999999 ? tokyo_low : 0, _Digits) + "\n";
    info += "Range: " + DoubleToString(range_pips, 1) + " pips\n";
    info += "Range Valid: " + (range_formed ? "YES" : "NO") + "\n";
    info += "Trade Today: " + (trade_done_today ? "DONE" : "WAITING") + "\n";
    info += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    // 本日の損益
    double today_profit = GetTodayProfit();
    info += "Today P/L: " + DoubleToString(today_profit, 2) + "\n";

    Comment(info);
}

//+------------------------------------------------------------------+
//| 本日の損益取得                                                    |
//+------------------------------------------------------------------+
double GetTodayProfit()
{
    double profit = 0;
    datetime today_start = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));

    // クローズ済みの本日分
    HistorySelect(today_start, TimeCurrent());
    int total = HistoryDealsTotal();
    for(int i = 0; i < total; i++)
    {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == MagicNumber)
        {
            profit += HistoryDealGetDouble(ticket, DEAL_PROFIT);
        }
    }

    // オープン中のポジション
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
                profit += PositionGetDouble(POSITION_PROFIT);
            }
        }
    }

    return profit;
}
//+------------------------------------------------------------------+
