// ========== DOSYA: sentinel-optimizer/src/simulator.rs ==========
use sentinel_core::math::model::PureMathModel;
use sentinel_core::math::zscore::OnlineZScore;
use sentinel_core::risk::engine::{RiskConfig, RiskEngine};
use sentinel_core::types::{SignalType, TradeSignal};
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, VecDeque};

fn deserialize_binance_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    match s.to_lowercase().as_str() {
        "true" | "t" | "1" => Ok(true),
        "false" | "f" | "0" => Ok(false),
        _ => Err(serde::de::Error::custom("expected boolean")),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HistoricalTick {
    pub price: f64,
    pub qty: f64,
    #[serde(rename = "time")]
    pub timestamp: i64,
    #[serde(deserialize_with = "deserialize_binance_bool")]
    pub is_buyer_maker: bool,
}

pub struct SimulationResult {
    pub pnl: f64,
    pub sharpe: f64,
    pub max_dd: f64,
    pub trades: usize,
}

#[derive(Clone, Default)]
struct Bucket {
    #[allow(dead_code)]
    open: f64,
    close: f64,
    high: f64,
    low: f64,
    buy_vol: f64,
    sell_vol: f64,
    ticks: i64,
}

pub fn run_simulation(dna: &[f32], ticks: &[HistoricalTick], symbol: &str) -> SimulationResult {
    let weights = dna[0..36].to_vec();
    let biases = dna[36..39].to_vec();

    let model = match PureMathModel::new(weights, biases) {
        Ok(m) => m,
        Err(_) => {
            return SimulationResult {
                pnl: 0.0,
                sharpe: 0.0,
                max_dd: 0.0,
                trades: 0,
            }
        }
    };

    let config = RiskConfig {
        initial_balance: 1000.0,
        max_drawdown_usd: 950.0,
        defensive_drawdown_usd: 800.0,
        cooldown_ms: dna[41] as i64,
        min_hold_time_ms: 1000,
        max_hold_time_ms: 3_600_000,
        base_risk_pct: (dna[42] as f64).clamp(0.15, 0.90),
        base_leverage: 1.0,
        take_profit_pct: (dna[39] as f64).clamp(0.008, 0.06),
        stop_loss_pct: (dna[40] as f64).clamp(0.004, 0.03),
    };

    let mut risk_engine = RiskEngine::new(config.clone());
    let mut z_scores = vec![OnlineZScore::new(1000); 12];
    let mut buckets: VecDeque<Bucket> = VecDeque::with_capacity(400);
    let mut current_bucket = Bucket {
        open: 0.0,
        high: f64::MIN,
        low: f64::MAX,
        ..Default::default()
    };
    let mut current_sec = 0;

    let mut balance = config.initial_balance;
    let mut peak_equity = balance;
    let mut returns = Vec::new();
    let mut trade_count = 0;
    let mut last_signal_ms = 0;
    let mut current_prices = HashMap::new();

    let mut fast_ema = ticks[0].price;
    let mut slow_ema = fast_ema;

    for tick in ticks {
        let sec = tick.timestamp / 1000;
        current_prices.insert(symbol.to_string(), tick.price);

        fast_ema = tick.price * 0.01 + fast_ema * 0.99;
        slow_ema = tick.price * 0.001 + slow_ema * 0.999;

        if current_sec == 0 {
            current_sec = sec;
        }

        if sec > current_sec {
            buckets.push_back(current_bucket.clone());
            if buckets.len() > 300 {
                buckets.pop_front();
            }
            current_sec = sec;
            current_bucket = Bucket {
                open: tick.price,
                close: tick.price,
                high: tick.price,
                low: tick.price,
                ..Default::default()
            };

            if buckets.len() >= 100 {
                let _last_close = buckets.back().unwrap().close;
                let mut highest = f64::MIN;
                let mut lowest = f64::MAX;
                let mut total_buy = 0.0;
                let mut total_sell = 0.0;
                let mut total_ticks = 0;
                let mut gains = 0.0;
                let mut losses = 0.0;
                let mut prev_c = buckets[0].close;

                for b in &buckets {
                    if b.high > highest {
                        highest = b.high;
                    }
                    if b.low < lowest {
                        lowest = b.low;
                    }
                    total_buy += b.buy_vol;
                    total_sell += b.sell_vol;
                    total_ticks += b.ticks;
                    let d = b.close - prev_c;
                    if d > 0.0 {
                        gains += d;
                    } else {
                        losses += d.abs();
                    }
                    prev_c = b.close;
                }

                let ema_diff = (fast_ema - slow_ema) / slow_ema;
                let price_to_slow = (tick.price - slow_ema) / slow_ema;
                let trade_imb = if (total_buy + total_sell) > 0.0 {
                    (total_buy - total_sell) / (total_buy + total_sell)
                } else {
                    0.0
                };
                let rsi = if (gains + losses) > 0.0 {
                    100.0 - (100.0 / (1.0 + (gains / losses.max(1e-9))))
                } else {
                    50.0
                };
                let volatility = if lowest > 0.0 {
                    (highest - lowest) / lowest
                } else {
                    0.0
                };
                let intensity = total_ticks as f64 / buckets.len() as f64;
                let hour = ((tick.timestamp / 3600000) % 24) as f64;
                let time_sine = (hour * std::f64::consts::PI / 12.0).sin();
                let taker_ratio = if (total_buy + total_sell) > 0.0 {
                    total_buy / (total_buy + total_sell)
                } else {
                    0.5
                };

                let mut features = [0.0f32; 12];
                features[0] = z_scores[0].update(ema_diff, 10000.0) as f32;
                features[1] = z_scores[1].update(price_to_slow, 10000.0) as f32;
                features[2] = z_scores[2].update(trade_imb, 1.0) as f32;
                features[3] = z_scores[3].update(rsi, 1.0) as f32;
                features[4] = z_scores[4].update(volatility, 1000.0) as f32;
                features[5] = z_scores[5].update(intensity, 1.0) as f32;
                features[6] = z_scores[6].update(time_sine, 1.0) as f32;
                features[7] = z_scores[7].update(taker_ratio, 1.0) as f32;

                if tick.timestamp - last_signal_ms >= 5000 {
                    if let Ok((sig_type, conf)) = model.predict(&features) {
                        if sig_type != SignalType::Hold && conf > 0.45 {
                            last_signal_ms = tick.timestamp;
                            let signal = TradeSignal {
                                symbol: symbol.to_string(),
                                signal_type: sig_type,
                                confidence_score: conf,
                                recommended_leverage: 1.0,
                                timestamp: tick.timestamp,
                            };

                            if let Ok(qty) = risk_engine.evaluate_signal(
                                &signal,
                                tick.price,
                                balance,
                                tick.timestamp,
                            ) {
                                let side = if sig_type == SignalType::Buy
                                    || sig_type == SignalType::StrongBuy
                                {
                                    "BUY"
                                } else {
                                    "SELL"
                                };
                                let exec_price = if side == "BUY" {
                                    tick.price * 1.0001
                                } else {
                                    tick.price * 0.9999
                                };
                                risk_engine.process_execution(
                                    symbol,
                                    side,
                                    exec_price,
                                    qty,
                                    tick.timestamp,
                                );
                                balance -= (exec_price * qty) * 0.0002;
                            }
                        }
                    }
                }
            }
        }

        current_bucket.close = tick.price;
        if tick.price > current_bucket.high {
            current_bucket.high = tick.price;
        }
        if tick.price < current_bucket.low {
            current_bucket.low = tick.price;
        }
        current_bucket.ticks += 1;
        if tick.is_buyer_maker {
            current_bucket.sell_vol += tick.qty;
        } else {
            current_bucket.buy_vol += tick.qty;
        }

        let close_orders = risk_engine.check_tp_sl(&current_prices, tick.timestamp);
        for (sym, side, qty, price) in close_orders {
            let exec_price = if side == "BUY" {
                price * 1.0001
            } else {
                price * 0.9999
            };
            let realized =
                risk_engine.process_execution(&sym, side, exec_price, qty, tick.timestamp);
            let net_pnl = realized - ((exec_price * qty) * 0.0002);
            balance += net_pnl;
            returns.push(net_pnl / 1000.0);
            trade_count += 1;
            if balance > peak_equity {
                peak_equity = balance;
            }
        }
    }

    let max_dd = if peak_equity > 1000.0 {
        ((peak_equity - balance) / peak_equity) * 100.0
    } else {
        ((1000.0 - balance) / 1000.0) * 100.0
    };
    let sharpe = if returns.len() > 2 {
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var: f64 =
            returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        if var > 0.0 {
            (mean / var.sqrt()) * 15.81
        } else {
            0.0
        }
    } else {
        0.0
    };

    SimulationResult {
        pnl: balance - 1000.0,
        sharpe,
        max_dd,
        trades: trade_count,
    }
}
