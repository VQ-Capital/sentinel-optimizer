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
        Err(_) => return dead_result(),
    };

    let risk_config = RiskConfig {
        initial_balance: 1000.0,
        max_drawdown_usd: 950.0,
        defensive_drawdown_usd: 800.0,
        cooldown_ms: dna[41] as i64,
        min_hold_time_ms: 1000,
        max_hold_time_ms: 3_600_000,
        // 🔥 CERRAHİ: Risk katı şekilde %1 ile %5 arasına kilitlendi. Kamikaze yasaklandı.
        base_risk_pct: (dna[42] as f64).clamp(0.01, 0.05),
        base_leverage: 1.0,
        take_profit_pct: (dna[39] as f64).clamp(0.006, 0.04),
        stop_loss_pct: (dna[40] as f64).clamp(0.004, 0.02),
    };

    let mut risk_engine = RiskEngine::new(risk_config.clone());
    let mut z_scores = vec![OnlineZScore::new(1000); 12];
    let mut buckets: VecDeque<Bucket> = VecDeque::with_capacity(64);
    let mut current_bucket = Bucket {
        open: ticks[0].price,
        high: f64::MIN,
        low: f64::MAX,
        ..Default::default()
    };

    let mut current_sec = ticks[0].timestamp / 1000;
    let mut balance = risk_config.initial_balance;
    let mut peak_equity = balance;
    let mut returns = Vec::new();
    let mut trade_count = 0;
    let mut last_signal_ms = 0;
    let mut current_prices = HashMap::new();

    for tick in ticks {
        let sec = tick.timestamp / 1000;
        current_prices.insert(symbol.to_string(), tick.price);

        if sec > current_sec {
            current_bucket.close = tick.price;
            buckets.push_back(current_bucket.clone());
            if buckets.len() > 60 {
                buckets.pop_front();
            }

            current_sec = sec;
            current_bucket = Bucket {
                open: tick.price,
                high: tick.price,
                low: tick.price,
                ..Default::default()
            };

            if buckets.len() >= 25 {
                let first_open = buckets.front().unwrap().open;
                let last_close = buckets.back().unwrap().close;

                let mut total_buy_vol = 0.0;
                let mut total_sell_vol = 0.0;
                let mut total_ticks = 0;
                let mut highest = f64::MIN;
                let mut lowest = f64::MAX;
                let mut gains = 0.0;
                let mut losses = 0.0;
                let mut prev_c = first_open;
                let mut sum_price = 0.0;

                for b in &buckets {
                    total_buy_vol += b.buy_vol;
                    total_sell_vol += b.sell_vol;
                    total_ticks += b.ticks;
                    sum_price += b.close;
                    if b.high > highest {
                        highest = b.high;
                    }
                    if b.low < lowest {
                        lowest = b.low;
                    }
                    let d = b.close - prev_c;
                    if d > 0.0 {
                        gains += d;
                    } else {
                        losses += d.abs();
                    }
                    prev_c = b.close;
                }

                let velocity = (last_close - first_open) / first_open;
                let trade_imb = if (total_buy_vol + total_sell_vol) > 0.0 {
                    (total_buy_vol - total_sell_vol) / (total_buy_vol + total_sell_vol)
                } else {
                    0.0
                };
                let volatility = (highest - lowest) / lowest;
                let rsi = if (gains + losses) > 0.0 {
                    100.0 - (100.0 / (1.0 + (gains / losses.max(1e-9))))
                } else {
                    50.0
                };
                let taker_ratio = if (total_buy_vol + total_sell_vol) > 0.0 {
                    total_buy_vol / (total_buy_vol + total_sell_vol)
                } else {
                    0.5
                };
                let intensity = total_ticks as f64 / buckets.len() as f64;
                let pos_in_range = if highest > lowest {
                    (last_close - lowest) / (highest - lowest)
                } else {
                    0.5
                };
                let hour = ((tick.timestamp / 3600000) % 24) as f64;
                let time_sin = (hour * std::f64::consts::PI / 12.0).sin();
                let avg_price = sum_price / buckets.len() as f64;
                let price_to_mean = (last_close - avg_price) / avg_price;

                let mut features = [0.0f32; 12];
                features[0] = z_scores[0].update(velocity, 10000.0) as f32;
                features[1] = z_scores[1].update(trade_imb, 1.0) as f32;
                features[2] = z_scores[2].update(0.0, 1.0) as f32;
                features[3] = z_scores[3].update(0.0, 1.0) as f32;
                features[4] = z_scores[4].update(rsi, 1.0) as f32;
                features[5] = z_scores[5].update(volatility, 10000.0) as f32;
                features[6] = z_scores[6].update(taker_ratio, 1.0) as f32;
                features[7] = z_scores[7].update(intensity, 1.0) as f32;
                features[8] = z_scores[8].update(pos_in_range, 1.0) as f32;
                features[9] = z_scores[9].update(total_buy_vol + total_sell_vol, 1.0) as f32;
                features[10] = z_scores[10].update(time_sin, 1.0) as f32;
                features[11] = z_scores[11].update(price_to_mean, 1000.0) as f32;

                if tick.timestamp - last_signal_ms >= risk_config.cooldown_ms {
                    if let Ok((sig_type, conf)) = model.predict(&features) {
                        if sig_type != SignalType::Hold && conf > 0.42 {
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
                                    tick.price * 1.0002
                                } else {
                                    tick.price * 0.9998
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
                price * 1.0002
            } else {
                price * 0.9998
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

    let max_dd = if peak_equity > 0.0 {
        ((peak_equity - balance) / peak_equity) * 100.0
    } else {
        0.0
    };
    let sharpe = calculate_sharpe(&returns);

    SimulationResult {
        pnl: balance - 1000.0,
        sharpe,
        max_dd,
        trades: trade_count,
    }
}

fn calculate_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 5 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    if var > 0.0 {
        (mean / var.sqrt()) * 15.81
    } else {
        0.0
    }
}

fn dead_result() -> SimulationResult {
    SimulationResult {
        pnl: -1000.0,
        sharpe: 0.0,
        max_dd: 100.0,
        trades: 0,
    }
}
