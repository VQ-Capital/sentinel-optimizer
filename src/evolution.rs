// ========== DOSYA: sentinel-optimizer/src/evolution.rs ==========
use crate::settings::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub weights: Vec<f32>,
    pub fitness: f64,
    pub pnl: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub trades: usize,
    pub generation: usize,
    #[serde(default)]
    pub win_rate: f64,
    #[serde(default)]
    pub profit_factor: f64,
}

pub fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    win_rate: f64,
    profit_factor: f64,
) -> f64 {
    // 1. KORKAKLIK CEZASI
    if trades < MIN_REQUIRED_TRADES {
        let diff = MIN_REQUIRED_TRADES.saturating_sub(trades) as f64;
        return -10_000_000.0 - (diff * diff * 10_000.0) + (pnl * 10.0);
    }

    // 2. İFLAS CEZASI
    if max_dd >= MAX_ALLOWED_DD {
        return -5_000_000.0 - (max_dd * 10_000.0);
    }

    let mut penalty = 0.0;

    // 3. MATEMATİKSEL HEDEFLER CEZASI
    if profit_factor < TARGET_PROFIT_FACTOR {
        let diff = TARGET_PROFIT_FACTOR - profit_factor;
        penalty += diff * diff * 5000.0;
    }
    if win_rate < TARGET_WIN_RATE {
        let diff = TARGET_WIN_RATE - win_rate;
        penalty += diff * 50.0;
    }

    if pnl <= 0.0 {
        // 🔥 ZARARDAYKEN: Komisyon korkusunu yenmesi için PnL cezası hafif.
        // Ama Profit Factor 1.0 üzerindeyse cesaretlendirme bonusu ver.
        let pnl_score = pnl * 10_000.0;

        let guide_bonus = if profit_factor > 1.0 {
            profit_factor * 50_000.0
        } else {
            0.0
        };
        let wr_guide = win_rate * 500.0;

        pnl_score + guide_bonus + wr_guide - penalty - (max_dd * 100.0)
    } else {
        // 🚀 IŞIĞA ÇIKIŞ (NİRVANA)
        let jackpot = 1_000_000.0;
        let pnl_score = pnl * 100_000.0;
        let pf_bonus = profit_factor.max(1.0) * 50_000.0;
        let wr_bonus = win_rate * 5000.0;

        // 🔥 CERRAHİ: Clippy hatası düzeltildi! .max().min() yerine clamp() kullanıldı.
        let sharpe_bonus = sharpe.clamp(0.0, 10.0) * 10_000.0;

        jackpot + pnl_score + pf_bonus + wr_bonus + sharpe_bonus - penalty - (max_dd * 100.0)
    }
}

pub fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(138);

    for _ in 0..128 {
        dna.push(rng.gen_range(-0.5..0.5));
    }

    dna.push(rng.gen_range(-0.5..0.0)); // Hold
    dna.push(rng.gen_range(0.1..0.5)); // Buy
    dna.push(rng.gen_range(0.1..0.5)); // Sell

    dna.push(rng.gen_range(DNA_TP_MIN..DNA_TP_MAX));
    dna.push(rng.gen_range(DNA_SL_MIN..DNA_SL_MAX));
    dna.push(rng.gen_range(DNA_COOLDOWN_MIN..DNA_COOLDOWN_MAX));
    dna.push(rng.gen_range(0.01..0.05));
    dna.push(rng.gen_range(DNA_CONFIDENCE_MIN..DNA_CONFIDENCE_MAX));
    dna.push(rng.gen_range(DNA_LEVERAGE_MIN..DNA_LEVERAGE_MAX));
    dna.push(rng.gen_range(DNA_HOLD_TIME_MIN..DNA_HOLD_TIME_MAX));

    Genome {
        weights: dna,
        fitness: f64::NEG_INFINITY,
        pnl: 0.0,
        sharpe: 0.0,
        max_drawdown: 0.0,
        trades: 0,
        generation: 0,
        win_rate: 0.0,
        profit_factor: 0.0,
    }
}

pub fn evolve_population(
    current_pop: &[Genome],
    total_size: usize,
    mut_rate: f32,
    is_cataclysm: bool,
) -> Vec<Genome> {
    let mut new_pop = Vec::with_capacity(total_size);
    let mut rng = rand::thread_rng();

    let elite_count = if is_cataclysm {
        (total_size / 20).max(2)
    } else {
        (total_size / 10).max(2)
    };

    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    let random_injection = if is_cataclysm {
        (total_size as f32 * EXTINCTION_DEATH_RATE) as usize
    } else {
        (total_size as f32 * 0.10) as usize
    };

    while new_pop.len() < total_size - random_injection {
        let p1 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let p2 = &current_pop[rng.gen_range(0..elite_count * 2)];

        let mut child_dna = Vec::with_capacity(138);
        for i in 0..138 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };

            if i < 131 && rng.gen_bool(0.05) {
                gene = -gene;
            }

            if rng.gen_bool(mut_rate as f64) {
                let severity = if is_cataclysm { 3.0 } else { 1.0 };
                let is_micro = rng.gen_bool(0.8);

                if i < 128 {
                    let scale = if is_micro { 0.02 } else { 0.3 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if (128..131).contains(&i) {
                    let scale = if is_micro { 0.01 } else { 0.1 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 131 || i == 132 {
                    let scale = if is_micro { 0.0005 } else { 0.005 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 133 {
                    let scale = if is_micro { 10.0 } else { 100.0 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 134 {
                    let scale = if is_micro { 0.001 } else { 0.01 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 135 {
                    let scale = if is_micro { 0.002 } else { 0.02 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 136 {
                    let scale = if is_micro { 0.1 } else { 1.0 };
                    gene += rng.gen_range(-scale..scale) * severity;
                } else if i == 137 {
                    let scale = if is_micro { 300_000.0 } else { 3_600_000.0 };
                    gene += rng.gen_range(-scale..scale) * severity;
                }
            }

            if i == 131 {
                gene = gene.clamp(DNA_TP_MIN, DNA_TP_MAX);
            } else if i == 132 {
                gene = gene.clamp(DNA_SL_MIN, DNA_SL_MAX);
            } else if i == 133 {
                gene = gene.clamp(DNA_COOLDOWN_MIN, DNA_COOLDOWN_MAX);
            } else if i == 134 {
                gene = gene.clamp(0.01, 0.05);
            } else if i == 135 {
                gene = gene.clamp(DNA_CONFIDENCE_MIN, DNA_CONFIDENCE_MAX);
            } else if i == 136 {
                gene = gene.clamp(DNA_LEVERAGE_MIN, DNA_LEVERAGE_MAX);
            } else if i == 137 {
                gene = gene.clamp(DNA_HOLD_TIME_MIN, DNA_HOLD_TIME_MAX);
            } else if (128..131).contains(&i) {
                gene = gene.clamp(-1.0, 1.0);
            } else {
                gene = gene.clamp(-3.0, 3.0);
            }

            child_dna.push(gene);
        }
        new_pop.push(Genome {
            weights: child_dna,
            fitness: f64::NEG_INFINITY,
            pnl: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
            trades: 0,
            generation: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
        });
    }

    while new_pop.len() < total_size {
        new_pop.push(create_random_genome());
    }
    new_pop
}
