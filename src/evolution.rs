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
    _win_rate: f64, // Alt çizgi (Kullanılmıyor, çünkü PnL odaklıyız)
    profit_factor: f64,
) -> f64 {
    // 1. KORKAKLIK CEZASI
    if trades < MIN_REQUIRED_TRADES {
        let diff = MIN_REQUIRED_TRADES.saturating_sub(trades) as f64;
        return -50_000.0 - (diff * 500.0);
    }

    // 2. İFLAS CEZASI
    if max_dd >= MAX_ALLOWED_DD {
        return -100_000.0 - (max_dd * 1000.0);
    }

    // 🔥 3. PnL ODAKLI YAKLAŞIM (Kral PnL'dir, diğer metrikler teferruattır)
    if pnl < 0.0 {
        // ZARARDAYSA: Sadece PnL'yi ve düşük PF'yi cezalandır. WinRate'i umursama.
        let pnl_penalty = pnl * 10_000.0; // Negatif PnL eksi puan getirir (-0.5$ = -5000 puan)
        let pf_penalty = if profit_factor < 1.0 {
            (1.0 - profit_factor) * 5000.0
        } else {
            0.0
        };
        pnl_penalty - pf_penalty
    } else {
        // KÂRDAYSA: PnL'yi devasa ödüllendir!
        let pnl_bonus = pnl * 50_000.0;
        let sharpe_bonus = sharpe.max(0.0) * 5000.0;
        let pf_bonus = profit_factor.max(1.0) * 5000.0;
        let dd_penalty = max_dd * 2000.0;

        pnl_bonus + sharpe_bonus + pf_bonus - dd_penalty
    }
}

pub fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(136);

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

        let mut child_dna = Vec::with_capacity(136);
        for i in 0..136 {
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

                if i < 128 {
                    gene += rng.gen_range(-0.5..0.5) * severity;
                } else if (128..131).contains(&i) {
                    gene += rng.gen_range(-0.2..0.2) * severity;
                } else if i == 131 || i == 132 {
                    gene += rng.gen_range(-0.005..0.005) * severity;
                } else if i == 133 {
                    gene += rng.gen_range(-100.0..100.0) * severity;
                } else if i == 134 {
                    gene += rng.gen_range(-0.01..0.01) * severity;
                } else if i == 135 {
                    gene += rng.gen_range(-0.02..0.02) * severity;
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
