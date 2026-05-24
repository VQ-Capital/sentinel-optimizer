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

// ========== DOSYA: sentinel-optimizer/src/evolution.rs İÇİNDEKİ calculate_fitness FONKSİYONU ==========

pub fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    win_rate: f64,
    profit_factor: f64,
) -> f64 {
    let mut penalty = 0.0;
    let mut bonus = 0.0;

    // 1. KORKAKLIK CEZASI (Uçurum yok, sürekli eğim)
    if trades < MIN_REQUIRED_TRADES {
        let diff = MIN_REQUIRED_TRADES.saturating_sub(trades) as f64;
        penalty += diff * diff * 50.0;
    }

    // 2. İFLAS CEZASI
    if max_dd >= MAX_ALLOWED_DD {
        let diff = max_dd - MAX_ALLOWED_DD;
        penalty += diff * 1000.0;
    }

    // 3. MATEMATİKSEL HEDEFLER CEZASI (Çarpanlar düşürüldü ki PnL'i ezmesin)
    if profit_factor < TARGET_PROFIT_FACTOR {
        let diff = TARGET_PROFIT_FACTOR - profit_factor;
        penalty += diff * diff * 5000.0;
    }
    if win_rate < TARGET_WIN_RATE {
        let diff = TARGET_WIN_RATE - win_rate;
        penalty += diff * diff * 50.0;
    }

    // 4. AKTİVİTE VE BAŞARI ÖDÜLÜ
    let activity_bonus = (trades as f64) * (win_rate / 100.0) * 10.0;
    bonus += activity_bonus;

    if profit_factor >= 1.0 {
        bonus += profit_factor * 10_000.0;
        bonus += sharpe * 5000.0;
    }

    // 5. PnL EN ÖNEMLİ METRİKTİR
    let pnl_score = pnl * 1000.0;

    // TOPLAM FİTNESS (Tüm çarpanlar dengeli bir şekilde birleştirildi)
    pnl_score + bonus - penalty - (max_dd * 100.0)
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

// ========== DOSYA: sentinel-optimizer/src/evolution.rs (SADECE evolve_population kısmını değiştir) ==========

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

    // Elitleri koru
    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    // Kıyamet anında dışarıdan tamamen yeni, rastgele kan enjekte et (Diversity Injection)
    let random_injection = if is_cataclysm {
        (total_size as f32 * EXTINCTION_DEATH_RATE) as usize
    } else {
        (total_size as f32 * 0.10) as usize
    };

    while new_pop.len() < total_size - random_injection {
        // Uniform Crossover (Çaprazlama)
        let p1 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let p2 = &current_pop[rng.gen_range(0..elite_count * 2)];

        let mut child_dna = Vec::with_capacity(136);
        for i in 0..136 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };

            // Gen işaretini tersine çevirme (Sign Flip Mutation)
            if i < 131 && rng.gen_bool(0.05) {
                gene = -gene;
            }

            // Ağırlık Mutasyonu (Gaussian/Uniform Noise)
            if rng.gen_bool(mut_rate as f64) {
                let severity = if is_cataclysm { 3.0 } else { 1.0 }; // Kıyamette şiddeti 3x artır

                if i < 128 {
                    // Hidden layer ağırlıkları
                    gene += rng.gen_range(-0.5..0.5) * severity;
                } else if (128..131).contains(&i) {
                    // Bias'lar
                    gene += rng.gen_range(-0.2..0.2) * severity;
                } else if i == 131 || i == 132 {
                    // TP / SL mutasyonu
                    gene += rng.gen_range(-0.005..0.005) * severity;
                } else if i == 133 {
                    // Cooldown mutasyonu
                    gene += rng.gen_range(-100.0..100.0) * severity;
                } else if i == 134 {
                    // Risk mutasyonu
                    gene += rng.gen_range(-0.01..0.01) * severity;
                } else if i == 135 {
                    // Confidence mutasyonu
                    gene += rng.gen_range(-0.02..0.02) * severity;
                }
            }

            // Sınırlandırmalar (Clamping)
            if i == 131 {
                gene = gene.clamp(DNA_TP_MIN, DNA_TP_MAX);
            } else if i == 132 {
                gene = gene.clamp(DNA_SL_MIN, DNA_SL_MAX);
            } else if i == 133 {
                gene = gene.clamp(DNA_COOLDOWN_MIN, DNA_COOLDOWN_MAX);
            } else if i == 134 {
                gene = gene.clamp(0.01, 0.05); // Risk limiti
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

    // Nüfusun geri kalanını yepyeni rastgele bireylerle doldur
    while new_pop.len() < total_size {
        new_pop.push(create_random_genome());
    }
    new_pop
}

#[cfg(test)]
mod tests;
