// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

mod audit;
mod settings;
mod simulator;

use audit::AuditEngine;
use settings::*;
use simulator::{run_simulation, HistoricalTick};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital V17.0 Non-Linear MLP Engine", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "../sentinel-data/datasets/BTCUSDT_01_2026.csv"
    )]
    csv_file_path: String,
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,
    #[arg(short, long, default_value = "500")]
    generations: usize,
    #[arg(short, long, default_value = "250")]
    population: usize,
}

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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧬 VQ-CAPITAL V17.0 NON-LINEAR MLP ENGINE BAŞLATILIYOR...");

    let args = Args::parse();
    let audit = AuditEngine::new();

    let _ = audit.initialize_csv();

    settings::print_experiment_manifest(
        &args.symbol,
        &args.csv_file_path,
        args.generations,
        args.population,
    );

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.csv_file_path)
        .context("CSV dosyasi okunamadi")?;
    let ticks: Vec<HistoricalTick> = reader.deserialize::<HistoricalTick>().flatten().collect();
    info!("✅ {} adet tick RAM'de. Arena Hazır!", ticks.len());

    let mut population: Vec<Genome> = (0..args.population)
        .map(|_| create_random_genome())
        .collect();
    let mut best_all_time = population[0].clone();
    best_all_time.fitness = f64::NEG_INFINITY;
    let mut has_loaded_memory = false;

    if let Some(alpha_dna) = audit.load_best_genome() {
        if alpha_dna.weights.len() == 136 {
            info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
            population[0] = alpha_dna.clone();
            best_all_time = alpha_dna;
            has_loaded_memory = true;
        } else {
            warn!("⚠️ Eski nesil DNA tespit edildi (Uzunluk: {}). Yeni MLP mimarisi (136 Gen) için sıfırdan başlanıyor.", alpha_dna.weights.len());
        }
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = MUTATION_BASE_RATE;

    for gen in 1..=args.generations {
        let start_time = std::time::Instant::now();

        population.par_iter_mut().for_each(|genome| {
            let result = run_simulation(&genome.weights, &ticks, &args.symbol);
            genome.pnl = result.pnl;
            genome.sharpe = result.sharpe;
            genome.max_drawdown = result.max_dd;
            genome.trades = result.trades;
            genome.win_rate = result.win_rate;
            genome.profit_factor = result.profit_factor;

            genome.fitness = calculate_fitness(
                result.pnl,
                result.sharpe,
                result.max_dd,
                result.trades,
                result.win_rate,
                result.profit_factor,
            );
            genome.generation = gen;
        });

        population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let gen_best = &population[0];
        let mut is_record = false;
        let is_first_run_of_loaded = has_loaded_memory && gen == 1;

        if gen_best.fitness > (best_all_time.fitness + 0.1) || is_first_run_of_loaded {
            best_all_time = gen_best.clone();
            stagnation_counter = 0;
            current_mutation_rate = MUTATION_BASE_RATE;
            is_record = true;
            if !is_first_run_of_loaded {
                let _ = audit.save_record_break(&best_all_time);
            }
        } else {
            stagnation_counter += 1;
            if stagnation_counter > STAGNATION_LIMIT {
                warn!(
                    "🌋 [MASS EXTINCTION] Stagnation! Nüfusun %{} imha ediliyor.",
                    EXTINCTION_DEATH_RATE * 100.0
                );
                current_mutation_rate = MUTATION_CATACLYSM_RATE;
            } else {
                current_mutation_rate =
                    (current_mutation_rate + 0.05).min(MUTATION_CATACLYSM_RATE - 0.1);
            }
        }

        let time_sec = start_time.elapsed().as_secs_f32();
        let _ = audit.log_generation(gen, gen_best, current_mutation_rate, time_sec, is_record);

        population = evolve_population(
            &population,
            args.population,
            current_mutation_rate,
            stagnation_counter > STAGNATION_LIMIT,
        );
        if stagnation_counter > STAGNATION_LIMIT {
            stagnation_counter = 0;
        }
    }
    Ok(())
}

fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    _win_rate: f64,
    profit_factor: f64,
) -> f64 {
    if trades < MIN_REQUIRED_TRADES {
        return -10_000_000.0 - ((MIN_REQUIRED_TRADES.saturating_sub(trades)) as f64 * 1000.0);
    }
    if max_dd >= MAX_ALLOWED_DD {
        return -5_000_000.0 - (max_dd * 10_000.0);
    }
    if profit_factor < 1.0 {
        return (pnl * 1000.0) + (sharpe * 1000.0) - ((1.0 - profit_factor) * 50_000.0);
    }
    let expected_value_per_trade = pnl / (trades as f64);
    let base_score = pnl * 5000.0;
    let ev_bonus = expected_value_per_trade * 1_000_000.0;
    let sharpe_bonus = sharpe * 50_000.0;
    let pf_bonus = profit_factor * 20_000.0;
    base_score + ev_bonus + sharpe_bonus + pf_bonus - (max_dd * 5000.0)
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(136);

    // İlk 128 gen (w1: 96, b1: 8, w2: 24)
    for _ in 0..128 {
        dna.push(rng.gen_range(-1.0..1.0));
    }

    // b2: 3 Çıktı nöronunun yanlılıkları (Hold, Buy, Sell)
    dna.push(rng.gen_range(-0.8..-0.2));
    dna.push(rng.gen_range(0.2..1.0));
    dna.push(rng.gen_range(0.2..1.0));

    // Risk ve Genetik Parametreler (131..136)
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

fn evolve_population(
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
        (total_size as f32 * 0.15) as usize
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
                let severity = if is_cataclysm { 2.0 } else { 1.0 };
                if i < 128 {
                    gene += rng.gen_range(-0.5..0.5) * severity;
                } else if (128..131).contains(&i) {
                    gene += rng.gen_range(-0.3..0.3) * severity;
                } else if i == 131 || i == 132 {
                    gene += rng.gen_range(-0.001..0.001) * severity;
                } else if i == 133 {
                    gene += rng.gen_range(-100.0..100.0) * severity;
                } else if i == 135 {
                    gene += rng.gen_range(-0.02..0.02) * severity;
                } else {
                    gene += rng.gen_range(-0.005..0.005);
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
