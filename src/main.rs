// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::Result;
use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

mod audit;
mod simulator;
use audit::AuditEngine;
use simulator::{run_simulation, HistoricalTick};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital V14.1 Alpha Strike", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "../sentinel-data/datasets/BTCUSDT_7D.csv"
    )]
    csv_file_path: String,
    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,
    #[arg(short, long, default_value = "200")]
    generations: usize,
    #[arg(short, long, default_value = "500")]
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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧬 VQ-CAPITAL V14.1 ALPHA-STRIKE ENGINE BAŞLATILIYOR...");

    let args = Args::parse();
    let audit = AuditEngine::new();

    // 1. Veriyi RAM'e al (Deterministik Arena)
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.csv_file_path)?;
    let ticks: Vec<HistoricalTick> = reader.deserialize::<HistoricalTick>().flatten().collect();
    info!("✅ {} adet tick RAM'de. Arena Hazır!", ticks.len());

    // 2. Popülasyonu Başlat
    let mut population: Vec<Genome> = (0..args.population)
        .map(|_| create_random_genome())
        .collect();
    let mut best_all_time = population[0].clone();
    best_all_time.fitness = -9999999.0;

    // 3. Hafıza Enjeksiyonu
    if let Some(alpha_dna) = audit.load_best_genome() {
        info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
        population[0] = alpha_dna.clone();
        best_all_time = alpha_dna;
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = 0.25f32;

    for gen in 1..=args.generations {
        let start_time = std::time::Instant::now();

        // 4. PARALEL SİMÜLASYON (Rayon)
        population.par_iter_mut().for_each(|genome| {
            let result = run_simulation(&genome.weights, &ticks, &args.symbol);
            genome.pnl = result.pnl;
            genome.sharpe = result.sharpe;
            genome.max_drawdown = result.max_dd;
            genome.trades = result.trades;
            genome.fitness =
                calculate_fitness(result.pnl, result.sharpe, result.max_dd, result.trades);
            genome.generation = gen;
        });

        // 5. Elitizm ve Sıralama
        population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let gen_best = &population[0];
        let mut is_record = false;

        if gen_best.fitness > best_all_time.fitness {
            best_all_time = gen_best.clone();
            stagnation_counter = 0;
            current_mutation_rate = 0.25;
            is_record = true;
            audit.save_record_break(&best_all_time)?;
        } else {
            stagnation_counter += 1;
            if stagnation_counter > 5 {
                current_mutation_rate = (current_mutation_rate + 0.05).min(0.85);
            }
        }

        // 6. Audit & Log
        audit.log_generation(gen, gen_best, current_mutation_rate)?;
        audit.print_progress(gen, gen_best, is_record, start_time.elapsed().as_secs_f32());

        // 7. Evrim
        population = evolve_population(&population, args.population, current_mutation_rate);
    }
    Ok(())
}

fn calculate_fitness(pnl: f64, sharpe: f64, max_dd: f64, trades: usize) -> f64 {
    if trades < 40 {
        return -10000.0 + (trades as f64 * 100.0);
    }
    if pnl <= 0.0 {
        pnl * (1.0 + (max_dd / 10.0)) - 500.0
    } else {
        pnl * sharpe.max(0.1) * (trades as f64).log10()
    }
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = (0..39).map(|_| rng.gen_range(-1.0..1.0)).collect(); // 36 weights + 3 biases
    dna.push(rng.gen_range(0.008..0.030)); // 39: TP
    dna.push(rng.gen_range(0.004..0.015)); // 40: SL
    dna.push(rng.gen_range(1000.0..10000.0)); // 41: Cooldown
    dna.push(rng.gen_range(0.15..0.50)); // 42: Risk Pct
    Genome {
        weights: dna,
        fitness: -9999999.0,
        pnl: 0.0,
        sharpe: 0.0,
        max_drawdown: 0.0,
        trades: 0,
        generation: 0,
    }
}

fn evolve_population(current_pop: &[Genome], total_size: usize, mut_rate: f32) -> Vec<Genome> {
    let mut new_pop = Vec::with_capacity(total_size);
    let mut rng = rand::thread_rng();
    let elite_count = (total_size / 20).max(2);
    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    while new_pop.len() < total_size {
        let p1 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let p2 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let mut child_dna = Vec::with_capacity(43);
        for i in 0..43 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };
            if rng.gen_bool(mut_rate as f64) {
                if i < 39 {
                    gene += rng.gen_range(-0.2..0.2);
                } else if i == 39 || i == 40 {
                    gene += rng.gen_range(-0.001..0.001);
                } else if i == 41 {
                    gene += rng.gen_range(-500.0..5000.0);
                } else {
                    gene += rng.gen_range(-0.05..0.05);
                }
            }
            // Clamping
            if i == 39 {
                gene = gene.clamp(0.005, 0.06);
            } else if i == 40 {
                gene = gene.clamp(0.004, 0.04);
            } else if i == 41 {
                gene = gene.clamp(500.0, 30000.0);
            } else if i == 42 {
                gene = gene.clamp(0.10, 0.85);
            } else {
                gene = gene.clamp(-3.0, 3.0);
            }
            child_dna.push(gene);
        }
        new_pop.push(Genome {
            weights: child_dna,
            fitness: -9999999.0,
            pnl: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
            trades: 0,
            generation: 0,
        });
    }
    new_pop
}
