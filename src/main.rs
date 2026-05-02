// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::Result;
use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use tracing::{info, warn};

mod simulator;
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

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.csv_file_path)?;
    let mut ticks = Vec::with_capacity(5_000_000);
    for tick in reader.deserialize::<HistoricalTick>().flatten() {
        ticks.push(tick);
    }
    info!("✅ {} adet tick RAM'de. Arena Hazır!", ticks.len());

    let log_path = "optimization_audit_log.csv";
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    if log_file.metadata()?.len() == 0 {
        writeln!(
            log_file,
            "Timestamp,Gen,PnL,Sharpe,MaxDD,Trades,Fit,MutRate,TP,SL,Risk"
        )?;
    }

    let mut population: Vec<Genome> = (0..args.population)
        .map(|_| create_random_genome())
        .collect();
    let mut best_all_time = Genome {
        fitness: -9999999.0,
        weights: vec![],
        pnl: 0.0,
        sharpe: 0.0,
        max_drawdown: 0.0,
        trades: 0,
        generation: 0,
    };

    // Hafıza yükleme (Varsa)
    if let Ok(data) = fs::read_to_string("hall_of_fame.json") {
        if let Ok(history) = serde_json::from_str::<Vec<Genome>>(&data) {
            if let Some(last_best) = history.first() {
                info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
                population[0] = last_best.clone();
                best_all_time = last_best.clone();
            }
        }
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = 0.25f32;

    for gen in 1..=args.generations {
        let start_time = std::time::Instant::now();

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

        population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let gen_best = &population[0];

        if gen_best.fitness > best_all_time.fitness {
            best_all_time = gen_best.clone();
            stagnation_counter = 0;
            current_mutation_rate = 0.25f32;
            info!("🌟 REKOR! [Gen {}] PnL: ${:.2} | Trades: {} | Fit: {:.2} | TP: {:.3} | SL: {:.3} | Risk: {:.1}%",
                gen, gen_best.pnl, gen_best.trades, gen_best.fitness, gen_best.weights[39], gen_best.weights[40], gen_best.weights[42] * 100.0);
            let _ = fs::write(
                "hall_of_fame.json",
                serde_json::to_string_pretty(&vec![&best_all_time]).unwrap_or_default(),
            );
        } else {
            stagnation_counter += 1;
            if stagnation_counter > 5 {
                current_mutation_rate = (current_mutation_rate + 0.05f32).min(0.85f32);
                warn!(
                    "⚠️ DURGUNLUK ({} nesil)! Mutasyon: {:.2}",
                    stagnation_counter, current_mutation_rate
                );
            }
            info!(
                "🔄 Gen: {} | PnL: ${:.2} | Trades: {} | Fit: {:.2} | Süre: {:.2}s",
                gen,
                gen_best.pnl,
                gen_best.trades,
                gen_best.fitness,
                start_time.elapsed().as_secs_f32()
            );
        }

        writeln!(
            log_file,
            "{},{},{:.2},{:.2},{:.2},{},{:.2},{:.2},{:.4},{:.4},{:.2}",
            chrono::Utc::now().to_rfc3339(),
            gen,
            gen_best.pnl,
            gen_best.sharpe,
            gen_best.max_drawdown,
            gen_best.trades,
            gen_best.fitness,
            current_mutation_rate,
            gen_best.weights[39],
            gen_best.weights[40],
            gen_best.weights[42]
        )?;

        population = evolve_population(&population, args.population, current_mutation_rate);
    }
    Ok(())
}

fn calculate_fitness(pnl: f64, sharpe: f64, max_dd: f64, trades: usize) -> f64 {
    if trades < 60 {
        return -10000.0 + (trades as f64 * 100.0);
    }
    if pnl <= 0.0 {
        pnl * (1.0 + (max_dd / 10.0)) - 100.0 // Zarar edenlere daha sert ceza
    } else {
        // Alpha Strike Reward
        pnl * sharpe.powi(2).max(0.1) * (trades as f64).log10()
    }
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = (0..39).map(|_| rng.gen_range(-1.5..1.5)).collect();
    dna.push(rng.gen_range(0.008..0.030)); // 39: TP
    dna.push(rng.gen_range(0.004..0.015)); // 40: SL
    dna.push(rng.gen_range(500.0..5000.0)); // 41: Cooldown
    dna.push(rng.gen_range(0.20..0.70)); // 42: Risk Pct (Minimum %20!)
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
    let elite_count = total_size / 10;
    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    while new_pop.len() < (total_size as f64 * 0.90) as usize {
        let p1 = &current_pop[rng.gen_range(0..elite_count)];
        let p2 = &current_pop[rng.gen_range(0..elite_count)];
        let mut child_dna = Vec::with_capacity(43);
        for i in 0..43 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };
            if rng.gen_bool(mut_rate as f64) {
                if i < 39 {
                    gene += rng.gen_range(-0.3..0.3);
                } else if i == 39 || i == 40 {
                    gene += rng.gen_range(-0.002..0.002);
                } else if i == 41 {
                    gene += rng.gen_range(-200.0..200.0);
                } else {
                    gene += rng.gen_range(-0.1..0.1);
                }
            }
            if i == 39 {
                gene = gene.clamp(0.006, 0.06);
            } else if i == 40 {
                gene = gene.clamp(0.004, 0.04);
            } else if i == 41 {
                gene = gene.clamp(200.0, 10000.0);
            } else if i == 42 {
                gene = gene.clamp(0.15, 0.85);
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
    while new_pop.len() < total_size {
        new_pop.push(create_random_genome());
    }
    new_pop
}
