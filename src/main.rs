// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use tracing::{info, warn};

mod audit;
mod evolution;
mod settings;
mod simulator;

use audit::AuditEngine;
use evolution::{calculate_fitness, create_random_genome, evolve_population, Genome};
use settings::*;
use simulator::{run_simulation, HistoricalTick};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital V17.5 Non-Linear MLP Engine", long_about = None)]
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

    // 🔥 YENİ: Hızlı Sağlamlama (Validation) Modu
    #[arg(long, default_value_t = false)]
    oos_test: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(&args.csv_file_path)
        .context("CSV dosyasi okunamadi")?;
    let ticks: Vec<HistoricalTick> = reader.deserialize::<HistoricalTick>().flatten().collect();

    // 🚀🚀🚀 OUT-OF-SAMPLE VALIDATION MODU 🚀🚀🚀
    if args.oos_test {
        info!(
            "⚡ VQ-CAPITAL OOS-VALIDATOR BAŞLATILIYOR... (Dataset: {})",
            args.csv_file_path
        );

        let mut baked_dna = Vec::with_capacity(138);
        baked_dna.extend_from_slice(sentinel_core::dna::W1);
        baked_dna.extend_from_slice(sentinel_core::dna::B1);
        baked_dna.extend_from_slice(sentinel_core::dna::W2);
        baked_dna.extend_from_slice(sentinel_core::dna::B2);

        baked_dna.push(sentinel_core::dna::TAKE_PROFIT_PCT as f32);
        baked_dna.push(sentinel_core::dna::STOP_LOSS_PCT as f32);
        baked_dna.push(sentinel_core::dna::COOLDOWN_MS as f32);
        baked_dna.push(sentinel_core::dna::RISK_PCT as f32);
        baked_dna.push(sentinel_core::dna::MIN_CONFIDENCE as f32);
        baked_dna.push(sentinel_core::dna::LEVERAGE as f32);
        baked_dna.push(sentinel_core::dna::MAX_HOLD_TIME_MS as f32);

        let start_time = std::time::Instant::now();
        let result = run_simulation(&baked_dna, &ticks, &args.symbol);

        println!("\n=========================================================");
        println!(
            "📈 OOS VALIDATION SONUCU (SÜRE: {:.2}s)",
            start_time.elapsed().as_secs_f64()
        );
        println!("=========================================================");
        println!("PnL           : ${:.4}", result.pnl);
        println!("İşlem Sayısı  : {}", result.trades);
        println!("Win Rate      : {:.2}%", result.win_rate);
        println!("Profit Factor : {:.2}", result.profit_factor);
        println!("Sharpe Ratio  : {:.2}", result.sharpe);
        println!("Max Drawdown  : {:.2}%", result.max_dd);
        println!("=========================================================\n");
        return Ok(());
    }

    info!("🧬 VQ-CAPITAL V17.5 NON-LINEAR MLP ENGINE BAŞLATILIYOR...");
    let audit = AuditEngine::new();
    let _ = audit.initialize_csv();

    settings::print_experiment_manifest(
        &args.symbol,
        &args.csv_file_path,
        args.generations,
        args.population,
    );

    info!("✅ {} adet tick RAM'de. Arena Hazır!", ticks.len());

    let mut population: Vec<Genome> = (0..args.population)
        .map(|_| create_random_genome())
        .collect();
    let mut best_all_time = population[0].clone();
    best_all_time.fitness = f64::NEG_INFINITY;
    let mut has_loaded_memory = false;

    if let Some(alpha_dna) = audit.load_best_genome() {
        if alpha_dna.weights.len() == 138 {
            info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
            population[0] = alpha_dna.clone();
            best_all_time = alpha_dna;
            has_loaded_memory = true;
        } else {
            warn!("⚠️ Eski nesil DNA tespit edildi. Yeni MLP mimarisi (138 Gen) için sıfırdan başlanıyor.");
        }
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = MUTATION_BASE_RATE;
    let mut gens_without_record = 0;

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
            gens_without_record = 0;
            current_mutation_rate = MUTATION_BASE_RATE;
            is_record = true;
            if !is_first_run_of_loaded {
                let _ = audit.save_record_break(&best_all_time);
            }
        } else {
            stagnation_counter += 1;
            gens_without_record += 1;

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

        if gens_without_record >= EARLY_STOP_LIMIT {
            warn!(
                "🛑 [EARLY STOPPING] {} jenerasyondur rekor kırılamadı.",
                EARLY_STOP_LIMIT
            );
            break;
        }

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

    info!("🏁 Eğitim Döngüsü Tamamlandı.");
    Ok(())
}
