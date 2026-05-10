// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

mod audit;
mod simulator;
use audit::AuditEngine;
use simulator::{run_simulation, HistoricalTick};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital V14.8 Alpha Strike", long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "../sentinel-data/datasets/BTCUSDT_30D.csv"
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
    info!("🧬 VQ-CAPITAL V14.8 ALPHA-STRIKE ENGINE BAŞLATILIYOR...");

    let args = Args::parse();
    let audit = AuditEngine::new();

    let _ = audit.initialize_csv();

    // 🔥 SENİN FİKRİN: DIŞ DENETÇİLER İÇİN DENEY KONFİGÜRASYONU LOGLAMASI
    println!("\n=======================================================");
    println!("🧪 EXPERIMENT CONFIGURATION (DENEY PARAMETRELERİ)");
    println!("=======================================================");
    println!("📍 Target Symbol : {}", args.symbol);
    println!("📂 Dataset Path  : {}", args.csv_file_path);
    println!("🧬 Generations   : {}", args.generations);
    println!("👥 Population    : {}", args.population);
    println!("⚙️ Fitness Rule  : Strict Continuous Function (PF & WinRate Penalty)");
    println!("=======================================================\n");

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
        info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
        population[0] = alpha_dna.clone();
        best_all_time = alpha_dna;
        has_loaded_memory = true;
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = 0.05f32;

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

        if gen_best.fitness > best_all_time.fitness || is_first_run_of_loaded {
            best_all_time = gen_best.clone();
            stagnation_counter = 0;
            current_mutation_rate = 0.05;
            is_record = true;

            if !is_first_run_of_loaded {
                let _ = audit.save_record_break(&best_all_time);
            }
        } else {
            stagnation_counter += 1;
            if stagnation_counter > 15 {
                warn!("🌋 [CATACLYSM] Stagnation! Boosting Diversity.");
                current_mutation_rate = 0.35;
            } else {
                current_mutation_rate = (current_mutation_rate + 0.02).min(0.20);
            }
        }

        let time_sec = start_time.elapsed().as_secs_f32();

        // 🔥 CERRAHİ: Audit modülü artık tek seferde hem ekrana hem CSV'ye basıyor.
        let _ = audit.log_generation(gen, gen_best, current_mutation_rate, time_sec, is_record);

        population = evolve_population(
            &population,
            args.population,
            current_mutation_rate,
            stagnation_counter > 15,
        );
        if stagnation_counter > 15 {
            stagnation_counter = 0;
        }
    }
    Ok(())
}

// 🛡️ REWARD HACKING KORUMASI: SÜREKLİ FONKSİYON (Early Return Yasak)
fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    win_rate: f64,
    profit_factor: f64,
) -> f64 {
    let mut fitness = 0.0;

    // 1. Ana Hedef: Kâr (PnL)
    fitness += pnl * 100.0;

    // 2. Risk Cezası
    fitness -= max_dd * 1000.0;
    if max_dd >= 50.0 {
        fitness -= 50_000.0; // Ağır ölüm cezası
    }

    // 3. İşlem Hacmi (Tembellik veya Hiperaktivite Cezası)
    if trades < 50 {
        fitness -= (50 - trades) as f64 * 500.0; // Tembellik cezası (49 işlem yapıp kaçamaz!)
    } else if trades > 2000 {
        fitness -= (trades - 2000) as f64 * 10.0; // Hiperaktif komisyon israfı cezası
    }

    // 4. Kazanma Oranı (Win Rate) Cezası
    if trades > 0 && win_rate < 40.0 {
        fitness -= (40.0 - win_rate) * 500.0;
    }

    // 5. Profit Factor (Sistemin Gerçek Sağlığı)
    if trades > 0 {
        if profit_factor >= 1.0 {
            fitness += profit_factor * 2000.0; // Kârlı sistemlere büyük ödül
        } else {
            fitness -= (1.0 - profit_factor) * 5000.0; // Kumara ağır ceza
        }
    } else {
        fitness -= 20000.0; // Hiç işlem yapmayana net ceza
    }

    // 6. Sharpe Çarpanı (Sadece Kârdayken anlamlıdır)
    if pnl > 0.0 && sharpe > 0.0 {
        fitness += sharpe * 500.0;
    }

    fitness
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(43);

    for _ in 0..36 {
        dna.push(rng.gen_range(-1.0..1.0));
    }

    dna.push(rng.gen_range(0.2..0.8));
    dna.push(rng.gen_range(-0.1..0.1));
    dna.push(rng.gen_range(-0.1..0.1));

    dna.push(rng.gen_range(0.005..0.040)); // TP: Genişletildi
    dna.push(rng.gen_range(0.003..0.020)); // SL: Genişletildi
    dna.push(rng.gen_range(1000.0..15000.0)); // Cooldown
    dna.push(rng.gen_range(0.01..0.05)); // Risk

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
        1
    } else {
        (total_size / 20).max(2)
    };
    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    let random_injection = if is_cataclysm { total_size / 2 } else { 0 };

    while new_pop.len() < total_size - random_injection {
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
                if i < 36 {
                    gene += rng.gen_range(-0.2..0.2); // 🔥 Mutasyon genliği artırıldı
                } else if (36..39).contains(&i) {
                    gene += rng.gen_range(-0.05..0.05);
                } else if i == 39 || i == 40 {
                    gene += rng.gen_range(-0.001..0.001);
                } else if i == 41 {
                    gene += rng.gen_range(-250.0..250.0);
                } else {
                    gene += rng.gen_range(-0.01..0.01);
                }
            }

            if i == 39 {
                gene = gene.clamp(0.005, 0.05);
            } else if i == 40 {
                gene = gene.clamp(0.003, 0.03);
            } else if i == 41 {
                gene = gene.clamp(1000.0, 30000.0);
            } else if i == 42 {
                gene = gene.clamp(0.01, 0.05);
            } else if (36..39).contains(&i) {
                gene = gene.clamp(-0.5, 0.5);
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
