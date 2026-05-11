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
#[command(author, version, about = "VQ-Capital V16.0 Apex Predator", long_about = None)]
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
    info!("🧬 VQ-CAPITAL V16.0 APEX PREDATOR ENGINE BAŞLATILIYOR...");

    let args = Args::parse();
    let audit = AuditEngine::new();

    let _ = audit.initialize_csv();

    println!("\n=======================================================");
    println!("🧪 EXPERIMENT CONFIGURATION (DENEY PARAMETRELERİ)");
    println!("=======================================================");
    println!("📍 Target Symbol : {}", args.symbol);
    println!("📂 Dataset Path  : {}", args.csv_file_path);
    println!("🧬 Generations   : {}", args.generations);
    println!("👥 Population    : {}", args.population);
    println!("⚙️ Fitness Rule  : 44D Genome (Dynamic Trigger) & Capitalist Reward");
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
        // 🔥 CERRAHİ: Geriye Dönük Uyumluluk (43 Genli modeli 44'e tamamlar)
        let mut fixed_dna = alpha_dna.clone();
        if fixed_dna.weights.len() == 43 {
            fixed_dna.weights.push(0.50); // Varsayılan Güven Sınırı
        }
        info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
        population[0] = fixed_dna.clone();
        best_all_time = fixed_dna;
        has_loaded_memory = true;
    }

    let mut stagnation_counter = 0;
    let mut current_mutation_rate = 0.08f32;

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

            // 🔥 KAPİTALİST FİTNESS (Kârı Yüceltir)
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

        // "Fark edilir" bir rekor kırmasını istiyoruz (Virgülden sonrasıyla bizi kandırmasın)
        if gen_best.fitness > (best_all_time.fitness + 0.1) || is_first_run_of_loaded {
            best_all_time = gen_best.clone();
            stagnation_counter = 0;
            current_mutation_rate = 0.08;
            is_record = true;

            if !is_first_run_of_loaded {
                let _ = audit.save_record_break(&best_all_time);
            }
        } else {
            stagnation_counter += 1;
            if stagnation_counter > 10 {
                warn!("🌋 [MASS EXTINCTION] Gözlemci 3 Tavsiyesi: %90 Uzaylı İstiklası!");
                current_mutation_rate = 0.40;
            } else {
                current_mutation_rate = (current_mutation_rate + 0.03).min(0.25);
            }
        }

        let time_sec = start_time.elapsed().as_secs_f32();
        let _ = audit.log_generation(gen, gen_best, current_mutation_rate, time_sec, is_record);

        population = evolve_population(
            &population,
            args.population,
            current_mutation_rate,
            stagnation_counter > 10,
        );

        if stagnation_counter > 10 {
            stagnation_counter = 0;
        }
    }
    Ok(())
}

// 🛡️ YENİ KAPİTALİST FİTNESS (Kâr ve Başarı Odaklı)
fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    win_rate: f64,
    profit_factor: f64,
) -> f64 {
    if trades < 10 {
        return -1_000_000.0;
    }
    if max_dd >= 50.0 {
        return -500_000.0;
    }

    let mut fitness = 0.0;

    // 1. PNL ANA HEDEFTİR
    if pnl > 0.0 {
        fitness += pnl * 10000.0; // Kâr ediyorsan kralsın
    } else {
        fitness += pnl * 10.0; // Zarar ediyorsan sadece puanın düşer (Katı ceza yok)
    }

    // 2. KALİTE GÖSTERGELERİ
    fitness += win_rate * 100.0;

    if profit_factor > 1.0 {
        fitness += profit_factor * 5000.0;
    } else {
        fitness -= (1.0 - profit_factor) * 1000.0;
    }

    if sharpe > 0.0 {
        fitness += sharpe * 500.0;
    }

    // 3. KISITLAMALAR
    fitness -= max_dd * 100.0;
    if trades > 2000 {
        fitness -= (trades - 2000) as f64 * 5.0;
    }

    fitness
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(44); // 🔥 44 GEN!

    for _ in 0..36 {
        dna.push(rng.gen_range(-2.0..2.0));
    }

    dna.push(rng.gen_range(-0.5..0.5)); // Hold Bias
    dna.push(rng.gen_range(-1.0..1.0)); // Buy Bias
    dna.push(rng.gen_range(-1.0..1.0)); // Sell Bias

    dna.push(rng.gen_range(0.005..0.040)); // TP
    dna.push(rng.gen_range(0.003..0.020)); // SL
    dna.push(rng.gen_range(1000.0..15000.0)); // Cooldown
    dna.push(rng.gen_range(0.01..0.05)); // Risk

    // 🔥 YENİ 44. GEN: Confidence Threshold (AI ne kadar emin olursa tetiği çeker?)
    dna.push(rng.gen_range(0.40..0.90));

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

    // 🔥 GÖZLEMCİ 3 TAVSİYESİ: GERÇEK YOK OLUŞ!
    // Cataclysm durumunda nüfusun %90'ı tamamen SİLİNİR ve uzaylılar gelir!
    let random_injection = if is_cataclysm {
        (total_size as f32 * 0.9) as usize
    } else {
        (total_size as f32 * 0.1) as usize
    };

    while new_pop.len() < total_size - random_injection {
        let p1 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let p2 = &current_pop[rng.gen_range(0..elite_count * 2)];

        let mut child_dna = Vec::with_capacity(44);
        for i in 0..44 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };

            // 🔥 SIGN FLIP (İŞARET TERSİNE ÇEVİRME) MUTASYONU
            if i < 39 && rng.gen_bool(0.02) {
                gene = -gene;
            }

            if rng.gen_bool(mut_rate as f64) {
                if i < 36 {
                    gene += rng.gen_range(-0.5..0.5);
                } else if (36..39).contains(&i) {
                    gene += rng.gen_range(-0.2..0.2);
                } else if i == 39 || i == 40 {
                    gene += rng.gen_range(-0.002..0.002);
                } else if i == 41 {
                    gene += rng.gen_range(-500.0..500.0);
                } else if i == 43 {
                    gene += rng.gen_range(-0.05..0.05); // Confidence Mutation
                } else {
                    gene += rng.gen_range(-0.01..0.01);
                }
            }

            if i == 39 {
                gene = gene.clamp(0.005, 0.05);
            } else if i == 40 {
                gene = gene.clamp(0.003, 0.03);
            } else if i == 41 {
                gene = gene.clamp(500.0, 30000.0);
            } else if i == 42 {
                gene = gene.clamp(0.01, 0.05);
            } else if i == 43 {
                gene = gene.clamp(0.40, 0.95); // 44. Gen Clamp
            } else if (36..39).contains(&i) {
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
