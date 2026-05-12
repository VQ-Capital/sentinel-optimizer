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
#[command(author, version, about = "VQ-Capital V15.4 Omni-Revolution", long_about = None)]
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
    info!("🧬 VQ-CAPITAL V15.4 OMNI-REVOLUTION ENGINE BAŞLATILIYOR...");

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
    println!("⚙️ Fitness Rule  : Asymmetric Profit Reward & Multi-Point Extinction");
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
        let mut fixed_dna = alpha_dna.clone();
        if fixed_dna.weights.len() == 43 {
            fixed_dna.weights.push(0.50);
        }
        info!("🧠 Hafıza Geri Yüklendi. Alpha DNA sisteme aşılandı.");
        population[0] = fixed_dna.clone();
        best_all_time = fixed_dna;
        has_loaded_memory = true;
    }

    let mut stagnation_counter = 0;
    // 🔥 CERRAHİ: Mutasyon tabanı %15'e çıkarıldı. Genler çok daha hızlı değişecek.
    let mut current_mutation_rate = 0.15f32;

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
            current_mutation_rate = 0.15; // Rekor gelince normale dön
            is_record = true;

            if !is_first_run_of_loaded {
                let _ = audit.save_record_break(&best_all_time);
            }
        } else {
            stagnation_counter += 1;
            if stagnation_counter > 5 {
                // 🔥 CERRAHİ: Plato toleransı 5 nesile indirildi! Acımasız hız.
                warn!("🌋 [MASS EXTINCTION] Stagnation! %95 Nüfus imha ediliyor.");
                current_mutation_rate = 0.50; // %50 Multi-point kaotik mutasyon
            } else {
                current_mutation_rate = (current_mutation_rate + 0.05).min(0.40);
            }
        }

        let time_sec = start_time.elapsed().as_secs_f32();
        let _ = audit.log_generation(gen, gen_best, current_mutation_rate, time_sec, is_record);

        population = evolve_population(
            &population,
            args.population,
            current_mutation_rate,
            stagnation_counter > 5,
        );

        if stagnation_counter > 5 {
            stagnation_counter = 0;
        }
    }
    Ok(())
}

// 🛡️ V15.4 ASİMETRİK ÖDÜL FITNESS (The Kutsal Kâse)
fn calculate_fitness(
    pnl: f64,
    sharpe: f64,
    max_dd: f64,
    trades: usize,
    win_rate: f64,
    profit_factor: f64,
) -> f64 {
    // 1. ÖLÜM SINIRLARI
    if trades < 50 {
        return -2_000_000.0; // Keskin nişancılara af yok. En az 50 işlem!
    }
    if max_dd >= 40.0 {
        return -1_000_000.0; // Sermaye erimesi
    }

    let mut fitness = 0.0;

    // 2. İSTATİSTİKSEL GEÇERLİLİK
    if trades < 250 {
        fitness -= (250 - trades) as f64 * 500.0;
    } else if trades > 2500 {
        fitness -= (trades - 2500) as f64 * 20.0; // Overtrading komisyon israfı
    }

    // 🔥 3. KAZANMA ORANI (ASİMETRİK ÖDÜL)
    if win_rate >= 45.0 {
        fitness += 1_000_000.0; // Mükemmel WinRate Barajı Aşıldı = İlah Puanı
        fitness += (win_rate - 45.0) * 100_000.0;
    } else if win_rate < 30.0 {
        fitness -= (30.0 - win_rate) * 5000.0; // Çöp WinRate cezası
    } else {
        fitness += win_rate * 100.0; // Gelişim teşviki
    }

    // 🔥 4. KÂR ÇARPANI (PF KUTSAL KÂSESİ)
    if profit_factor >= 1.0 {
        fitness += 5_000_000.0; // KÂR EDEN STRATEJİ KRALLIĞA ÇIKAR!
        fitness += profit_factor * 500_000.0;
    } else {
        fitness -= (1.0 - profit_factor) * 10_000.0; // Kâr edemeyen ağır cezalandırılır
    }

    // 🔥 5. PNL ASİMETRİSİ
    if pnl > 0.0 {
        fitness += pnl * 500_000.0; // Kârın her doları paha biçilemez
    } else {
        fitness += pnl * 50.0; // Zarar edenin sadece cüzdanı eksilir (Hızlıca kâra dönmesi için hafif ceza)
    }

    // 6. RİSK VE GÜVENLİK
    if sharpe > 0.0 {
        fitness += sharpe * 5000.0;
    }
    fitness -= max_dd * 1000.0;

    fitness
}

fn create_random_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = Vec::with_capacity(44);

    for _ in 0..36 {
        dna.push(rng.gen_range(-1.0..1.0));
    }

    dna.push(rng.gen_range(-0.5..0.2)); // Hold Bias (İşlem yapmaya itmek için negatif)
    dna.push(rng.gen_range(0.0..1.0)); // Buy Bias
    dna.push(rng.gen_range(0.0..1.0)); // Sell Bias

    dna.push(rng.gen_range(0.005..0.040)); // TP
    dna.push(rng.gen_range(0.003..0.020)); // SL
    dna.push(rng.gen_range(500.0..2000.0)); // Cooldown (Daha hızlı işlem açsın)
    dna.push(rng.gen_range(0.01..0.05)); // Risk

    // Confidence (0.40 - 0.65) => Tetiğe aşırı rahat basacak, korkaklık yok!
    dna.push(rng.gen_range(0.40..0.65));

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

    // 🔥 GERÇEK YOK OLUŞ: Kilitlenme varsa sadece EN İYİ 1 KİŞİ yaşar, Elitler bile imha edilir.
    let elite_count = if is_cataclysm {
        1
    } else {
        (total_size / 20).max(2)
    };
    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    // Katliam varsa Nüfusun %95'i uzaydan (rastgele) gelir!
    let random_injection = if is_cataclysm {
        (total_size as f32 * 0.95) as usize
    } else {
        (total_size as f32 * 0.15) as usize
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

            // Sign Flip Mutation (Ters İşlem Hatasını Çözmek İçin %5 İhtimal)
            if i < 39 && rng.gen_bool(0.05) {
                gene = -gene;
            }

            if rng.gen_bool(mut_rate as f64) {
                // Multi-Point Genlik: Mutasyon daha sert!
                let severity = if is_cataclysm { 2.0 } else { 1.0 };

                if i < 36 {
                    gene += rng.gen_range(-0.5..0.5) * severity;
                } else if (36..39).contains(&i) {
                    gene += rng.gen_range(-0.2..0.2) * severity;
                } else if i == 39 || i == 40 {
                    gene += rng.gen_range(-0.002..0.002) * severity;
                } else if i == 41 {
                    gene += rng.gen_range(-200.0..200.0) * severity;
                } else if i == 43 {
                    gene += rng.gen_range(-0.03..0.03) * severity;
                } else {
                    gene += rng.gen_range(-0.01..0.01);
                }
            }

            // Clamp (Sınırlar)
            if i == 39 {
                gene = gene.clamp(0.005, 0.05);
            } else if i == 40 {
                gene = gene.clamp(0.003, 0.03);
            } else if i == 41 {
                gene = gene.clamp(100.0, 30000.0);
            } else if i == 42 {
                gene = gene.clamp(0.01, 0.05);
            } else if i == 43 {
                gene = gene.clamp(0.40, 0.70); // 🔥 Sınır İndi: Aşırı emin olup bekleyemez!
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
