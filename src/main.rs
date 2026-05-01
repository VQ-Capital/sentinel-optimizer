// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process::Command;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital Singularity Optimizer V8", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "../sentinel-inference/src/weights.rs")]
    inference_file: String,

    #[arg(
        short,
        long,
        default_value = "../sentinel-data/datasets/BTCUSDT_1D.csv"
    )]
    csv_file_path: String,

    #[arg(short, long, default_value = "BTCUSDT")]
    symbol: String,

    #[arg(short, long, default_value = "nats://localhost:14222")]
    nats_url: String,

    #[arg(short, long, default_value = "5000")]
    max_mps: usize,

    #[arg(short, long, default_value = "5")]
    generations: usize,

    #[arg(short, long, default_value = "5")]
    population: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Genome {
    weights: Vec<f32>, // 39 Gen: 36 Ağırlık + 3 Bias
    fitness: f64,
    pnl: f64,
    sharpe: f64,
    generation: usize,
    population_id: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧬 VQ-CAPITAL SINGULARITY ENGINE V8.1 (Self-Calibrating Edition) BAŞLATILIYOR...");

    let args = Args::parse();
    let http_client = reqwest::Client::new();

    let log_path = "optimization_audit_log.csv";
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .context("Log hatası")?;

    if log_file.metadata()?.len() == 0 {
        writeln!(
            log_file,
            "Timestamp,Generation,Population,PnL,Sharpe,Fitness,Status"
        )?;
    }

    let mut hall_of_fame: Vec<Genome> = Vec::new();

    let mut best_genome = if let Ok(data) = fs::read_to_string("hall_of_fame.json") {
        if let Ok(mut history) = serde_json::from_str::<Vec<Genome>>(&data) {
            if let Some(last_best) = history.pop() {
                info!("🧠 Geri Yükleme Başarılı! Önceki DNA'dan devam ediliyor.");
                let mut migrated_weights = last_best.weights.clone();
                if migrated_weights.len() == 36 {
                    migrated_weights.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
                hall_of_fame = history;

                Genome {
                    weights: migrated_weights,
                    fitness: last_best.fitness,
                    pnl: last_best.pnl,
                    sharpe: last_best.sharpe,
                    generation: 0,
                    population_id: 0,
                }
            } else {
                default_genome()
            }
        } else {
            default_genome()
        }
    } else {
        default_genome()
    };

    for gen in 1..=args.generations {
        info!("==================================================");
        info!("🧬 JENERASYON {}/{} EVRİM DÖNGÜSÜ", gen, args.generations);
        info!("==================================================");

        let mut generation_best = best_genome.clone();

        for pop in 1..=args.population {
            info!("🔬 [G:{}-P:{}] Test Ediliyor...", gen, pop);

            let current_weights = if gen == 1 && pop == 1 {
                best_genome.weights.clone()
            } else {
                mutate(&best_genome.weights, 0.40)
            };

            inject_weights(&args.inference_file, &current_weights)?;

            info!("   ⚙️ Sentinel-Inference Derleniyor...");
            execute_command("../sentinel-inference", "cargo", &["build", "--release"])?;

            info!("   🐳 Konteyner Yeniden Başlatılıyor...");
            execute_command(
                "../sentinel-infra",
                "docker",
                &["restart", "sentinel-sentinel-inference-1"],
            )?;
            sleep(Duration::from_secs(6)).await;

            info!("   🧹 Tüm Hafıza Sıfırlanıyor...");
            truncate_db(&http_client).await?;

            info!("   ⏳ Backtest Enjeksiyonu Başladı...");
            execute_command(
                "../sentinel-backtest",
                "cargo",
                &[
                    "run",
                    "--release",
                    "--",
                    "--csv-file-path",
                    &args.csv_file_path,
                    "--symbol",
                    &args.symbol,
                    "--nats-url",
                    &args.nats_url,
                    "--max-mps",
                    &args.max_mps.to_string(),
                ],
            )?;

            sleep(Duration::from_secs(12)).await;

            info!("   📄 Tearsheet Çıkarılıyor...");
            execute_command("../sentinel-tearsheet", "cargo", &["run", "--release"])?;

            let (pnl, sharpe) = parse_tearsheet("../sentinel-tearsheet/TEARSHEET.md")?;

            let fitness = if pnl == 0.0 && sharpe == 0.0 {
                -500.0 // Tembellik cezası
            } else if pnl > 0.0 {
                pnl * (1.0 + sharpe)
            } else {
                pnl - 2.0
            };

            // 🔥 CERRAHİ: DATASET SHIFT (KAYMA) KALİBRASYONU
            let mut is_baseline = false;
            if gen == 1 && pop == 1 {
                generation_best.fitness = -999999.0; // Eski datasetin sahte skorunu ez!
                is_baseline = true;
            }

            let status = if is_baseline {
                "📍 BASELINE"
            } else if fitness > generation_best.fitness {
                "🌟 IMPROVED"
            } else {
                "❌ REJECTED"
            };

            writeln!(
                log_file,
                "{},{},{},{:.4},{:.2},{:.4},{}",
                Utc::now().to_rfc3339(),
                gen,
                pop,
                pnl,
                sharpe,
                fitness,
                status
            )?;

            info!(
                "   📊 Sonuç -> PnL: ${:.4} | Sharpe: {:.2} | Fitness: {:.2} | {}",
                pnl, sharpe, fitness, status
            );

            if fitness > generation_best.fitness || is_baseline {
                generation_best = Genome {
                    weights: current_weights,
                    fitness,
                    pnl,
                    sharpe,
                    generation: gen,
                    population_id: pop,
                };
                if is_baseline {
                    info!(
                        "   📍 YENİ DATASET İÇİN BAZ ALINAN REFERANS SKOR: {:.4}",
                        fitness
                    );
                } else {
                    info!("   🌟 YENİ REKOR BULUNDU!");
                }
            }
        }
        best_genome = generation_best;
        hall_of_fame.push(best_genome.clone());
        fs::write(
            "hall_of_fame.json",
            serde_json::to_string_pretty(&hall_of_fame)?,
        )?;
    }

    info!("🏁 EVRİM TAMAMLANDI. En kârlı DNA Beyne enjekte edildi.");
    inject_weights(&args.inference_file, &best_genome.weights)?;
    execute_command("../sentinel-inference", "cargo", &["build", "--release"])?;
    execute_command(
        "../sentinel-infra",
        "docker",
        &["restart", "sentinel-sentinel-inference-1"],
    )?;
    Ok(())
}

fn default_genome() -> Genome {
    let mut rng = rand::thread_rng();
    let mut dna: Vec<f32> = (0..36).map(|_| rng.gen_range(-0.1..0.1)).collect();
    dna.extend_from_slice(&[0.0, 0.0, 0.0]);
    Genome {
        weights: dna,
        fitness: -9999.0,
        pnl: 0.0,
        sharpe: 0.0,
        generation: 0,
        population_id: 0,
    }
}

fn mutate(base: &[f32], mutation_rate: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    base.iter()
        .enumerate()
        .map(|(i, &w)| {
            let current_rate = if i >= 36 {
                mutation_rate / 2.0
            } else {
                mutation_rate
            };
            if rng.gen_bool(0.50) {
                w + rng.gen_range(-current_rate..current_rate)
            } else {
                w
            }
        })
        .collect()
}

fn inject_weights(file_path: &str, dna: &[f32]) -> Result<()> {
    let content = fs::read_to_string(file_path).context("Inference dosyası bulunamadı")?;

    let re_weights =
        Regex::new(r"pub fn get_dna_weights\(\) -> Vec<f32> \{\s*vec\!\[[\s\S]*?\]\s*\}")?;
    let mut new_weights = String::from(
        "pub fn get_dna_weights() -> Vec<f32> {\n    vec![\n        //  HOLD,    BUY,    SELL\n",
    );
    let labels = [
        "Price Velocity (Z-Score)",
        "Orderbook Imbalance",
        "Neural Sentiment",
        "Chain Urgency",
        "RSI",
        "Volatility",
        "Taker Ratio",
        "Intensity (Tick count)",
        "Position in Range",
        "Orderbook Depth",
        "Time Sine (Intraday)",
        "Last Close Price",
    ];
    for i in 0..12 {
        new_weights.push_str(&format!(
            "         {:.4},  {:.4},  {:.4}, // F{}: {}\n",
            dna[i * 3],
            dna[i * 3 + 1],
            dna[i * 3 + 2],
            i,
            labels[i]
        ));
    }
    new_weights.push_str("    ]\n}");
    let content_w = re_weights.replace(&content, new_weights.as_str());

    let re_biases =
        Regex::new(r"pub fn get_dna_biases\(\) -> Vec<f32> \{\s*vec\!\[[\s\S]*?\]\s*\}")?;
    let mut new_biases = String::from(
        "pub fn get_dna_biases() -> Vec<f32> {\n    vec![\n        // HOLD, BUY, SELL\n",
    );
    new_biases.push_str(&format!(
        "        {:.4}, {:.4}, {:.4}, \n",
        dna[36], dna[37], dna[38]
    ));
    new_biases.push_str("    ]\n}");

    let final_content = re_biases.replace(&content_w, new_biases.as_str());

    fs::write(file_path, final_content.to_string())?;
    Ok(())
}

fn execute_command(dir: &str, cmd: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(cmd).args(args).current_dir(dir).status()?;
    if !status.success() {
        warn!("⚠️ Alt komut başarısız: {} {:?}", cmd, args);
    }
    Ok(())
}

async fn truncate_db(client: &reqwest::Client) -> Result<()> {
    let queries = [
        "TRUNCATE TABLE paper_trades;",
        "TRUNCATE TABLE performance;",
        "TRUNCATE TABLE market_states;",
        "TRUNCATE TABLE execution_rejections;",
    ];
    for query in queries {
        let _ = client
            .get("http://localhost:19000/exec")
            .query(&[("query", query)])
            .send()
            .await?;
    }
    let _ = client
        .post("http://localhost:16333/collections/market_states_12d/points/delete")
        .json(&json!({ "filter": { "must": [] } }))
        .send()
        .await?;
    Ok(())
}

fn parse_tearsheet(file_path: &str) -> Result<(f64, f64)> {
    let content = fs::read_to_string(file_path).unwrap_or_default();
    let pnl_re = Regex::new(r"Net PnL\*\* \| `\$([\-\d\.]+)`").unwrap();
    let sharpe_re = Regex::new(r"Sharpe Ratio\*\* \| `([\-\d\.]+)`").unwrap();

    let pnl = pnl_re
        .captures(&content)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<f64>().ok())
        .unwrap_or(0.0);
    let sharpe = sharpe_re
        .captures(&content)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<f64>().ok())
        .unwrap_or(0.0);
    Ok((pnl, sharpe))
}
