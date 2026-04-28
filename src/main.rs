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
use tracing::{info, warn}; // 🔥 CERRAHİ: Chrono importu eklendi

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital Singularity Optimizer V7", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "../sentinel-inference/src/main.rs")]
    inference_file: String,

    #[arg(short, long, default_value = "../sentinel-data/datasets/test_data.csv")]
    csv_data: String,

    #[arg(short, long, default_value = "5")]
    generations: usize,

    #[arg(short, long, default_value = "5")]
    population: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Genome {
    weights: Vec<f32>,
    fitness: f64,
    pnl: f64,
    sharpe: f64,
    generation: usize,
    population_id: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧬 VQ-CAPITAL SINGULARITY ENGINE V7 (Deep-Audit Edition) BAŞLATILIYOR...");

    let args = Args::parse();
    let http_client = reqwest::Client::new();

    // 1. Analiz Log Dosyasını Hazırla
    let log_path = "optimization_audit_log.csv";
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .context("Log dosyası oluşturulamadı")?;

    // Başlık satırı (Eğer dosya yeni oluşturulduysa)
    if log_file.metadata()?.len() == 0 {
        writeln!(
            log_file,
            "Timestamp,Generation,Population,PnL,Sharpe,Fitness,Status"
        )?;
    }

    let mut hall_of_fame: Vec<Genome> = Vec::new();

    // Başlangıç Katsayıları (Gen-0)
    let mut best_genome = Genome {
        weights: vec![
            0.0, 0.5, -0.5, 0.0, 0.4, -0.4, 0.0, 0.3, -0.3, 0.0, -0.1, 0.1, 0.0, -0.2, 0.2, 0.1,
            -0.1, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, -0.2, 0.2, 0.0, 0.1, -0.1, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ],
        fitness: -9999.0,
        pnl: 0.0,
        sharpe: 0.0,
        generation: 0,
        population_id: 0,
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
                mutate(&best_genome.weights, 0.35)
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
                    "--csv-file",
                    &args.csv_data,
                    "--nats-url",
                    "nats://localhost:14222",
                    "--max-mps",
                    "5000",
                ],
            )?;

            sleep(Duration::from_secs(12)).await;

            info!("   📄 Tearsheet Çıkarılıyor...");
            execute_command("../sentinel-tearsheet", "cargo", &["run", "--release"])?;

            let (pnl, sharpe) = parse_tearsheet("../sentinel-tearsheet/TEARSHEET.md")?;

            let fitness = if pnl > 0.0 {
                pnl * (1.0 + sharpe)
            } else {
                pnl - 2.0
            };

            let status = if fitness > generation_best.fitness {
                "🌟 IMPROVED"
            } else {
                "❌ REJECTED"
            };

            // 📊 DENETİM KAYDI: CSV'ye Yaz
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
                "   📊 Sonuç -> PnL: ${:.4} | Sharpe: {:.2} | Status: {}",
                pnl, sharpe, status
            );

            if fitness > generation_best.fitness {
                generation_best = Genome {
                    weights: current_weights,
                    fitness,
                    pnl,
                    sharpe,
                    generation: gen,
                    population_id: pop,
                };
                info!("   🌟 YENİ REKOR BULUNDU!");
            }
        }

        best_genome = generation_best;
        hall_of_fame.push(best_genome.clone());

        // Hall of Fame Kaydet (Pretty Print)
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

fn mutate(base: &[f32], mutation_rate: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    base.iter()
        .map(|&w| {
            if rng.gen_bool(0.45) {
                w + rng.gen_range(-mutation_rate..mutation_rate)
            } else {
                w
            }
        })
        .collect()
}

fn inject_weights(file_path: &str, weights: &[f32]) -> Result<()> {
    let content = fs::read_to_string(file_path).context("Inference dosyası bulunamadı")?;
    let re = Regex::new(r"let weights_data = vec!\[[\s\S]*?\];")?;

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

    let mut new_vec =
        String::from("let weights_data = vec![\n            //  HOLD,    BUY,    SELL\n");
    for i in 0..12 {
        new_vec.push_str(&format!(
            "             {:.4},  {:.4},  {:.4}, // F{}: {}\n",
            weights[i * 3],
            weights[i * 3 + 1],
            weights[i * 3 + 2],
            i,
            labels[i]
        ));
    }
    new_vec.push_str("        ];");

    let new_content = re.replace(&content, new_vec.as_str());
    fs::write(file_path, new_content.to_string())?;
    Ok(())
}

fn execute_command(dir: &str, cmd: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .status()
        .context(format!("Komut hatası: {} {:?}", cmd, args))?;

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
