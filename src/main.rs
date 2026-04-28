// ========== DOSYA: sentinel-optimizer/src/main.rs ==========
use anyhow::{Context, Result};
use clap::Parser;
use rand::Rng;
use regex::Regex;
use std::fs;
use std::process::Command;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about = "VQ-Capital Singularity Optimizer", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "../sentinel-inference/src/main.rs")]
    inference_file: String,

    #[arg(short, long, default_value = "../data/test_data.csv")]
    csv_data: String,

    #[arg(short, long, default_value = "3")]
    generations: usize,

    #[arg(short, long, default_value = "5")]
    population: usize,
}

#[derive(Clone, Debug)]
struct Genome {
    weights: Vec<f32>,
    fitness: f64,
    pnl: f64,
    sharpe: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🧬 VQ-CAPITAL SINGULARITY ENGINE BAŞLATILIYOR...");

    let args = Args::parse();
    let http_client = reqwest::Client::new();

    // 1. Başlangıç Ağırlıklarını (Gen-0) Belirle
    let mut best_genome = Genome {
        weights: vec![
            0.0, 0.5, -0.5, 0.0, 0.4, -0.4, 0.0, 0.3, -0.3, 0.0, -0.1, 0.1, 0.0, -0.2, 0.2, 0.1,
            -0.1, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, -0.2, 0.2, 0.0, 0.1, -0.1, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ],
        fitness: -9999.0,
        pnl: 0.0,
        sharpe: 0.0,
    };

    info!(
        "🚀 Jenerasyon Döngüsü Başlıyor (Gen: {}, Pop: {})",
        args.generations, args.population
    );

    for gen in 1..=args.generations {
        info!("==================================================");
        info!("🧬 JENERASYON {}/{}", gen, args.generations);
        info!("==================================================");

        let mut generation_best = best_genome.clone();

        for pop in 1..=args.population {
            info!("🔬 [Gen:{}-Pop:{}] Birey Test Ediliyor...", gen, pop);

            // A. Mutasyon (Mutation)
            let current_weights = if pop == 1 {
                best_genome.weights.clone() // İlk birey her zaman önceki neslin en iyisidir (Elitizm)
            } else {
                mutate(&best_genome.weights, 0.15)
            };

            // B. Koda Enjekte Et (Hard-Code)
            inject_weights(&args.inference_file, &current_weights)?;

            // C. Inference'ı Derle
            info!("   ⚙️ Sentinel-Inference Derleniyor...");
            execute_command("../sentinel-inference", "cargo", &["build", "--release"])?;

            // D. Docker'ı Yeniden Başlat (Yeni binary'i alsın)
            info!("   🐳 Konteyner Yeniden Başlatılıyor...");
            execute_command(
                "../sentinel-infra",
                "docker",
                &["restart", "sentinel-sentinel-inference-1"],
            )?;
            sleep(Duration::from_secs(3)).await; // NATS'a bağlanması için bekle

            // E. QuestDB'yi Temizle (Yeni simülasyon için)
            info!("   🧹 QuestDB Geçmişi Temizleniyor...");
            truncate_db(&http_client).await?;

            // F. Zaman Makinesi (Backtest) Çalıştır
            info!("   ⏳ Backtest Enjeksiyonu Başladı...");
            // 🔥 CERRAHİ: NATS Portu 14222 olarak Host makineye uygun ayarlandı
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
                ],
            )?;
            sleep(Duration::from_secs(2)).await; // İşlemlerin QuestDB'ye yazılması için buffer

            // G. Tearsheet Raporu Üret
            info!("   📄 Tearsheet Çıkarılıyor...");
            execute_command("../sentinel-tearsheet", "cargo", &["run", "--release"])?;

            // H. Hayatta Kalma (Fitness) Ölçümü
            let (pnl, sharpe) = parse_tearsheet("../sentinel-tearsheet/TEARSHEET.md")?;

            // Fitness Fonksiyonu (Sharpe ve PnL'i birleştiren özel formül)
            let fitness = if pnl > 0.0 && sharpe > 0.0 {
                pnl * sharpe
            } else {
                pnl + (sharpe * 100.0)
            };

            info!(
                "   📊 Sonuç -> PnL: ${:.2} | Sharpe: {:.2} | Fitness: {:.2}",
                pnl, sharpe, fitness
            );

            if fitness > generation_best.fitness {
                generation_best = Genome {
                    weights: current_weights,
                    fitness,
                    pnl,
                    sharpe,
                };
                info!("   🌟 YENİ LİDER BULUNDU!");
            }
        }

        best_genome = generation_best;
        info!(
            "🏆 Jenerasyon {} Lideri -> PnL: ${:.2}, Sharpe: {:.2}",
            gen, best_genome.pnl, best_genome.sharpe
        );
    }

    info!("==================================================");
    info!("🦅 SINGULARITY ULAŞILDI. EN İYİ GENOM KALIÇLAŞTIRILIYOR.");
    info!("==================================================");

    // Final Enjeksiyon ve Deploy
    inject_weights(&args.inference_file, &best_genome.weights)?;
    execute_command("../sentinel-inference", "cargo", &["build", "--release"])?;
    execute_command(
        "../sentinel-infra",
        "docker",
        &["restart", "sentinel-sentinel-inference-1"],
    )?;

    info!("✅ Otonom Optimizasyon Tamamlandı. Sistem maksimum kârlılığa kilitlendi!");
    Ok(())
}

fn mutate(base: &[f32], mutation_rate: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    base.iter()
        .map(|&w| {
            if rng.gen_bool(0.35) {
                // %35 Mutasyon Şansı
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

    let mut new_vec = String::from("let weights_data = vec![\n");
    for i in 0..12 {
        new_vec.push_str(&format!(
            "             {:.4},  {:.4},  {:.4},\n",
            weights[i * 3],
            weights[i * 3 + 1],
            weights[i * 3 + 2]
        ));
    }
    new_vec.push_str("        ];");

    let new_content = re.replace(&content, new_vec.as_str());
    fs::write(file_path, new_content.to_string())?;
    Ok(())
}

fn execute_command(dir: &str, cmd: &str, args: &[&str]) -> Result<()> {
    // 🔥 CERRAHİ: Artık Stdio::null() (Gizleme) yok. HFT akışını ekranda canlı göreceksin.
    let status = Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .status()
        .context(format!("Komut çalıştırılamadı: {} {:?}", cmd, args))?;

    if !status.success() {
        warn!("⚠️ Alt komut başarısız oldu: {} {:?}", cmd, args);
    }
    Ok(())
}

async fn truncate_db(client: &reqwest::Client) -> Result<()> {
    let queries = [
        "TRUNCATE TABLE paper_trades;",
        "TRUNCATE TABLE performance;",
    ];

    for query in queries {
        let _ = client
            .get("http://localhost:19000/exec")
            .query(&[("query", query)])
            .send()
            .await?;
    }
    Ok(())
}

fn parse_tearsheet(file_path: &str) -> Result<(f64, f64)> {
    let content = fs::read_to_string(file_path).unwrap_or_default();

    let pnl_re = Regex::new(r"Net PnL\*\* \| `\$([\-\d\.]+)`")?;
    let sharpe_re = Regex::new(r"Sharpe Ratio\*\* \| `([\-\d\.]+)`")?;

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
