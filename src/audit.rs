// ========== DOSYA: sentinel-optimizer/src/audit.rs ==========
use crate::Genome;
use anyhow::{Context, Result};
use chrono::Utc;
use std::fs::{self, OpenOptions};
use std::io::Write;

pub struct AuditEngine {
    pub log_path: String,
    pub hof_path: String,
}

impl AuditEngine {
    pub fn new() -> Self {
        Self {
            log_path: "optimization_audit_log.csv".to_string(),
            hof_path: "hall_of_fame.json".to_string(),
        }
    }

    /// Hafızadaki en iyi genomu yükler (Alpha DNA enjeksiyonu için)
    pub fn load_best_genome(&self) -> Option<Genome> {
        if let Ok(data) = fs::read_to_string(&self.hof_path) {
            if let Ok(history) = serde_json::from_str::<Vec<Genome>>(&data) {
                return history.first().cloned();
            }
        }
        None
    }

    /// Yeni bir rekor kırıldığında Hall of Fame'i günceller
    pub fn save_record_break(&self, genome: &Genome) -> Result<()> {
        let data = serde_json::to_string_pretty(&vec![genome])?;
        fs::write(&self.hof_path, data).context("JSON yazma hatası")?;
        Ok(())
    }

    /// Her jenerasyonun sonucunu CSV'ye damgalar (Append Mode)
    pub fn log_generation(&self, gen: usize, best: &Genome, mutation_rate: f32) -> Result<()> {
        let file_exists = std::path::Path::new(&self.log_path).exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        if !file_exists || fs::metadata(&self.log_path)?.len() == 0 {
            writeln!(
                file,
                "Timestamp,Gen,PnL,Sharpe,MaxDD,Trades,Fit,MutRate,TP,SL,Risk,Cooldown"
            )?;
        }

        writeln!(
            file,
            "{},{},{:.2},{:.2},{:.2},{},{:.2},{:.2},{:.4},{:.4},{:.2},{:.0}",
            Utc::now().to_rfc3339(),
            gen,
            best.pnl,
            best.sharpe,
            best.max_drawdown,
            best.trades,
            best.fitness,
            mutation_rate,
            best.weights[39], // TP
            best.weights[40], // SL
            best.weights[42], // Risk
            best.weights[41], // Cooldown
        )?;
        Ok(())
    }

    /// Konsola kurumsal formatta ilerleme basar
    pub fn print_progress(&self, gen: usize, best: &Genome, is_record: bool, duration: f32) {
        let prefix = if is_record {
            "🌟 REKOR!"
        } else {
            "🔄 İlerleme"
        };
        println!(
            "{} [Gen {}] PnL: ${:>8.2} | Trades: {:>5} | Fit: {:>10.2} | TP: {:.3} | SL: {:.3} | Risk: %{:<2.0} | Süre: {:.2}s",
            prefix,
            gen,
            best.pnl,
            best.trades,
            best.fitness,
            best.weights[39],
            best.weights[40],
            best.weights[42] * 100.0,
            duration
        );
    }
}
