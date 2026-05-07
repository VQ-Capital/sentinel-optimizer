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

    /// 🔥 CERRAHİ: Her optimizer başladığında CSV'yi sıfırlar ve yeni başlıkları yazar.
    pub fn initialize_csv(&self) -> Result<()> {
        let mut file = fs::File::create(&self.log_path)?;
        writeln!(
            file,
            "Timestamp,Gen,PnL,Sharpe,MaxDD,Trades,Fit,MutRate,TP,SL,Risk,Cooldown,TimeSec"
        )?;
        Ok(())
    }

    pub fn load_best_genome(&self) -> Option<Genome> {
        if let Ok(data) = fs::read_to_string(&self.hof_path) {
            if let Ok(history) = serde_json::from_str::<Vec<Genome>>(&data) {
                return history.first().cloned();
            }
        }
        None
    }

    pub fn save_record_break(&self, genome: &Genome) -> Result<()> {
        let data = serde_json::to_string_pretty(&vec![genome])?;
        fs::write(&self.hof_path, data).context("JSON yazma hatası")?;
        Ok(())
    }

    /// Her jenerasyonun sonucunu CSV'ye damgalar
    pub fn log_generation(
        &self,
        gen: usize,
        best: &Genome,
        mut_rate: f32,
        time_sec: f32,
    ) -> Result<()> {
        let mut file = OpenOptions::new().append(true).open(&self.log_path)?;

        // 🔥 HFT HASSASİYETİ: Tüm metrikler 6 ondalığa (6 decimals) kilitlendi.
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.2},{:.6},{:.6},{:.6},{:.0},{:.2}",
            Utc::now().to_rfc3339(),
            gen,
            best.pnl,
            best.sharpe,
            best.max_drawdown,
            best.trades,
            best.fitness,
            mut_rate,
            best.weights[39], // TP
            best.weights[40], // SL
            best.weights[42], // Risk
            best.weights[41], // Cooldown
            time_sec
        )?;
        Ok(())
    }

    /// Konsola kurumsal formatta ilerleme basar (CSV formatıyla BİREBİR AYNI)
    pub fn print_progress(
        &self,
        gen: usize,
        best: &Genome,
        is_record: bool,
        mut_rate: f32,
        duration: f32,
    ) {
        let prefix = if is_record {
            "🌟 REKOR!  "
        } else {
            "🔄 İlerleme"
        };

        println!(
            "{} [Gen {:>3}] PnL: {:>10.6} | Sharpe: {:>10.6} | MaxDD: {:>8.6} | Trades: {:>4} | Fit: {:>10.6} | MutRate: {:.2} | TP: {:.6} | SL: {:.6} | Risk: {:.6} | Cooldown: {:>5.0} | Süre: {:.2}s",
            prefix,
            gen,
            best.pnl,
            best.sharpe,
            best.max_drawdown,
            best.trades,
            best.fitness,
            mut_rate,
            best.weights[39],
            best.weights[40],
            best.weights[42],
            best.weights[41],
            duration
        );
    }
}
