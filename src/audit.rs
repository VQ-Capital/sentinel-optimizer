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

    /// Her jenerasyonun sonucunu CSV'ye damgalar
    pub fn log_generation(&self, gen: usize, best: &Genome, mutation_rate: f32) -> Result<()> {
        // 🔥 CERRAHİ: Gen 1 ise dosyayı sıfırla (truncate), değilse sonuna ekle (append).
        // Böylece kullanıcı asla manuel olarak CSV silmek zorunda kalmaz.
        let is_first_gen = gen == 1;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(is_first_gen)
            .append(!is_first_gen)
            .open(&self.log_path)?;

        if is_first_gen {
            writeln!(
                file,
                "Timestamp,Gen,PnL,Sharpe,MaxDD,Trades,Fit,MutRate,TP,SL,Risk,Cooldown"
            )?;
        }

        // 🔥 HFT HASSASİYETİ: Tüm metrikler 6 ondalığa (6 decimals) kilitlendi. JSON ile BİREBİR aynı olacak.
        writeln!(
            file,
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.2},{:.6},{:.6},{:.6},{:.0}",
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
            "🌟 REKOR!  "
        } else {
            "🔄 İlerleme"
        };
        // 🔥 HFT HASSASİYETİ: Ekrana basılan değerler de 6 ondalığa çekildi. CSV ile BİREBİR uyumlu.
        println!(
            "{} [Gen {:>3}] PnL: ${:>9.4} | Trades: {:>4} | Fit: {:>10.4} | TP: {:.6} | SL: {:.6} | Risk: {:.6} | Süre: {:.2}s",
            prefix,
            gen,
            best.pnl,
            best.trades,
            best.fitness,
            best.weights[39],
            best.weights[40],
            best.weights[42],
            duration
        );
    }
}
