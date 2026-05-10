// ========== DOSYA: sentinel-optimizer/src/audit.rs ==========
use crate::Genome;
use anyhow::{Context, Result};
use chrono::Utc;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;

/// Hem CSV hem de Konsol çıktıları için "Tek Doğruluk Kaynağı" (Single Source of Truth)
pub struct AuditRecord {
    pub timestamp: String,
    pub gen: usize,
    pub pnl: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub trades: usize,
    pub fitness: f64,
    pub mut_rate: f32,
    pub tp: f32,       // 🔥 CERRAHİ: f64 -> f32 (Rust Type Safety)
    pub sl: f32,       // 🔥 CERRAHİ: f64 -> f32 (Rust Type Safety)
    pub risk: f32,     // 🔥 CERRAHİ: f64 -> f32 (Rust Type Safety)
    pub cooldown: f32, // 🔥 CERRAHİ: f64 -> f32 (Rust Type Safety)
    pub time_sec: f32,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub is_record: bool,
}

impl AuditRecord {
    /// CSV dosyasının en üstündeki başlık satırı
    pub const CSV_HEADER: &'static str =
        "Timestamp,Gen,PnL,Sharpe,MaxDD,Trades,Fit,MutRate,TP,SL,Risk,Cooldown,TimeSec,WinRate,PF";

    /// Mevcut Genome ve optimizasyon verilerinden standart bir kayıt oluşturur
    pub fn from_genome(
        gen: usize,
        best: &Genome,
        mut_rate: f32,
        time_sec: f32,
        is_record: bool,
    ) -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            gen,
            pnl: best.pnl,
            sharpe: best.sharpe,
            max_drawdown: best.max_drawdown,
            trades: best.trades,
            fitness: best.fitness,
            mut_rate,
            // Genome içindeki weight indexlerini burada isimlendiriyoruz (Magic Number koruması)
            tp: best.weights[39],
            sl: best.weights[40],
            cooldown: best.weights[41],
            risk: best.weights[42],
            time_sec,
            win_rate: best.win_rate,
            profit_factor: best.profit_factor,
            is_record,
        }
    }

    /// Kaydı CSV formatında bir satıra dönüştürür
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.6},{:.6},{:.6},{},{:.6},{:.2},{:.6},{:.6},{:.6},{:.0},{:.2},{:.2},{:.4}",
            self.timestamp,
            self.gen,
            self.pnl,
            self.sharpe,
            self.max_drawdown,
            self.trades,
            self.fitness,
            self.mut_rate,
            self.tp,
            self.sl,
            self.risk,
            self.cooldown,
            self.time_sec,
            self.win_rate,
            self.profit_factor
        )
    }
}

/// Konsol çıktısı için formatlama kuralı (println! için)
impl fmt::Display for AuditRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.is_record {
            "🌟 REKOR!  "
        } else {
            "🔄 İlerleme"
        };

        write!(
            f,
            "{}[Gen {:>3}] PnL: {:>8.4} | Win: {:>5.2}% | PF: {:>5.2} | Sharpe: {:>6.2} | MaxDD: {:>6.4} | Trades: {:>4} | Fit: {:>9.2} | MutRate: {:.2} | TP: {:.5} | SL: {:.5} | Risk: {:.5} | Cldwn: {:>5.0}",
            prefix, self.gen, self.pnl, self.win_rate, self.profit_factor, self.sharpe,
            self.max_drawdown, self.trades, self.fitness, self.mut_rate,
            self.tp, self.sl, self.risk, self.cooldown
        )
    }
}

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

    /// CSV dosyasını oluşturur ve başlığı yazar
    pub fn initialize_csv(&self) -> Result<()> {
        if !std::path::Path::new(&self.log_path).exists() {
            let mut file = fs::File::create(&self.log_path)?;
            writeln!(file, "{}", AuditRecord::CSV_HEADER)?;
        }
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

    /// ANA METOD: Hem konsola basar hem CSV'ye yazar.
    /// Veri tutarsızlığını imkansız hale getirir.
    pub fn log_generation(
        &self,
        gen: usize,
        best: &Genome,
        mut_rate: f32,
        time_sec: f32,
        is_record: bool,
    ) -> Result<()> {
        let record = AuditRecord::from_genome(gen, best, mut_rate, time_sec, is_record);

        // 1. CSV'ye ekle
        let mut file = OpenOptions::new().append(true).open(&self.log_path)?;
        writeln!(file, "{}", record.to_csv_row())?;

        // 2. Konsola yazdır
        println!("{}", record);

        Ok(())
    }
}
