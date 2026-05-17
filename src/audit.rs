// ========== DOSYA: sentinel-optimizer/src/audit.rs ==========
use crate::Genome;
use anyhow::{Context, Result};
use chrono::Utc;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;

pub struct AuditRecord {
    pub timestamp: String,
    pub status: String,
    pub gen: usize,
    pub fitness: f64,
    pub pnl: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub trades: usize,
    pub mut_rate: f32,
    pub tp: f32,
    pub sl: f32,
    pub risk: f32,
    pub cooldown: f32,
    pub conf: f32,
    pub time_sec: f32,
    pub is_record: bool,
}

impl AuditRecord {
    pub const CSV_HEADER: &'static str =
        "Timestamp,Status,Gen,Fitness,PnL,WinRate,PF,Sharpe,MaxDD,Trades,MutRate,TP,SL,Risk,Cooldown,Conf,TimeSec";

    pub fn from_genome(
        gen: usize,
        best: &Genome,
        mut_rate: f32,
        time_sec: f32,
        is_record: bool,
    ) -> Self {
        Self {
            timestamp: Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            status: if is_record {
                "RECORD".to_string()
            } else {
                "UPDATE".to_string()
            },
            gen,
            fitness: best.fitness,
            pnl: best.pnl,
            win_rate: best.win_rate,
            profit_factor: best.profit_factor,
            sharpe: best.sharpe,
            max_drawdown: best.max_drawdown,
            trades: best.trades,
            mut_rate,
            tp: best.weights[131],
            sl: best.weights[132],
            cooldown: best.weights[133],
            risk: best.weights[134],
            conf: best.weights[135],
            time_sec,
            is_record,
        }
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{:.2},{:.4},{:.2},{:.2},{:.2},{:.4},{},{:.2},{:.6},{:.6},{:.6},{:.0},{:.3},{:.2}",
            self.timestamp,
            self.status,
            self.gen,
            self.fitness,
            self.pnl,
            self.win_rate,
            self.profit_factor,
            self.sharpe,
            self.max_drawdown,
            self.trades,
            self.mut_rate,
            self.tp,
            self.sl,
            self.risk,
            self.cooldown,
            self.conf,
            self.time_sec
        )
    }
}

impl fmt::Display for AuditRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let emoji = if self.is_record { "🌟" } else { "🔄" };

        write!(
            f,
            "{} [{}] {:>6} | Gen: {:>4} | Fit: {:>10.2} | PnL: {:>8.4} | Win: {:>5.2}% | PF: {:>4.2} | Sharpe: {:>6.2} | DD: {:>6.4} | Trd: {:>4} | Mut: {:.2} | TP: {:.4} | SL: {:.4} | Rsk: {:.4} | Cld: {:>4.0} | Cnf: {:.2} | T: {:>6.2}s",
            emoji,
            self.timestamp,
            self.status,
            self.gen,
            self.fitness,
            self.pnl,
            self.win_rate,
            self.profit_factor,
            self.sharpe,
            self.max_drawdown,
            self.trades,
            self.mut_rate,
            self.tp,
            self.sl,
            self.risk,
            self.cooldown,
            self.conf,
            self.time_sec
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

    pub fn log_generation(
        &self,
        gen: usize,
        best: &Genome,
        mut_rate: f32,
        time_sec: f32,
        is_record: bool,
    ) -> Result<()> {
        let record = AuditRecord::from_genome(gen, best, mut_rate, time_sec, is_record);

        let mut file = OpenOptions::new().append(true).open(&self.log_path)?;
        writeln!(file, "{}", record.to_csv_row())?;
        println!("{}", record);

        Ok(())
    }
}
