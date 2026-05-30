// ========== DOSYA: sentinel-optimizer/src/settings.rs ==========

pub const INITIAL_BALANCE: f64 = 1000.0;
pub const FEE_RATE: f64 = 0.0002;
pub const BASE_SLIPPAGE_PCT: f64 = 0.00005;

pub const STAGNATION_LIMIT: u32 = 15;
pub const MUTATION_BASE_RATE: f32 = 0.05;
pub const MUTATION_CATACLYSM_RATE: f32 = 0.30;
pub const EXTINCTION_DEATH_RATE: f32 = 0.75;

pub const EARLY_STOP_LIMIT: u32 = 30;

pub const MIN_REQUIRED_TRADES: usize = 150;
pub const MAX_ALLOWED_DD: f64 = 30.0;

pub const TARGET_WIN_RATE: f64 = 30.0;
pub const TARGET_PROFIT_FACTOR: f64 = 1.05;

pub const DNA_TP_MIN: f32 = 0.001;
pub const DNA_TP_MAX: f32 = 0.015;
pub const DNA_SL_MIN: f32 = 0.001;
pub const DNA_SL_MAX: f32 = 0.010;
pub const DNA_COOLDOWN_MIN: f32 = 50.0;
pub const DNA_COOLDOWN_MAX: f32 = 1000.0;
pub const DNA_CONFIDENCE_MIN: f32 = 0.334;
pub const DNA_CONFIDENCE_MAX: f32 = 0.600;

// 🔥 YENİ: YAPAY ZEKA KALDIRAÇ VE BEKLEME SÜRESİNİ KENDİ BULACAK
pub const DNA_LEVERAGE_MIN: f32 = 1.0;
pub const DNA_LEVERAGE_MAX: f32 = 10.0;
pub const DNA_HOLD_TIME_MIN: f32 = 3_600_000.0; // 1 Saat
pub const DNA_HOLD_TIME_MAX: f32 = 86_400_000.0; // 24 Saat

pub fn print_experiment_manifest(symbol: &str, dataset: &str, gen: usize, pop: usize) {
    println!(
        "\n==================================================================================="
    );
    println!("🧪 VQ-CAPITAL EXPERIMENT MANIFESTO (DIŞ DENETÇİ KILAVUZU)");
    println!("===================================================================================");
    println!("📍 Target Data   : {} | Dataset: {}", symbol, dataset);
    println!(
        "👥 Evolution     : {} Generations | {} Population",
        gen, pop
    );
    println!(
        "🛑 Strict Limits : Min Trades: {} | Max Drawdown: %{}",
        MIN_REQUIRED_TRADES, MAX_ALLOWED_DD
    );
    println!(
        "🎯 Target Goals  : Min WinRate: %{} | Min ProfitFactor: {}",
        TARGET_WIN_RATE, TARGET_PROFIT_FACTOR
    );
    println!(
        "⏹️ Early Stopping: {} Generations without improvement",
        EARLY_STOP_LIMIT
    );
    println!("🧬 DNA Structure : Expanded to 138 Genes (Includes AI-Driven Leverage & Hold Time)");
    println!(
        "===================================================================================\n"
    );
}
