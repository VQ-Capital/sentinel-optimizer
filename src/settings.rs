// ========== DOSYA: sentinel-optimizer/src/settings.rs ==========

pub const INITIAL_BALANCE: f64 = 1000.0;
pub const FEE_RATE: f64 = 0.0002;
pub const BASE_SLIPPAGE_PCT: f64 = 0.00005;

pub const STAGNATION_LIMIT: u32 = 12;
pub const MUTATION_BASE_RATE: f32 = 0.20;
pub const MUTATION_CATACLYSM_RATE: f32 = 0.70;
pub const EXTINCTION_DEATH_RATE: f32 = 0.75;

pub const EARLY_STOP_LIMIT: u32 = 30;

// 🔥 QUANT KALİBRASYONU: İşlem zorunluluğu 500'den 150'ye düşürüldü. (Günde ortalama 5 kaliteli işlem)
pub const MIN_REQUIRED_TRADES: usize = 150;
pub const MAX_ALLOWED_DD: f64 = 30.0;

pub const TARGET_WIN_RATE: f64 = 40.0;
pub const TARGET_PROFIT_FACTOR: f64 = 1.1;

// 🔥 QUANT KALİBRASYONU: Hedef Kâr (TP) tavanı %3'ten %5'e çıkarıldı. AI daha büyük trendleri kovalayabilecek.
pub const DNA_TP_MIN: f32 = 0.005;
pub const DNA_TP_MAX: f32 = 0.050;
pub const DNA_SL_MIN: f32 = 0.002;
pub const DNA_SL_MAX: f32 = 0.025;
pub const DNA_COOLDOWN_MIN: f32 = 100.0;
pub const DNA_COOLDOWN_MAX: f32 = 2000.0;

pub const DNA_CONFIDENCE_MIN: f32 = 0.334;
pub const DNA_CONFIDENCE_MAX: f32 = 0.400;

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
    println!(
        "===================================================================================\n"
    );
}
