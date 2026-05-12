// ========== DOSYA: sentinel-optimizer/src/settings.rs ==========

/// 🏦 1. PİYASA VE SİMÜLATÖR KURALLARI (Gerçekçilik Ayarları)
pub const INITIAL_BALANCE: f64 = 1000.0;
pub const FEE_RATE: f64 = 0.0; // 0.0 = Maker (Sıfır Komisyon), 0.0002 = Taker (Piyasa Emri)
pub const BASE_SLIPPAGE_PCT: f64 = 0.00001; // %0.001 Kayma (Slippage)

/// 🧬 2. GENETİK ALGORİTMA DİNAMİKLERİ (Evrim Ayarları)
pub const STAGNATION_LIMIT: u32 = 5; // Kaç nesil rekor kırılamazsa kıyamet kopar?
pub const MUTATION_BASE_RATE: f32 = 0.15; // Normal mutasyon şansı
pub const MUTATION_CATACLYSM_RATE: f32 = 0.50; // Kıyamet anındaki mutasyon şansı
pub const EXTINCTION_DEATH_RATE: f32 = 0.95; // Kıyamette nüfusun yüzde kaçı silinir? (%95)

/// 🎯 3. FİTNESS (UYGUNLUK) HEDEFLERİ VE SINIRLARI
pub const MIN_REQUIRED_TRADES: usize = 3000; // HFT için 30 günde istenen minimum işlem
pub const MAX_ALLOWED_DD: f64 = 30.0; // Maksimum tolere edilebilir sermaye erimesi (%)
pub const TARGET_WIN_RATE: f64 = 30.0; // Hedeflenen minimum kazanma oranı (%)
pub const TARGET_PROFIT_FACTOR: f64 = 1.0; // 1.0 altı zarar demektir.

/// 🧬 4. DNA SINIRLARI (Genetik Keşif Alanı)
pub const DNA_TP_MIN: f32 = 0.002;
pub const DNA_TP_MAX: f32 = 0.020;
pub const DNA_SL_MIN: f32 = 0.001;
pub const DNA_SL_MAX: f32 = 0.015;
pub const DNA_COOLDOWN_MIN: f32 = 100.0; // ms
pub const DNA_COOLDOWN_MAX: f32 = 5000.0; // ms
pub const DNA_CONFIDENCE_MIN: f32 = 0.35;
pub const DNA_CONFIDENCE_MAX: f32 = 0.65;

/// 🖨️ DIŞ DENETÇİLER İÇİN MANİFESTO YAZDIRICI
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
        "🏦 Market Rules  : Fee: {}% | Slippage: {}%",
        FEE_RATE * 100.0,
        BASE_SLIPPAGE_PCT * 100.0
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
        "🧬 Genetics      : Stagnation Limit: {} Gen | Extinction Rate: %{}",
        STAGNATION_LIMIT,
        EXTINCTION_DEATH_RATE * 100.0
    );
    println!(
        "===================================================================================\n"
    );
}
