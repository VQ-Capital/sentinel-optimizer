// ========== DOSYA: sentinel-optimizer/src/settings.rs ==========

// Binance Spot Piyasası: Maker (Tahtaya yazan) %0.10, Taker (Piyasadan alan) %0.10.
// Binance Vadeli İşlemler (Futures): Long/Short yapabildiğimiz yer burasıdır. Sıradan kullanıcı için Maker %0.02, Taker %0.05'tir.
// BtcTurk Pro: Kripto-Kripto çiftlerinde Maker %0.05, Taker %0.09'dur.
// Hyperliquid (Merkeziyetsiz HFT): Maker %0.01, Taker %0.035'tir.

/// 🏦 1. PİYASA VE SİMÜLATÖR KURALLARI (Gerçekçilik Ayarları)
pub const INITIAL_BALANCE: f64 = 1000.0;

// 🔥 CERRAHİ: Binance Vadeli İşlemler (Futures) VIP 0 Maker Komisyonu (%0.02)
pub const FEE_RATE: f64 = 0.0002;

// 🔥 CERRAHİ: Gerçekçi HFT Kayması (Slippage) - %0.005 (0.5 bps)
pub const BASE_SLIPPAGE_PCT: f64 = 0.00005;

/// 🧬 2. GENETİK ALGORİTMA DİNAMİKLERİ (Evrim Ayarları)
pub const STAGNATION_LIMIT: u32 = 8; // Durgunluk toleransı
pub const MUTATION_BASE_RATE: f32 = 0.15; // Normal mutasyon şansı
pub const MUTATION_CATACLYSM_RATE: f32 = 0.60; // Kıyamet anındaki mutasyon şansı
pub const EXTINCTION_DEATH_RATE: f32 = 0.60; // Kıyamette nüfusun %60'ı silinir

/// 🎯 3. FİTNESS (UYGUNLUK) HEDEFLERİ VE SINIRLARI
// 🔥 CERRAHİ: Quick ve 1D testler için makul bir alt limit belirlendi (Eski: 250)
pub const MIN_REQUIRED_TRADES: usize = 50;
pub const MAX_ALLOWED_DD: f64 = 25.0; // Risk toleransı
pub const TARGET_WIN_RATE: f64 = 40.0; // Komisyonu yenmek için hedef WinRate %40
pub const TARGET_PROFIT_FACTOR: f64 = 1.2; // 1.2 altı uzun vadede risklidir

/// 🧬 4. DNA SINIRLARI (Genetik Keşif Alanı)
pub const DNA_TP_MIN: f32 = 0.002;
pub const DNA_TP_MAX: f32 = 0.020;
pub const DNA_SL_MIN: f32 = 0.001;
pub const DNA_SL_MAX: f32 = 0.015;
pub const DNA_COOLDOWN_MIN: f32 = 100.0; // ms
pub const DNA_COOLDOWN_MAX: f32 = 5000.0; // ms
                                          // 🔥 CERRAHİ: Softmax (1/3 = 0.33) tabanından dolayı AI'ın işlem açmasını kolaylaştırıyoruz.
pub const DNA_CONFIDENCE_MIN: f32 = 0.34;
pub const DNA_CONFIDENCE_MAX: f32 = 0.45;

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
        "🏦 Market Rules  : Fee: %{} (Binance VIP 0) | Slippage: %{}",
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
