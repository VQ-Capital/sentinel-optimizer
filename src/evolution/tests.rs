use super::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

// ============================================================================
// HELPER FONKSIYONLAR (Deterministik RNG ile)
// ============================================================================

fn create_random_genome_seeded(seed: u64) -> Genome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut dna: Vec<f32> = Vec::with_capacity(136);

    for _ in 0..128 {
        dna.push(rng.gen_range(-0.5..0.5));
    }

    dna.push(rng.gen_range(-0.5..0.0)); // Hold
    dna.push(rng.gen_range(0.1..0.5)); // Buy
    dna.push(rng.gen_range(0.1..0.5)); // Sell

    dna.push(rng.gen_range(DNA_TP_MIN..DNA_TP_MAX));
    dna.push(rng.gen_range(DNA_SL_MIN..DNA_SL_MAX));
    dna.push(rng.gen_range(DNA_COOLDOWN_MIN..DNA_COOLDOWN_MAX));
    dna.push(rng.gen_range(0.01..0.05));
    dna.push(rng.gen_range(DNA_CONFIDENCE_MIN..DNA_CONFIDENCE_MAX));

    Genome {
        weights: dna,
        fitness: f64::NEG_INFINITY,
        pnl: 0.0,
        sharpe: 0.0,
        max_drawdown: 0.0,
        trades: 0,
        generation: 0,
        win_rate: 0.0,
        profit_factor: 0.0,
    }
}

fn evolve_population_seeded(
    current_pop: &[Genome],
    total_size: usize,
    mut_rate: f32,
    is_cataclysm: bool,
    seed: u64,
) -> Vec<Genome> {
    let mut new_pop = Vec::with_capacity(total_size);
    let mut rng = StdRng::seed_from_u64(seed);

    let elite_count = if is_cataclysm {
        (total_size / 20).max(2)
    } else {
        (total_size / 10).max(2)
    };

    new_pop.extend_from_slice(&current_pop[0..elite_count]);

    let random_injection = if is_cataclysm {
        (total_size as f32 * EXTINCTION_DEATH_RATE) as usize
    } else {
        (total_size as f32 * 0.15) as usize
    };

    while new_pop.len() < total_size - random_injection {
        let p1 = &current_pop[rng.gen_range(0..elite_count * 2)];
        let p2 = &current_pop[rng.gen_range(0..elite_count * 2)];

        let mut child_dna = Vec::with_capacity(136);
        for i in 0..136 {
            let mut gene = if rng.gen_bool(0.5) {
                p1.weights[i]
            } else {
                p2.weights[i]
            };

            if i < 131 && rng.gen_bool(0.05) {
                gene = -gene;
            }

            if rng.gen_bool(mut_rate as f64) {
                let severity = if is_cataclysm { 2.0 } else { 1.0 };
                if i < 128 {
                    gene += rng.gen_range(-0.2..0.2) * severity;
                } else if (128..131).contains(&i) {
                    gene += rng.gen_range(-0.1..0.1) * severity;
                } else if i == 131 || i == 132 {
                    gene += rng.gen_range(-0.001..0.001) * severity;
                } else if i == 133 {
                    gene += rng.gen_range(-50.0..50.0) * severity;
                } else if i == 135 {
                    gene += rng.gen_range(-0.01..0.01) * severity;
                } else {
                    gene += rng.gen_range(-0.005..0.005);
                }
            }

            if i == 131 {
                gene = gene.clamp(DNA_TP_MIN, DNA_TP_MAX);
            } else if i == 132 {
                gene = gene.clamp(DNA_SL_MIN, DNA_SL_MAX);
            } else if i == 133 {
                gene = gene.clamp(DNA_COOLDOWN_MIN, DNA_COOLDOWN_MAX);
            } else if i == 134 {
                gene = gene.clamp(0.01, 0.05);
            } else if i == 135 {
                gene = gene.clamp(DNA_CONFIDENCE_MIN, DNA_CONFIDENCE_MAX);
            } else if (128..131).contains(&i) {
                gene = gene.clamp(-1.0, 1.0);
            } else {
                gene = gene.clamp(-3.0, 3.0);
            }

            child_dna.push(gene);
        }
        new_pop.push(Genome {
            weights: child_dna,
            fitness: f64::NEG_INFINITY,
            pnl: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
            trades: 0,
            generation: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
        });
    }

    while new_pop.len() < total_size {
        new_pop.push(create_random_genome_seeded(seed + new_pop.len() as u64));
    }
    new_pop
}

fn create_mock_population(size: usize, fitness_values: Vec<f64>) -> Vec<Genome> {
    let mut pop = Vec::with_capacity(size);
    for (i, &fit) in fitness_values.iter().cycle().take(size).enumerate() {
        let mut g = create_random_genome();
        g.fitness = fit;
        g.pnl = fit / 10000.0;
        g.trades = 1000; // Ceza almasın
        g.generation = i;
        pop.push(g);
    }
    pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    pop
}

// ============================================================================
// 1. UNIT TESTLER (Temel doğruluk)
// ============================================================================

#[test]
fn test_fitness_korkaklik_cezasi() {
    let f = calculate_fitness(1000.0, 2.0, 0.1, 5, 55.0, 1.5);
    assert!(f < -20_000_000.0, "Korkaklık cezası uygulanmadı: {}", f);
}

#[test]
fn test_fitness_iflas_cezasi() {
    let f = calculate_fitness(1000.0, 2.0, MAX_ALLOWED_DD + 0.01, 1000, 55.0, 1.5);
    assert!(f < -10_000_000.0, "İflas cezası uygulanmadı: {}", f);
}

#[test]
fn test_fitness_karli_model_odullendirmesi() {
    let f = calculate_fitness(100.0, 2.5, 0.05, 500, 60.0, 2.0);
    assert!(f > 0.0, "Karlı model negatif skor aldı: {}", f);

    let f_low_sharpe = calculate_fitness(100.0, 0.5, 0.05, 500, 60.0, 2.0);
    assert!(f > f_low_sharpe, "Sharpe bonusu işe yaramıyor");
}

#[test]
fn test_fitness_zararli_model_negatif() {
    let f = calculate_fitness(-50.0, 1.0, 0.05, 500, 45.0, 1.2);
    assert!(
        f < 0.0 && f > -1_000_000.0,
        "Zararlı model skoru anormal: {}",
        f
    );
}

#[test]
fn test_fitness_activity_bonus() {
    let f_100 = calculate_fitness(100.0, 2.0, 0.05, 100, 60.0, 2.0);
    let f_500 = calculate_fitness(100.0, 2.0, 0.05, 500, 60.0, 2.0);
    assert!(f_500 > f_100, "Aktivite ödülü işe yaramıyor");
}

#[test]
fn test_random_genome_boyut() {
    let g = create_random_genome();
    assert_eq!(g.weights.len(), 136, "DNA boyutu yanlış");
}

#[test]
fn test_random_genome_clamping() {
    let g = create_random_genome();

    for i in 128..131 {
        assert!(
            g.weights[i] >= -1.0 && g.weights[i] <= 1.0,
            "Action threshold {} sınırlar dışında: {}",
            i,
            g.weights[i]
        );
    }

    assert!(g.weights[131] >= DNA_TP_MIN && g.weights[131] <= DNA_TP_MAX);
    assert!(g.weights[132] >= DNA_SL_MIN && g.weights[132] <= DNA_SL_MAX);
    assert!(g.weights[133] >= DNA_COOLDOWN_MIN && g.weights[133] <= DNA_COOLDOWN_MAX);
    assert!(g.weights[134] >= 0.01 && g.weights[134] <= 0.05);
    assert!(g.weights[135] >= DNA_CONFIDENCE_MIN && g.weights[135] <= DNA_CONFIDENCE_MAX);
}

#[test]
fn test_evolve_population_boyutu_korunur() {
    let pop = create_mock_population(100, vec![100.0, 50.0, 25.0, 10.0, 5.0]);
    let new_pop = evolve_population(&pop, 100, 0.1, false);
    assert_eq!(new_pop.len(), 100, "Popülasyon boyutu değişmiş");
}

#[test]
fn test_evolve_elite_korunur() {
    let pop = create_mock_population(100, (0..100).map(|i| i as f64 * 100.0).collect());
    let new_pop = evolve_population(&pop, 100, 0.0, false);

    for i in 0..10 {
        assert!(
            new_pop[i].fitness == f64::NEG_INFINITY || (new_pop[i].weights == pop[i].weights),
            "Elite genom {} değişmiş",
            i
        );
    }
}

#[test]
fn test_evolve_clamping_korunur() {
    let pop = create_mock_population(50, vec![1000.0; 50]);
    let new_pop = evolve_population(&pop, 50, 1.0, true);

    for g in &new_pop {
        assert!(
            g.weights[131] >= DNA_TP_MIN && g.weights[131] <= DNA_TP_MAX,
            "TP clamp başarısız: {}",
            g.weights[131]
        );
        assert!(
            g.weights[132] >= DNA_SL_MIN && g.weights[132] <= DNA_SL_MAX,
            "SL clamp başarısız: {}",
            g.weights[132]
        );
        assert!(
            g.weights[133] >= DNA_COOLDOWN_MIN && g.weights[133] <= DNA_COOLDOWN_MAX,
            "Cooldown clamp başarısız: {}",
            g.weights[133]
        );
    }
}

// ============================================================================
// 2. DETERMINISTIK SEED TESTLERI
// ============================================================================

#[test]
fn test_seeded_random_genome_deterministik() {
    let g1 = create_random_genome_seeded(42);
    let g2 = create_random_genome_seeded(42);

    assert_eq!(g1.weights, g2.weights, "Aynı seed farklı genom üretiyor!");
}

#[test]
fn test_seeded_evolution_deterministik() {
    let pop = create_mock_population(20, vec![100.0; 20]);
    let e1 = evolve_population_seeded(&pop, 20, 0.1, false, 12345);
    let e2 = evolve_population_seeded(&pop, 20, 0.1, false, 12345);

    for (i, (a, b)) in e1.iter().zip(e2.iter()).enumerate() {
        assert_eq!(a.weights, b.weights, "Seed {}: genom {} farklı!", 12345, i);
    }
}

#[test]
fn test_farkli_seed_farkli_genom() {
    let g1 = create_random_genome_seeded(42);
    let g2 = create_random_genome_seeded(43);

    let fark_var = g1
        .weights
        .iter()
        .zip(g2.weights.iter())
        .any(|(a, b)| a != b);
    assert!(fark_var, "Farklı seed aynı genom üretti");
}

// ============================================================================
// 3. FITNESS LANDSCAPE TESTLERI
// ============================================================================

#[test]
fn test_fitness_landscape_siralama() {
    let iyi = calculate_fitness(500.0, 3.0, 0.02, 1000, 65.0, 3.0);
    let orta = calculate_fitness(100.0, 1.5, 0.08, 500, 52.0, 1.5);
    let kotu = calculate_fitness(-100.0, 0.5, 0.20, 50, 35.0, 0.8);

    assert!(
        iyi > orta,
        "İyi skor ortadan düşük: iyi={}, orta={}",
        iyi,
        orta
    );
    assert!(
        orta > kotu,
        "Orta skor kötüden düşük: orta={}, kotu={}",
        orta,
        kotu
    );
    assert!(
        kotu < -20_000_000.0,
        "Kötü model aşırı ceza almadı: {}",
        kotu
    );
}

#[test]
fn test_fitness_landscape_optimum_robust() {
    let base = calculate_fitness(1000.0, 4.0, 0.01, 2000, 70.0, 5.0);

    let lower_pnl = calculate_fitness(500.0, 4.0, 0.01, 2000, 70.0, 5.0);
    let lower_sharpe = calculate_fitness(1000.0, 1.0, 0.01, 2000, 70.0, 5.0);
    let higher_dd = calculate_fitness(1000.0, 4.0, 0.15, 2000, 70.0, 5.0);

    assert!(base > lower_pnl, "PnL düşünce skor artmamalı");
    assert!(base > lower_sharpe, "Sharpe düşünce skor artmamalı");
    assert!(base > higher_dd, "DD yükselince skor artmamalı");
}

#[test]
fn test_fitness_landscape_monotonik_pnl() {
    let f1 = calculate_fitness(0.0, 2.0, 0.05, 1000, 55.0, 2.0);
    let f2 = calculate_fitness(100.0, 2.0, 0.05, 1000, 55.0, 2.0);
    let f3 = calculate_fitness(200.0, 2.0, 0.05, 1000, 55.0, 2.0);

    assert!(f2 > f1, "PnL 0->100 artmadı");
    assert!(f3 > f2, "PnL 100->200 artmadı");
}

// ============================================================================
// 4. ABLATION TESTLERI
// ============================================================================

#[test]
fn test_ablation_sharpe_bonus_etkisi() {
    let base = calculate_fitness(100.0, 2.0, 0.05, 1000, 55.0, 2.0);
    let no_sharpe = calculate_fitness(100.0, 0.0, 0.05, 1000, 55.0, 2.0);

    let diff = base - no_sharpe;
    let expected = 2.0 * 50_000.0;

    assert!(
        (diff - expected).abs() < 1.0,
        "Sharpe ablation beklenenden farklı: diff={}, expected={}",
        diff,
        expected
    );
}

#[test]
fn test_ablation_profit_factor_bonus_etkisi() {
    // PF 2.0 vs 1.5 (ikisi de > TARGET_PROFIT_FACTOR=1.1, penalty yok)
    // Sadece bonus farkı: 0.5 * 30_000 = 15_000
    let base = calculate_fitness(100.0, 2.0, 0.05, 1000, 55.0, 2.0);
    let lower_pf = calculate_fitness(100.0, 2.0, 0.05, 1000, 55.0, 1.5);

    let diff = base - lower_pf;
    let expected = (2.0 - 1.5) * 30_000.0;

    assert!(
        (diff - expected).abs() < 1.0,
        "PF ablation beklenenden farklı: diff={}, expected={}",
        diff,
        expected
    );
}

#[test]
fn test_ablation_activity_bonus_etkisi() {
    // trades >= MIN_REQUIRED_TRADES olmalı yoksa korkaklık cezası devreye girer
    let f_1000 = calculate_fitness(100.0, 2.0, 0.05, 1000, 60.0, 2.0);
    let f_1500 = calculate_fitness(100.0, 2.0, 0.05, 1500, 60.0, 2.0);

    let diff = f_1500 - f_1000;
    let expected = (1500.0 - 1000.0) * (60.0 / 100.0) * 100.0;

    assert!(
        (diff - expected).abs() < 1.0,
        "Activity ablation beklenenden farklı: diff={}, expected={}",
        diff,
        expected
    );
}

#[test]
fn test_ablation_pnl_penalty_vs_reward() {
    let pos = calculate_fitness(100.0, 2.0, 0.05, 1000, 55.0, 2.0);
    let neg = calculate_fitness(-100.0, 2.0, 0.05, 1000, 55.0, 2.0);

    assert!(pos > 0.0, "Pozitif PnL negatif skor aldı");
    assert!(neg < 0.0, "Negatif PnL pozitif skor aldı");
    assert!(
        pos > neg.abs(),
        "Negatif PnL cezası pozitif ödülünden büyük olmamalı"
    );
}

// ============================================================================
// 5. OVERFIT / GENERALIZASYON TESTLERI
// ============================================================================

#[test]
fn test_fitness_farkli_market_regimeleri() {
    let bull = calculate_fitness(800.0, 2.5, 0.03, 1500, 60.0, 2.5);
    let bear = calculate_fitness(-300.0, -0.5, 0.25, 800, 40.0, 0.7);
    let sideways = calculate_fitness(50.0, 0.8, 0.08, 600, 48.0, 1.1);

    assert!(bull > sideways, "Bull piyasada skor sideways'ten düşük");
    assert!(sideways > bear, "Sideways skor bear'den düşük");
    // Bear market negatif olmalı ama -10M cezası almak zorunda değil
    // (trades=800>500, dd=0.25<<30, sadece zarar cezası)
    assert!(bear < 0.0, "Bear market pozitif skor aldı: {}", bear);
}

#[test]
fn test_overfit_az_trade_cezasi() {
    let overfit = calculate_fitness(2000.0, 5.0, 0.01, 20, 90.0, 10.0);
    let generalist = calculate_fitness(500.0, 1.8, 0.05, 1500, 55.0, 1.8);

    assert!(
        overfit < -20_000_000.0,
        "Overfit model (20 trade) korkaklık cezası almadı: {}",
        overfit
    );

    assert!(
        generalist > 0.0,
        "Genelleştiren model negatif skor aldı: {}",
        generalist
    );
}

#[test]
fn test_drawdown_genelleme_toleransi() {
    let low_dd = calculate_fitness(300.0, 2.0, 0.02, 1000, 55.0, 2.0);
    let high_dd = calculate_fitness(300.0, 2.0, 0.25, 1000, 55.0, 2.0);

    assert!(low_dd > high_dd, "Düşük DD yüksek DD'den düşük skor aldı");

    let diff = low_dd - high_dd;
    assert!(diff > 1000.0, "DD cezası çok düşük: {}", diff);
}

// ============================================================================
// 6. REGRESSION / SNAPSHOT TESTI
// ============================================================================

#[test]
fn test_fitness_regression_finite_kontrolu() {
    let f1 = calculate_fitness(150.0, 1.8, 0.08, 300, 52.0, 1.8);
    let f2 = calculate_fitness(-10.0, 0.5, 0.15, 100, 40.0, 0.9);

    assert!(f1.is_finite(), "Fitness f1 NaN/Inf: {}", f1);
    assert!(f2.is_finite(), "Fitness f2 NaN/Inf: {}", f2);
}

#[test]
fn test_multi_generation_elite_korunur() {
    let mut pop = (0..50).map(|_| create_random_genome()).collect::<Vec<_>>();

    for g in pop.iter_mut() {
        g.fitness = (g.weights[0] as f64 * 1000.0) + 5000.0;
        g.trades = 1000;
    }
    pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    let initial_best = pop[0].fitness;

    for gen in 0..10 {
        pop = evolve_population_seeded(&pop, 50, 0.05, false, 999 + gen as u64);
        // Fitness'ı sabit formülle yeniden hesapla
        for g in pop.iter_mut() {
            g.fitness = (g.weights[0] as f64 * 1000.0) + 5000.0;
            g.trades = 1000;
        }
        pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    }

    // Elite korunması "mevcut jenerasyonun en iyilerini" korur.
    // İlk jenerasyonun spesifik genomunun 10 jenerasyon sonra hala popülasyonda
    // olması beklenmez (daha iyi çocuklar onu geçer).
    // Ama popülasyonun en iyi fitness'ı asla düşmemeli (elite'ler korunuyor).
    let final_best = pop[0].fitness;
    assert!(
        final_best >= initial_best,
        "Fitness düştü: {} -> {}",
        initial_best,
        final_best
    );
}

// ============================================================================
// 7. CONVERGENCE DEMO (Integration-style test with print output)
//    cargo test test_evolution_convergence_demo -- --nocapture
// ============================================================================

fn evaluate_mock_genome(g: &mut Genome) {
    let w = &g.weights;

    // PnL: Ağırlıkların lineer kombinasyonu (evrim bunu optimize etmeye çalışacak)
    let pnl = (w[0] as f64 * 1000.0) + (w[1] as f64 * 500.0) + (w[2] as f64 * 200.0) + 100.0;

    // Sharpe: Action threshold'lardan türet
    let sharpe = ((w[128] + w[129] + w[130]) as f64 + 1.5).max(0.0) * 1.5;

    // Max Drawdown: TP ve SL'den türet (küçük kalması lazım)
    let max_dd = ((w[131] + w[132]) as f64 * 100.0).abs();

    // Trades: Cooldown'dan türet (az cooldown = çok trade)
    let trades = ((2000.0 - w[133]) * 2.0).max(0.0) as usize + 550;

    // Win Rate: Confidence'dan türet
    let win_rate = (w[135] as f64) * 150.0;

    // Profit Factor: SL'den türet (her zaman > 1.1 olacak şekilde)
    let profit_factor = (1.0 / (w[132] as f64 + 0.01)).min(10.0);

    g.pnl = pnl;
    g.sharpe = sharpe;
    g.max_drawdown = max_dd;
    g.trades = trades;
    g.win_rate = win_rate;
    g.profit_factor = profit_factor;
    g.fitness = calculate_fitness(pnl, sharpe, max_dd, trades, win_rate, profit_factor);
}

#[test]
fn test_evolution_convergence_demo() {
    const POP_SIZE: usize = 250;
    const GENERATIONS: usize = 500;
    const SEED_BASE: u64 = 42;

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║     VQ-CAPITAL EVOLUTION CONVERGENCE DEMO (Mock Backtest)      ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");

    let mut pop = (0..POP_SIZE)
        .map(|i| create_random_genome_seeded(SEED_BASE + i as u64))
        .collect::<Vec<_>>();

    let mut best_history: Vec<f64> = Vec::with_capacity(GENERATIONS);
    let mut avg_history: Vec<f64> = Vec::with_capacity(GENERATIONS);

    for gen in 0..GENERATIONS {
        for g in pop.iter_mut() {
            evaluate_mock_genome(g);
        }

        pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best = pop[0].fitness;
        let avg = pop.iter().map(|g| g.fitness).sum::<f64>() / pop.len() as f64;
        let worst = pop.last().unwrap().fitness;

        best_history.push(best);
        avg_history.push(avg);

        println!(
            "║ Gen {:02} │ Best: {:>12.2} │ Avg: {:>12.2} │ Worst: {:>12.2} │ Trades: {} ║",
            gen, best, avg, worst, pop[0].trades
        );

        if gen < GENERATIONS - 1 {
            pop = evolve_population_seeded(
                &pop,
                POP_SIZE,
                0.15,
                false,
                SEED_BASE + 1000 + gen as u64,
            );
        }
    }

    let first_best = best_history[0];
    let last_best = best_history[GENERATIONS - 1];
    let improvement = ((last_best - first_best) / first_best.abs()) * 100.0;

    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  İlk Gen Best Fitness: {:>12.2}                                 ║",
        first_best
    );
    println!(
        "║  Son Gen Best Fitness: {:>12.2}                                 ║",
        last_best
    );
    println!(
        "║  İyileşme:             {:>11.2}%                                ║",
        improvement
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");

    assert!(
        last_best >= first_best,
        "Evrim çalışmıyor! Fitness düştü: {} -> {}",
        first_best,
        last_best
    );

    let last_5_avg = avg_history.iter().rev().take(5).sum::<f64>() / 5.0;
    let first_5_avg = avg_history.iter().take(5).sum::<f64>() / 5.0;
    assert!(
        last_5_avg > first_5_avg,
        "Popülasyon ortalaması iyileşmedi: {} -> {}",
        first_5_avg,
        last_5_avg
    );
}
