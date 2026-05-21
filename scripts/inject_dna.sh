#!/bin/bash
# ========== DOSYA: sentinel-optimizer/scripts/inject_dna.sh ==========
set -e

echo "🧬 [MLOps] Alpha DNA Enjeksiyon Protokolü Başlatıldı (SINGULARITY MODE)..."

if ! command -v jq > /dev/null 2>&1; then
    echo "⚙️ 'jq' bulunamadı, sisteme otomatik olarak kuruluyor..."
    sudo apt-get update -y && sudo apt-get install -y jq
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
ORG_ROOT="$DIR/../.."

HOF_FILE="$ORG_ROOT/sentinel-optimizer/hall_of_fame.json"
# 🔥 ARTIK TEK BİR DOĞRULUK KAYNAĞIMIZ VAR
DNA_FILE="$ORG_ROOT/sentinel-core/src/dna.rs"

if [ ! -f "$HOF_FILE" ]; then
    echo "🚨 hall_of_fame.json bulunamadı! Aranan dizin: $HOF_FILE"
    exit 1
fi

echo "🔍 En iyi Genom analiz ediliyor..."
FITNESS=$(jq -r '.[0].fitness' "$HOF_FILE")
PNL=$(jq -r '.[0].pnl' "$HOF_FILE")

if (( $(echo "$PNL <= 0.0" | bc -l) )); then
    echo "🚨 KRİTİK İPTAL: DNA PnL negatif ($PNL$). Zarar eden model Production'a pushlanamaz!"
    exit 1
fi

# Yeni MLP DNA Yapısı İçin Risk Parametrelerini Çekiyoruz
TP=$(jq -r '.[0].weights[131]' "$HOF_FILE")
SL=$(jq -r '.[0].weights[132]' "$HOF_FILE")
COOLDOWN=$(jq -r '.[0].weights[133] | round' "$HOF_FILE")
RISK=$(jq -r '.[0].weights[134]' "$HOF_FILE")
CONF=$(jq -r '.[0].weights[135]' "$HOF_FILE")

echo "💉 Yeni MLP DNA (136 Gen) sentinel-core/src/dna.rs dosyasına kazınıyor (BAKED)..."

cat <<EOF > "$DNA_FILE"
// ========== DOSYA: sentinel-core/src/dna.rs ==========
// 🚀 UYARI: Bu dosya sentinel-optimizer/scripts/inject_dna.sh tarafından OTOMATİK GÜNCELLENİR.
// 🧬 SINGLE SOURCE OF TRUTH (SSOT) FOR AI & RISK | FITNESS: $FITNESS | PNL: \$$PNL

pub const W1: &[f32] = &[
$(jq -r '.[0].weights[0:96] | _nwise(3) | "    \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
];

pub const B1: &[f32] = &[
$(jq -r '.[0].weights[96:104] | _nwise(4) | "    \(.[0]), \(.[1]), \(.[2]), \(.[3]),"' "$HOF_FILE")
];

pub const W2: &[f32] = &[
$(jq -r '.[0].weights[104:128] | _nwise(3) | "    \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
];

pub const B2: &[f32] = &[
$(jq -r '.[0].weights[128:131] | "    \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
];

// 🛡️ RISK & EXECUTION PARAMETERS (Evrimleşen Limitler)
pub const TAKE_PROFIT_PCT: f64 = $TP;
pub const STOP_LOSS_PCT: f64 = $SL;
pub const COOLDOWN_MS: i64 = $COOLDOWN;
pub const RISK_PCT: f64 = $RISK;
pub const MIN_CONFIDENCE: f64 = $CONF;

// 🏦 MARKET PHYSICS (Sabit Kural)
pub const FEE_RATE: f64 = 0.0002;
pub const BASE_SLIPPAGE_PCT: f64 = 0.00005;
EOF

echo "✅ DNA Enjeksiyonu tamamlandı. Derleme doğrulanıyor..."

cd "$ORG_ROOT/sentinel-core"
cargo check --quiet

echo "📤 GitHub'a gönderiliyor (CI/CD Tetikleniyor)..."
git add src/dna.rs

if git diff --staged --quiet; then
    echo "⚠️ Yeni DNA mevcut DNA ile aynı. Commit işlemine gerek yok."
else
    git commit -m "chore(model): 🧬 inject new SINGULARITY DNA [Fit: $FITNESS, PnL: $PNL]"
    git push origin main
    echo "🦅 GÖREV TAMAMLANDI! GitHub Actions şu an yeni imajı derliyor."
fi