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
DNA_FILE="$ORG_ROOT/sentinel-core/src/dna.rs"

if [ ! -f "$HOF_FILE" ]; then
    echo "🚨 hall_of_fame.json bulunamadı! Aranan dizin: $HOF_FILE"
    exit 1
fi

echo "🔍 En iyi Genom analiz ediliyor..."
FITNESS=$(jq -r '.[0].fitness' "$HOF_FILE")
PNL=$(jq -r '.[0].pnl' "$HOF_FILE")

# 🔥 CERRAHİ: Sadece Altyapı Testi için PnL kontrolünü pasife aldık
if (( $(echo "$PNL <= 0.0" | bc -l) )); then
    echo "🚨 KRİTİK İPTAL: DNA PnL negatif ($PNL$). Zarar eden model Production'a pushlanamaz!"
    exit 1
fi

TP=$(jq -r '.[0].weights[131]' "$HOF_FILE")
SL=$(jq -r '.[0].weights[132]' "$HOF_FILE")
COOLDOWN=$(jq -r '.[0].weights[133] | round' "$HOF_FILE")
RISK=$(jq -r '.[0].weights[134]' "$HOF_FILE")
CONF=$(jq -r '.[0].weights[135]' "$HOF_FILE")

echo "💉 Yeni MLP DNA (136 Gen) Ana sentinel-core/src/dna.rs dosyasına kazınıyor..."

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

// 🛡️ RISK & EXECUTION PARAMETERS
pub const TAKE_PROFIT_PCT: f64 = $TP;
pub const STOP_LOSS_PCT: f64 = $SL;
pub const COOLDOWN_MS: i64 = $COOLDOWN;
pub const RISK_PCT: f64 = $RISK;
pub const MIN_CONFIDENCE: f64 = $CONF;

// 🏦 MARKET PHYSICS
pub const FEE_RATE: f64 = 0.0002;
pub const BASE_SLIPPAGE_PCT: f64 = 0.00005;
EOF

echo "✅ 1. ANA ÇEKİRDEK GÜNCELLENİYOR..."
cd "$ORG_ROOT/sentinel-core"
cargo check --quiet

git add src/dna.rs
if git diff --staged --quiet; then
    echo "⚠️ Ana core reposunda değişiklik yok."
else
    git commit -m "chore(model): 🧬 inject new SINGULARITY DNA [Fit: $FITNESS, PnL: $PNL]"
    git push origin main
fi

echo "✅ 2. BAĞIMLI MİKROSERVİSLERİN ALT MODÜLLERİ (SUBMODULES) GÜNCELLENİYOR..."

REPOS=("sentinel-inference" "sentinel-execution" "sentinel-optimizer")

for REPO in "${REPOS[@]}"; do
    if [ -d "$ORG_ROOT/$REPO/sentinel-core" ]; then
        echo "🔄 $REPO güncelleniyor..."
        cd "$ORG_ROOT/$REPO"
        
        # Git Submodule'ü uzak sunucudaki son commit'e (main branch) zorlar
        git submodule update --remote sentinel-core
        
        git add sentinel-core
        if git diff --staged --quiet; then
            echo "   ➖ $REPO zaten güncel."
        else
            git commit -m "chore(core): 🧬 sync sentinel-core to latest DNA [Fit: $FITNESS]"
            git push origin main
            echo "   ✔️ $REPO senkronize edildi ve pushlandı."
        fi
    else
        echo "⚠️ $REPO içinde sentinel-core alt modülü bulunamadı, atlanıyor."
    fi
done

echo "🦅 GÖREV TAMAMLANDI! Tüm mikroservisler (Yerel ve Uzak) %100 Senkronize Edildi."