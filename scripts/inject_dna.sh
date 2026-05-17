#!/bin/bash
# ========== DOSYA: sentinel-optimizer/scripts/inject_dna.sh ==========
set -e

echo "🧬 [MLOps] Alpha DNA Enjeksiyon Protokolü Başlatıldı..."

if ! command -v jq > /dev/null 2>&1; then
    echo "⚙️ 'jq' bulunamadı, sisteme otomatik olarak kuruluyor..."
    sudo apt-get update -y && sudo apt-get install -y jq
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
ORG_ROOT="$DIR/../.."

HOF_FILE="$ORG_ROOT/sentinel-optimizer/hall_of_fame.json"
WEIGHTS_FILE="$ORG_ROOT/sentinel-inference/src/weights.rs"
ENV_FILE="$ORG_ROOT/sentinel-infra/.env"

if [ ! -f "$HOF_FILE" ]; then
    echo "🚨 hall_of_fame.json bulunamadı! Aranan dizin: $HOF_FILE"
    exit 1
fi

echo "🔍 En iyi Genom analiz ediliyor..."
FITNESS=$(jq -r '.[0].fitness' "$HOF_FILE")
PNL=$(jq -r '.[0].pnl' "$HOF_FILE")

# 🔥 CERRAHİ: Kırmızı Çizgi Koruması! Zarar eden modeli reddet.
if (( $(echo "$PNL <= 0.0" | bc -l) )); then
    echo "🚨 KRİTİK İPTAL: DNA PnL negatif ($PNL$). Zarar eden model Production'a pushlanamaz!"
    exit 1
fi

# Yeni MLP DNA Yapısı İçin Risk Parametrelerini Çekiyoruz
TP=$(jq -r '.[0].weights[131]' "$HOF_FILE")
SL=$(jq -r '.[0].weights[132]' "$HOF_FILE")
COOLDOWN=$(jq -r '.[0].weights[133] | round' "$HOF_FILE")
RISK=$(jq -r '.[0].weights[134]' "$HOF_FILE")

echo "💉 Yeni MLP DNA (136 Gen) weights.rs dosyasına yazılıyor..."

cat <<EOF > "$WEIGHTS_FILE"
// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// 🚀 UYARI: Bu dosya sentinel-optimizer tarafından OTOMATİK ENJEKTE EDİLMİŞTİR.
// 🧬 FITNESS: $FITNESS | PNL: \$$PNL

pub fn get_dna_w1() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[0:96] | _nwise(3) | "        \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
    ]
}

pub fn get_dna_b1() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[96:104] | _nwise(4) | "        \(.[0]), \(.[1]), \(.[2]), \(.[3]),"' "$HOF_FILE")
    ]
}

pub fn get_dna_w2() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[104:128] | _nwise(3) | "        \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
    ]
}

pub fn get_dna_b2() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[128:131] | "        \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
    ]
}
EOF

echo "⚙️ Risk Parametreleri sentinel-infra/.env dosyasına aşılanıyor..."
if [ -f "$ENV_FILE" ]; then
    sed -i "s/^TAKE_PROFIT=.*/TAKE_PROFIT=$TP/" "$ENV_FILE"
    sed -i "s/^STOP_LOSS=.*/STOP_LOSS=$SL/" "$ENV_FILE"
    sed -i "s/^COOLDOWN_MS=.*/COOLDOWN_MS=$COOLDOWN/" "$ENV_FILE"
    sed -i "s/^RISK_PCT=.*/RISK_PCT=$RISK/" "$ENV_FILE"
    echo "✅ .env dosyası güncellendi: TP=$TP | SL=$SL | COOLDOWN=$COOLDOWN | RISK=$RISK"
else
    echo "⚠️ $ENV_FILE bulunamadı! Lütfen manuel olarak güncelleyin."
fi

echo "✅ Kod enjeksiyonu tamamlandı. Derleme doğrulanıyor..."

cd "$ORG_ROOT/sentinel-inference"
cargo check --quiet

echo "📤 GitHub'a gönderiliyor (CI/CD Tetikleniyor)..."
git add src/weights.rs
git -C "$ORG_ROOT/sentinel-infra" add .env || true

if git diff --staged --quiet; then
    echo "⚠️ Yeni DNA mevcut DNA ile aynı. Commit işlemine gerek yok."
else
    git commit -m "chore(model): 🧬 inject new MLP alpha DNA [Fit: $FITNESS, PnL: $PNL]"
    git push origin main
    echo "🦅 GÖREV TAMAMLANDI! GitHub Actions şu an yeni imajı derliyor."
fi