# ========== DOSYA: sentinel-optimizer/scripts/inject_dna.sh ==========
#!/bin/sh
set -e

echo "🧬 [MLOps] Alpha DNA Enjeksiyon Protokolü Başlatıldı..."

# JQ kontrolü ve otomatik kurulum (Ubuntu/Debian için güvenli)
if ! command -v jq > /dev/null 2>&1; then
    echo "⚙️ 'jq' bulunamadı, sisteme otomatik olarak kuruluyor..."
    sudo apt-get update -y && sudo apt-get install -y jq
fi

# Betiğin nerede olduğunu güvenli ve evrensel bir şekilde bulur (POSIX uyumlu)
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

# Risk Parametrelerini çekiyoruz (Yüksek Hassasiyet)
TP=$(jq -r '.[0].weights[39]' "$HOF_FILE")
SL=$(jq -r '.[0].weights[40]' "$HOF_FILE")
COOLDOWN=$(jq -r '.[0].weights[41] | round' "$HOF_FILE")
RISK=$(jq -r '.[0].weights[42]' "$HOF_FILE")

echo "💉 Yeni DNA weights.rs dosyasına yazılıyor..."

cat <<EOF > "$WEIGHTS_FILE"
// ========== DOSYA: sentinel-inference/src/weights.rs ==========
// 🚀 UYARI: Bu dosya sentinel-optimizer tarafından OTOMATİK ENJEKTE EDİLMİŞTİR.
// 🧬 FITNESS: $FITNESS | PNL: \$$PNL

pub fn get_dna_weights() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[0:36] | _nwise(3) | "        \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
    ]
}

pub fn get_dna_biases() -> Vec<f32> {
    vec![
$(jq -r '.[0].weights[36:39] | "        \(.[0]), \(.[1]), \(.[2]),"' "$HOF_FILE")
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
    git commit -m "chore(model): 🧬 inject new alpha DNA [Fit: $FITNESS, PnL: $PNL]"
    git push origin main
    echo "🦅 GÖREV TAMAMLANDI! GitHub Actions şu an yeni imajı derliyor."
fi