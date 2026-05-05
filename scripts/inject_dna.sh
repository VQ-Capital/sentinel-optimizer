#!/bin/bash
# ========== DOSYA: sentinel-optimizer/scripts/inject_dna.sh ==========
set -e

echo "🧬 [MLOps] Alpha DNA Enjeksiyon Protokolü Başlatıldı..."

# Betiğin nerede olduğunu dinamik olarak bulur
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Tüm repoların bulunduğu ana klasör (VQ-Capital)
ORG_ROOT="$SCRIPT_DIR/../.."

HOF_FILE="$ORG_ROOT/sentinel-optimizer/hall_of_fame.json"
WEIGHTS_FILE="$ORG_ROOT/sentinel-inference/src/weights.rs"

# JQ kontrolü
if ! command -v jq &> /dev/null; then
    echo "🚨 'jq' kurulu değil. Lütfen 'sudo apt-get install jq' çalıştırın."
    exit 1
fi

if [ ! -f "$HOF_FILE" ]; then
    echo "🚨 hall_of_fame.json bulunamadı! Aranan dizin: $HOF_FILE"
    exit 1
fi

echo "🔍 En iyi Genom analiz ediliyor..."
# En iyi genomun weights dizisini al
WEIGHTS=$(jq -r '.[0].weights | join(", ")' "$HOF_FILE")
FITNESS=$(jq -r '.[0].fitness' "$HOF_FILE")
PNL=$(jq -r '.[0].pnl' "$HOF_FILE")

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

echo "✅ Kod enjeksiyonu tamamlandı. Derleme doğrulanıyor..."

# Cargo Check yap (Hata varsa pushlamasın)
cd "$ORG_ROOT/sentinel-inference"
cargo check --quiet

echo "📤 GitHub'a gönderiliyor (CI/CD Tetikleniyor)..."
git add src/weights.rs

# Eğer dosyada değişiklik yoksa (zaten en iyisiyse) hata fırlatmasını önler
if git diff --staged --quiet; then
    echo "⚠️ Yeni DNA mevcut DNA ile aynı. Commit işlemine gerek yok."
else
    git commit -m "chore(model): 🧬 inject new alpha DNA [Fitness: $FITNESS, PnL: $PNL]"
    git push origin main
    echo "🦅 GÖREV TAMAMLANDI! GitHub Actions şu an yeni imajı derliyor."
fi