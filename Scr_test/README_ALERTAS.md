# 🚨 Sistema de Alerta Precoce - Eventos Solares Classe X

## 📋 Funcionalidades

- **Detecção Automática** de condições precursoras de eventos classe X
- **Avaliação de Impactos** em infraestruturas críticas
- **Alertas em Tempo Real** via GitHub Actions
- **Visualizações Científicas** para análise de riscos

## ⚠️ Limiares de Alerta

| Tipo de Alerta | Velocidade | Bz GSM | Densidade | Impacto Esperado |
|----------------|------------|---------|-----------|------------------|
| Classe X | >600 km/s | >15 nT | >20 p/cc | Blackouts de rádio, satélites |
| Temp. Radiação | >700 km/s | - | >25 p/cc | Radiação aumentada, aviação |
| Temp. Geomag. | - | >20 nT | - | Redes elétricas, auroras |

## 📊 Saídas do Sistema

- `results/impact_forecast_*.json` - Relatório completo de riscos
- `plots/impact_forecast.png` - Visualização de alertas
- Logs detalhados em `logs/impact_predictor.log`

## 🔧 Como Usar

```bash
# Análise manual de impactos
python src/impact_predictor.py

# Verificar alertas ativos
cat results/impact_forecast_*.json | jq '.risk_assessment'
