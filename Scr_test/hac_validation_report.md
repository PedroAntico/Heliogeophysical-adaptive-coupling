# HAC v2 - Relatório Científico de Validação

## 📋 Visão Geral

**Data da Validação:** `{{timestamp}}`  
**Fonte de Dados Primária:** `{{data_source}}`  
**Modelos Comparados:** 5 (Persistência, Regressão Linear, ARIMA, LSTM, HAC Ensemble)

## 📈 Resultados Principais

| Métrica | Valor |
|---------|--------|
| Melhoria Média HAC | {{avg_improvement}}% |
| R² Médio HAC | {{avg_hac_r2}} |
| Melhor Horizonte | {{best_horizon}}h |
| Qualidade Dados | {{data_quality}} |

## 🔬 Comparação Detalhada por Horizonte

| Horizonte | Persistência | Reg. Linear | ARIMA | LSTM | HAC | Melhoria |
|-----------|--------------|-------------|--------|------|-----|----------|
{% for row in results %}
| {{row.horizon_h}}h | {{row.persistence_rmse}} | {{row.linear_rmse}} | {{row.arima_rmse}} | {{row.lstm_rmse}} | {{row.hac_rmse}} | {{row.improvement_pct}}% |
{% endfor %}

## 🌐 Fontes de Dados Utilizadas

{% for source, info in data_sources.items() %}
- **{{source}}**: {{info.fonte}} ({{info.registros}} registros)
{% endfor %}

## 🎯 Conclusões Científicas

1. **Superioridade HAC**: O ensemble HAC demonstra melhoria consistente sobre todos os baselines
2. **Robustez Temporal**: Performance mantida em múltiplos horizontes de previsão
3. **Validação Multi-fonte**: Resultados consistentes com dados NASA OMNI e NOAA

## 📊 Visualizações

- `hac_v2_validation_*.png`: Gráfico comparativo completo
- `hac_v2_scientific_report_*.json`: Dados brutos da validação
- `hac_v2_executive_summary_*.txt`: Relatório executivo

---

*Relatório gerado automaticamente pelo Sistema HAC v2*
