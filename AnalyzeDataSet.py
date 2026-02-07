import pandas as pa
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Läs data
data = pa.read_csv('training_data_VT2026.csv')

# Målkolumn = sista kolumnen
if len(data.columns) == 0:
    raise ValueError('Data är tom.')
target_col = data.columns[-1]

print(f"\n{'='*60}")
print(f"ANALYS AV CYKELKAPACITET ({target_col})")
print(f"{'='*60}\n")

# Visa målkolumnens fördelning
print(f"Målkolumn: {target_col}")
print(f"Unika värden: {data[target_col].unique()}")
print(f"Fördelning:\n{data[target_col].value_counts()}\n")

# Konvertera målkolumn till numerisk (0/1) för korrelationsberäkning
# low_bike_demand = 0, high_bike_demand = 1
target_mapping = {'low_bike_demand': 0, 'high_bike_demand': 1}
data['target_numeric'] = data[target_col].map(target_mapping)

if data['target_numeric'].isna().any():
    print("VARNING: Okända värden i målkolumnen!")

# Välj numeriska kolumner (exkludera mål och target_numeric)
numeric_cols = data.select_dtypes(include='number').columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in [target_col, 'target_numeric']]

# Ta bort konstanta/nästan konstanta kolumner
variance_threshold = 0.01
numeric_cols = [col for col in numeric_cols if data[col].var() > variance_threshold]

print(f"Analyserar {len(numeric_cols)} numeriska features:")
for col in numeric_cols:
    print(f"  - {col}")
print()

# ========== 1. FEATURE IMPORTANCE: KORRELATION & ANOVA ==========
print(f"\n{'='*60}")
print("1. FEATURE IMPORTANCE ANALYS")
print(f"{'='*60}\n")

# Beräkna point-biserial korrelation (Pearson med binär target)
correlations = {}
f_statistics = {}
p_values = {}

for col in numeric_cols:
    # Korrelation
    corr = data[col].corr(data['target_numeric'])
    correlations[col] = corr
    
    # ANOVA F-statistik
    low_group = data[data[target_col] == 'low_bike_demand'][col].dropna()
    high_group = data[data[target_col] == 'high_bike_demand'][col].dropna()
    f_stat, p_val = stats.f_oneway(low_group, high_group)
    f_statistics[col] = f_stat
    p_values[col] = p_val

# Skapa DataFrame för resultat
importance_df = pa.DataFrame({
    'Feature': numeric_cols,
    'Correlation': [correlations[col] for col in numeric_cols],
    'F-statistic': [f_statistics[col] for col in numeric_cols],
    'p-value': [p_values[col] for col in numeric_cols]
})
importance_df['Abs_Correlation'] = importance_df['Correlation'].abs()
importance_df = importance_df.sort_values('Abs_Correlation', ascending=False)

print("Top 10 features efter korrelation:")
print(importance_df.head(10).to_string(index=False))
print()

# Plotta korrelation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Korrelation
importance_df_sorted = importance_df.sort_values('Correlation')
colors = ['red' if x < 0 else 'green' for x in importance_df_sorted['Correlation']]
axes[0].barh(importance_df_sorted['Feature'], importance_df_sorted['Correlation'], color=colors, alpha=0.7)
axes[0].set_xlabel('Point-Biserial Correlation')
axes[0].set_title('Feature Correlation med high_bike_demand\n(Grön = positiv, Röd = negativ)')
axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].grid(axis='x', alpha=0.3)

# F-statistik
importance_df_f = importance_df.sort_values('F-statistic', ascending=True)
axes[1].barh(importance_df_f['Feature'], importance_df_f['F-statistic'], color='steelblue', alpha=0.7)
axes[1].set_xlabel('F-statistic (ANOVA)')
axes[1].set_title('Feature Importance (F-statistic)\nHögre = större skillnad mellan grupperna')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 2. BOXPLOTS FÖR TOP FEATURES ==========
print(f"\n{'='*60}")
print("2. BOXPLOTS FÖR TOP 8 FEATURES")
print(f"{'='*60}\n")

top_n = min(8, len(importance_df))
top_features = importance_df.head(top_n)['Feature'].tolist()

fig, axes = plt.subplots(nrows=(top_n + 1) // 2, ncols=2, figsize=(12, 4 * ((top_n + 1) // 2)))
axes = axes.flatten() if top_n > 1 else [axes]

for i, col in enumerate(top_features):
    data.boxplot(column=col, by=target_col, ax=axes[i])
    axes[i].set_title(f'{col}\n(corr={correlations[col]:.3f}, F={f_statistics[col]:.1f})')
    axes[i].set_xlabel('')
    axes[i].set_ylabel(col)
    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Stäng av tomma axlar
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Fördelning av features per cykelkapacitet', fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# ========== 3. PAIRPLOT MED FÄRGKODNING ==========
print(f"\n{'='*60}")
print("3. PAIRPLOT FÖR TOP 4 FEATURES")
print(f"{'='*60}\n")

top_4_features = importance_df.head(4)['Feature'].tolist()
pairplot_data = data[top_4_features + [target_col]].copy()

sns.pairplot(
    pairplot_data,
    hue=target_col,
    palette={'low_bike_demand': 'red', 'high_bike_demand': 'green'},
    diag_kind='kde',
    plot_kws={'alpha': 0.6},
    height=2.5
)
plt.suptitle('Pairplot: Top 4 features färgade efter cykelkapacitet', y=1.02)
plt.show()

# ========== 4. SAMMANFATTNING ==========
print(f"\n{'='*60}")
print("SAMMANFATTNING & INSIKTER")
print(f"{'='*60}\n")

print("Top 5 features som starkast korrelerar med hög cykelkapacitet:")
for idx, row in importance_df.head(5).iterrows():
    direction = "ökar" if row['Correlation'] > 0 else "minskar"
    significance = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
    print(f"  {row['Feature']:20s}: corr={row['Correlation']:+.3f} {significance:3s} - När {row['Feature']} {direction}, ökar sannolikheten för hög cykelkapacitet")

print("\n" + "="*60)
print("NÄSTA STEG FÖR MODELLERING:")
print("="*60)
print("1. Använd top features identifierade ovan som input till din modell")
print("2. Överväg att ta bort features med mycket låg korrelation (< 0.05)")
print("3. Testa logistic regression, random forest eller gradient boosting")
print("4. Feature engineering: skapa interaktioner (t.ex. temp*hour_of_day)")
print("5. Validera modellen med cross-validation")
print(f"{'='*60}\n")
