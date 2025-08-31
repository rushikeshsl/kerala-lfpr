import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
import seaborn as sns 
import streamlit as st  

st.title("Labor Force Participation in Kerala: A Logistic Regression Analysis")

st.header("1. Project Overview")

st.markdown("""
This project investigates the factors influencing labor force participation (LFPR) 
in the Indian state of Kerala. Despite high human development indicators, including a 
literacy rate above 94%, Kerala faces a notable challenge: low female labor force participation.

Using logistic regression, the study examines how demographic and socio-economic 
variables—such as age, gender, marital status, education, and sector (urban/rural) —
affect an individual’s likelihood of being in the workforce.
""")

st.header("2. Data Overview")

st.subheader("2.1 Data Source")
st.write("""The data for this analysis comes from the Periodic Labour Force Survey (PLFS) for the year July 2023 - 2024, which is conducted by the National Sample Survey Office (NSSO) in India.""")

st.subheader("2.2 Variables")
st.markdown("##### Dependent Variable")
st.markdown("""
* **Labour Force Participation (LFPR):** A binary variable (1/0) that indicates whether an individual is in the labor force.
""")

st.markdown("##### Independent Variables")
st.markdown("""
* **Sex:** A categorical variable coded as (Male: 0 and Female: 1) to analyze gender differences in LFPR.
* **Age:** A continuous variable representing the age of the individual.
* **Marital Status:** A categorical variable coded as (Never Married: 0 and Married: 1).
* **Sector:** A categorical variable coded as (Rural: 0 and Urban: 1) to capture differences between geographical areas.
* **No of Formal Education:** A continuous variable representing the number of years of formal education an individual has completed.
""")

st.subheader("2.3 Dataset")
kerala = pd.read_csv("kerala.csv",index_col=0)
st.dataframe(kerala)

st.header('3. Exploratory Data Analysis (EDA)')

st.subheader('3.1 Labor Force Participation Rate by Different Variables')
palette=['#3498db', '#e74c3c']

st.markdown("##### Labor Force Participation Rate (LFPR): Male vs Female")
sex_lfpr = kerala.groupby('Sex')['LFPR'].mean()
fig_sex, ax_sex = plt.subplots(figsize=(10,6))
sns.barplot(x=sex_lfpr.index, y=sex_lfpr.values, ax=ax_sex, palette=palette)
ax_sex.set_xticklabels(['Male', 'Female'])
ax_sex.set_title('LFPR by Sex')
ax_sex.set_ylabel('LFPR')
for i, v in enumerate(sex_lfpr.values):
    ax_sex.text(i, v - 0.01, f"{v:.2f}", ha='center', va='top', fontweight='bold', color='white', fontsize=12)
st.pyplot(fig_sex)
st.markdown("""
The LFPR for males is approximately **58.3%**, which is significantly higher than the LFPR for females, which stands at about **32.2%**. This highlights a substantial 
gender gap in labor force participation in Kerala, indicating that males are more likely to be part of the workforce than females.
""")

st.markdown("##### Labor Force Participation Rate (LFPR): Rural vs Urban")
sector_lfpr = kerala.groupby('Sector')['LFPR'].mean()
fig_sector, ax_sector = plt.subplots(figsize=(10, 6))
sns.barplot(x=sector_lfpr.index, y=sector_lfpr.values, ax=ax_sector, palette=palette)
ax_sector.set_xticklabels(['Rural', 'Urban'])
ax_sector.set_title('LFPR by Sector')
ax_sector.set_ylabel('LFPR')
for i, v in enumerate(sector_lfpr.values):
    ax_sector.text(i, v - 0.01, f"{v:.2f}", ha='center', va='top', fontweight='bold', color='white', fontsize=12)
st.pyplot(fig_sector)
st.markdown("""The LFPR in rural areas is approximately **45.8%**, which is only marginally higher than the LFPR in urban areas, which is around **44.9%**. This suggests that 
while there is a slight difference, the overall rate of labor force participation is relatively consistent across both rural and urban sectors in Kerala.""")

st.markdown("##### Labor Force Participation Rate (LFPR): Never Married vs Married")
marital_lfpr = kerala.groupby('Marital_Status')['LFPR'].mean()
fig_marital, ax_marital = plt.subplots(figsize=(10, 6))
sns.barplot(x=marital_lfpr.index, y=marital_lfpr.values, ax=ax_marital, palette=palette)
ax_marital.set_xticklabels(['Never Married', 'Married'])
ax_marital.set_title('LFPR by Marital Status')
ax_marital.set_ylabel('LFPR')
for i, v in enumerate(marital_lfpr.values):
    ax_marital.text(i, v - 0.01, f"{v:.2f}", ha='center', va='top', fontweight='bold', color='white', fontsize=12)
st.pyplot(fig_marital)
st.markdown("""There is a significant difference in LFPR based on marital status. The LFPR for married individuals is approximately **62.2%**, which is much greater than the 
**20.8%** LFPR for never-married individuals. This pattern may reflect social dynamics where getting married is often associated with taking on economic responsibilities and 
entering the workforce, or it could be due to demographic factors where individuals tend to marry at an age where they are already in the labor force.""")

st.subheader('3.2 Labor Force Participation Rate by Sex and Sector')
interxn = pd.crosstab(
    kerala['Sex'], kerala['Sector'], values=kerala['LFPR'], aggfunc='mean'
)
interxn.index = ['Male', 'Female']
interxn.columns = ['Rural', 'Urban']
st.table(interxn)
st.markdown("""Based on the data, Kerala's labor force participation reveals a significant gender gap, with male LFPR consistently high and nearly identical in both rural **(58.0%)** 
and urban **(58.6%)** areas. The most notable trend is the higher female LFPR in rural areas **(33.4%)** compared to urban areas **(31.1%)**. This rural-urban difference is primarily driven by 
the greater prevalence of agricultural and informal work in the countryside, where women's participation in family farms and casual labor is counted. In contrast, urban women, despite
having high educational attainment, often face a mismatch between their job expectations and the availability of suitable formal sector jobs, leading to a "discouraged worker" effect. 
Additionally, rising household income, often from remittances, can lead some urban women to withdraw from the workforce, a phenomenon less pronounced in rural settings.""")


st.header('4. Logistic Regression Analysis')
st.subheader('4.1 Without Interaction Terms')
st.latex(r"\text{LFPR}_i = \beta_0 + \beta_1 \cdot \text{Sector}_i + \beta_2 \cdot \text{Sex}_i + \beta_3 \cdot \text{Age}_i + \beta_4 \cdot \text{Marital\_Status}_i + \beta_5 \cdot \text{Year\_Edu}_i")

model = logit('LFPR ~ Sector + Sex + Age + Marital_Status + Year_Edu', data = kerala).fit()

st.markdown("#### Logistic Regression Summary")
st.code(model.summary().as_text(), language='text')

st.markdown("#### Interpretation of Logistic Regression Coefficients")

st.markdown("##### Model Overview")
st.write("""
- **Model Type:** Logit (Logistic Regression)  
- **Dependent Variable:** LFPR (Labor Force Participation Rate)  
- **Observations:** 14,028  
- **Pseudo R²:** 0.294 (moderate explanatory power)  
- **Significance:** All variables significant at p < 0.001
""")

st.markdown("##### Intercept (-2.483)")
st.write("""Represents the baseline log-odds of labor force participation for an individual who is male, lives in a rural area, is never married, has age 0, and has 0 years of education.
Although these baseline values are hypothetical (especially age and education = 0), the intercept is necessary for estimating the log-odds scale.
""")

st.markdown("##### Sex (0 = Male, 1 = Female; β = -1.908)")
st.write("""Being female is associated with a substantial reduction in the likelihood of participating in the labor force compared to males. \n
**Odds interpretation:** exp(-1.908) ≈ 0.148 → females have 85% lower odds of being in the labor force compared to males. \n
**Implication:** Gender remains a major determinant of LFPR in Kerala.
""")

st.markdown("##### Sector (0 = Rural, 1 = Urban; β = -0.292)")
st.write("""Living in an urban area slightly decreases the likelihood of labor force participation relative to rural residents. \n
**Odds interpretation:** exp(-0.292) ≈ 0.747 → urban residents have ~25% lower odds of participating. \n
**Implication:** This might reflect limited participation opportunities for women in urban areas or a higher prevalence of non-working populations in cities.
""")

st.markdown("##### Age (β = 0.0127)")
st.write("""Each additional year of age slightly increases the likelihood of being in the labor force. \n
**Odds interpretation:** exp(0.0127) ≈ 1.013 → a 1-year increase in age raises odds of participation by ~1.3%. \n
**Implication:** Labor force participation gradually increases with age, possibly due to experience, maturity, or family responsibilities.
""")

st.markdown("##### Marital Status (0 = Never Married, 1 = Married; β = 1.759)")
st.write("""Being married significantly increases the likelihood of participating in the labor force. \n
**Odds interpretation:** exp(1.759) ≈ 5.8 → married individuals are ~5.8 times more likely to participate than those never married. \n
**Implication:** Marriage may create financial responsibilities, motivating higher participation, especially among females. \n
""")

st.markdown("##### Years of Education (β = 0.189)")
st.write("""Each additional year of education increases the likelihood of labor force participation.\n
**Odds interpretation:** exp(0.189) ≈ 1.21 → each extra year of education raises odds of participation by ~21%. \n
**Implication:** Education positively impacts employability and engagement in the labor market.
""")

st.markdown(""" #### Overall Interpretation""")
st.markdown("""The logistic regression analysis reveals several key factors influencing labor force participation (LFPR) in Kerala. Gender 
emerges as a significant determinant, with females substantially less likely to participate compared to males, reflecting a persistent gender gap. 
Urban residents are slightly less likely to be in the labor force than those in rural areas, suggesting differences in employment opportunities or 
social norms. Age and education positively impact participation, indicating that older individuals and those with higher educational attainment are 
more likely to engage in economic activities. Marital status also plays an important role, as married individuals are considerably more likely to 
participate than those never married, highlighting the influence of social and familial responsibilities. Overall, the model explains a moderate 
portion of the variation in LFPR (Pseudo R² = 0.294), indicating that while these factors are important, other socio-economic and cultural variables 
may also contribute to labor force participation in Kerala.""")

