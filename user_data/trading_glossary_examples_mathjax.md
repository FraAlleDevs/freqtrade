# Trading Glossary with Examples

## 📊 Profit and Loss
- **Gross Profit/Loss** = before fees are subtracted.  
- **Net Profit/Loss** = gross − fees.

**Example**  
Buy at \$10{,}000 → sell at \$11{,}000.  
Gross profit: \( +\$1{,}000 \) ( \( +10\% \) ).  
If fees are \( 0.1\% \) per side (buy + sell = \(0.2\%\) total):  
Fees \(= 0.002 \times 11{,}000 = \$22\).  
Net profit \(= 1{,}000 - 22 = \$978\).

---

## 📈 CAGR (Compound Annual Growth Rate)
**Definition**  
\[
\mathrm{CAGR} \;=\; \left(\frac{\text{Ending Balance}}{\text{Starting Balance}}\right)^{\tfrac{1}{\text{Years}}} \;-\; 1
\]

**Example**  
Start \$1{,}000 → After 2 years \$1{,}210:  
\[
\mathrm{CAGR} \;=\; \left(\frac{1210}{1000}\right)^{\tfrac{1}{2}} - 1 \;\approx\; 0.10 \;=\; \mathbf{10\%/year}
\]

---

## 📉 Drawdown
**Definition (peak-to-trough)**  
\[
\mathrm{DD} \;=\; \frac{\text{Peak} - \text{Trough}}{\text{Peak}}
\]

**Example**  
Equity peaks at \$1{,}500 then falls to \$1{,}200:  
\[
\mathrm{DD} \;=\; \frac{1500 - 1200}{1500} = 0.20 \;=\; \mathbf{20\%}
\]

---

## 📊 Sharpe Ratio
**Definition (risk‑adjusted return)**  
\[
S \;=\; \frac{\mu - r_f}{\sigma}
\]
where \( \mu \) = average periodic return, \( r_f \) = risk‑free rate, \( \sigma \) = std. dev. of periodic returns.

**Example (daily)**  
Average daily return \( \mu = 1\% \), std dev \( \sigma = 0.5\% \), risk‑free \( r_f \approx 0\% \):  
\[
S \approx \frac{1\% - 0\%}{0.5\%} = 2.0 \;\Rightarrow\; \text{good risk‑adjusted performance.}
\]

---

## 🎯 Expectancy
**Definition (per trade)**  
\[
E \;=\; p_w \cdot \overline{W} \;-\; p_\ell \cdot \overline{L}
\]
where \( p_w \) = win rate, \( p_\ell = 1 - p_w \) = loss rate, \( \overline{W} \) = avg win %, \( \overline{L} \) = avg loss % (use positive numbers).

**Example**  
\( p_w = 0.60,\; \overline{W} = 5\%,\; \overline{L} = 3\% \)  
\[
E \;=\; 0.60 \cdot 5\% \;-\; 0.40 \cdot 3\% \;=\; 3.0\% - 1.2\% \;=\; \mathbf{+1.8\% \;per\; trade}
\]

---

## 📈 Profit Factor (PF)
**Definition**  
\[
\mathrm{PF} \;=\; \frac{\text{Gross Profit}}{\text{Gross Loss}}
\]

**Example**  
Gross profit = \$5{,}000, Gross loss = \$2{,}500  
\[
\mathrm{PF} \;=\; \frac{5000}{2500} \;=\; \mathbf{2.0}
\]
Interpretation: PF > 1 = profitable; PF < 1 = losing.
