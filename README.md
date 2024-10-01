# Lux-Marchesi model
Project for *Econophysics*, University of Leiden, A.Y. 2024/25

---
The Lux-Marchesi model is a computational agent-based model that simulates the dynamics of financial markets. Developed by Thomas Lux and Michele Marchesi, it is based on interactions between heterogeneous agents, namely "fundamentalists" and "chartists," whose behaviors contribute to price fluctuations. 

- **Fundamentalists** believe that asset prices will eventually revert to their fundamental value, so they trade based on the discrepancy between the current price and this intrinsic value. 
- **Chartists**, on the other hand, base their trades on trends and patterns in historical price movements, either following upward momentum (optimistic chartists) or betting against it (pessimistic chartists).

The model incorporates agent switching, meaning that individuals can change their trading strategy probabilistically based on market sentiment, risk perceptions, or external factors. This resembles a contagious process in the framework of compartmental models. The collective behavior of these agents leads to realistic features of financial markets, such as price bubbles, crashes, and volatility clustering. The Lux-Marchesi model captures key statistical properties observed in real markets, like fat-tailed distributions of returns and volatility persistence, making it useful for studying market dynamics and systemic risk.

## References
**[1]** Lux, T. and Marchesi, M., 1999. Scaling and criticality in a stochastic multi-agent model of a financial market. *Nature*, 397(6719), pp.498-500. <br>
**[2]** Lux, T., 1998. The socio-economic dynamics of speculative markets: interacting agents, chaos, and the fat tails of return distributions. *Journal of Economic Behavior & Organization*, 33(2), pp.143-165. <br>
**[3]** Lux, T. and Marchesi, M., 2000. Volatility clustering in financial markets: a microsimulation of interacting agents. *International journal of theoretical and applied finance*, 3(04), pp.675-702. <br>
