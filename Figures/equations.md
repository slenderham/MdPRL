$$
\begin{align}
V_{F/C/O}(t+1) &= V_{F/C/O}(t) + A_{F/C/O}(\alpha_{rew}\delta_{R_t=1}+\alpha_{unr}\delta_{R_t=0})(R_t-V(t))\\\ \\
\text{logit } P(O_i) &= \beta\times\begin{cases} \sum_F\ A_FV_{F_i} & \text{Feature-based model}\\
\omega\sum_F\  A_FV_{F_i}+(1-\omega) \sum_C\ A_CV_{C_i} & \text{Conjunction-based model}\\
\omega\sum_F\  A_FV_{F_i}+(1-\omega) V_{O_i} & \text{Object-based model}
\end{cases}\\\ \\
A_{F/C}(O_1, O_2)\ &= \frac{\exp[\gamma\ H(V_{F_1/C_1}, V_{F_2/C_2})\ ]}{\sum_{\hat{F}/\hat{C}}\exp[\gamma\ H(V_{\hat{F}_1/\hat{C}_1}, V_{\hat{F}_2/\hat{C}_2})\ ]},\ \ \  H\in{\{\text{zero, sum, diff, max}\}}\\\ \\ 
H_{\text{diff tied}}(V_{F_1/C_1}, V_{F_2/C_2}) &=\ \mid\omega\times (V_{F_1}-V_{F_2})+(1-\omega)\times (V_{C_1}-V_{C_2})\mid
\end{align}
$$